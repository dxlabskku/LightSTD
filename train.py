import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric_temporal.dataset import METRLADatasetLoader, PemsBayDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from ema_pytorch import EMA

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import hydra
import logging
import os

from metric import Metric
from model import LightSTD


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(f"{path}/checkpoints", exist_ok=True)
    os.makedirs(f"{path}/plots", exist_ok=True)

    if cfg.dataset == 'metrla':
        dataset = METRLADatasetLoader("data").get_dataset(num_timesteps_in=cfg.time_step, num_timesteps_out=cfg.time_step)

        cfg.num_nodes = 207
        means = [53.59967,0.4982691]
        stds = [20.209862,0.28815305]
    elif cfg.dataset == 'pemsbay':
        dataset = PemsBayDatasetLoader("data").get_dataset(num_timesteps_in=cfg.time_step, num_timesteps_out=cfg.time_step)

        cfg.num_nodes = 325
        means = [61.77375,0.4984733]
        stds = [9.293026,0.28541598]

    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
    train_dataset, valid_dataset = temporal_signal_split(train_dataset, train_ratio=0.9)

    def batch_dataset(dataset, batch_size, shuffle=False):
        history = np.array(dataset.features)
        future = np.array(dataset.targets)

        if future.ndim > 3:
            future = future[:, :, 0, :]

        history = torch.from_numpy(history)
        future = torch.from_numpy(future)
        dataset = TensorDataset(history, future)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    train_loader = batch_dataset(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = batch_dataset(valid_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = batch_dataset(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    edge_index = train_dataset[0].edge_index.to(cfg.device)


    net = LightSTD(
        input_dim=cfg.input_dim,
        num_nodes=cfg.num_nodes,
        periods=cfg.time_step,
        num_cond_blocks=cfg.num_cond_blocks,
        num_noise_blocks=cfg.num_noise_blocks,
        diff_steps=cfg.diff_steps,
        loss_type=cfg.loss_type,
        beta_end=cfg.beta_end,
        beta_schedule=cfg.beta_schedule,
        edge_index=edge_index,
        hidden_dim=cfg.hidden_dim,
        eta=cfg.eta,
    ).to(cfg.device)

    ema = EMA(
        net,
        beta=cfg.ema_decay,
        update_after_step=cfg.ema_update_after_step,
        update_every=cfg.ema_update_every,
    ).to(cfg.device)


    optimizer = Adam(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.max_learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=cfg.epochs,
    )

    metric = Metric(criteria=cfg.criteria, patience=cfg.patience * cfg.ema_update_every)

    train_losses = []

    for epoch in range(cfg.epochs):
        cumm_epoch_loss = 0.0
        cumm_epoch_valid_loss = 0.0

        net.train()
        for history, future in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}, Train"):
            history = history.permute(0, 3, 1, 2).to(cfg.device)
            future = future.permute(0, 2, 1).to(cfg.device)

            loss = net(x=future, prev=history)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()
            lr_scheduler.step()

            cumm_epoch_loss += loss.item()
        
        train_loss = cumm_epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        logging.info(f"Epoch {epoch + 1:04d} \t Train Loss {train_loss:.4f}")

        if epoch > cfg.ema_update_after_step and epoch % cfg.ema_update_every == cfg.ema_update_every - 1:
            future_pred = []
            future_true = []
            ema.eval()
            for history, future in tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}, Validation", colour="green"):
                history = history.permute(0, 3, 1, 2).to(cfg.device)
                future = future.permute(0, 2, 1).to(cfg.device)
            
                with torch.no_grad():
                    loss = ema.ema_model(x=future, prev=history)
                    pred = ema.ema_model.sample(prev=history, num_samples=1)
            
                cumm_epoch_valid_loss += loss.item()

                future_pred.append(pred.squeeze().cpu())
                future_true.append(future.cpu())
        
            future_pred = torch.cat(future_pred, dim=0) * stds[0] + means[0]
            future_true = torch.cat(future_true, dim=0) * stds[0] + means[0]

            valid_loss = cumm_epoch_valid_loss / len(valid_loader)
            
            metrics = metric(future_true, future_pred, loss=valid_loss, epoch=epoch)

            logging.info(f"Epoch {metrics['epoch'] + 1:04d} \t Valid Loss {metrics['loss']:.4f} \t Valid MAE {metrics['mae']:.4f} \t Valid RMSE {metrics['rmse']:.4f} \t Valid MAPE {metrics['mape']:.4f} \t Valid CRPS {metrics['crps']:.4f}")

            torch.save(ema.ema_model.state_dict(), f"{path}/checkpoints/params.pth")

            if metric.check_early_stop(epoch): break


    future_pred = []
    future_true = []
    history_true = []

    ema.eval()
    for history, future in tqdm(test_loader, desc="Test", colour="blue"):
        history = history.permute(0, 3, 1, 2).to(cfg.device)
        future = future.permute(0, 2, 1).to(cfg.device)
        pred = ema.ema_model.sample(prev=history, num_samples=cfg.num_samples)
        future_pred.append(pred.cpu())
        future_true.append(future.cpu())
        history_true.append(history[..., 0].cpu())


    future_pred = torch.cat(future_pred, dim=0) * stds[0] + means[0]
    future_true = torch.cat(future_true, dim=0) * stds[0] + means[0]
    history_true = torch.cat(history_true, dim=0) * stds[0] + means[0]
    future_pred_median = torch.median(future_pred, dim=1)[0]

    metrics = metric(future_true, future_pred_median, future_pred, valid=False)

    logging.info(f"Time Step {cfg.time_step}")

    logging.info(f"Masked MAE {metrics['mae']:.4f}")
    logging.info(f"Masked RMSE {metrics['rmse']:.4f}")
    logging.info(f"Masked MAPE {metrics['mape']:.4f}")
    logging.info(f"Masked CRPS {metrics['crps']:.4f}")


    rows = 4
    cols = 4
    q = torch.tensor([0.05, 0.25, 0.75, 0.95])
    fig, axs = plt.subplots(rows, cols, figsize=(24, 24))
    axx = axs.ravel()

    true_sample = torch.cat((history_true[0, :, :16], future_true[0, :, :16]), dim=0).numpy()
    pred_sample = future_pred_median[0, :, :16].numpy()
    quantile = torch.quantile(future_pred[0, :, :, :16], q, dim=0).numpy()


    for dim in range(rows * cols):
        axx[dim].plot(
            range(1, cfg.time_step * 2 + 1),
            true_sample[:, dim],
            label="observations",
        )
        axx[dim].plot(
            range(cfg.time_step + 1, cfg.time_step * 2 + 1),
            pred_sample[:, dim],
            label="Predictions",
            color="green",
        )
        axx[dim].fill_between(
            range(cfg.time_step + 1, cfg.time_step * 2 + 1),
            quantile[0, :, dim],
            quantile[-1, : ,dim],
            color="green",
            alpha=0.1 ** 0.3,
        )
        axx[dim].fill_between(
            range(cfg.time_step + 1, cfg.time_step * 2 + 1),
            quantile[1, :, dim],
            quantile[-2, : ,dim],
            color="green",
            alpha=0.25 ** 0.3,
        )
    axx[0].legend(loc=2, fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{path}/plots/predictions.png")
    plt.clf()

    epoch, mae, rmse, mape, crps, loss = metric.to_list()

    plt.plot(train_losses, label="Train Loss")
    plt.plot(epoch, loss, label="Valid Loss")
    plt.legend(loc="upper right")
    plt.ylim([0, 0.5])
    plt.savefig(f"{path}/plots/loss.png")
    plt.clf()

    plt.plot(epoch, mae, label="Valid MAE")
    plt.plot(epoch, rmse, label="Valid RMSE")
    plt.plot(epoch, mape, label="Valid MAPE")
    plt.plot(epoch, crps, label="Valid CRPS")
    plt.legend(loc="upper right")
    plt.ylim([0, 20])
    plt.savefig(f"{path}/plots/metrics.png")
    plt.clf()


if __name__ == "__main__":
    main()
