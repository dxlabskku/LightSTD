import numpy as np
import torch


def masked_metric(agg_fn, error_fn, pred, target, null_value=0.0, agg_dim=0):
    mask = (target != null_value).float()
    target_ = target.clone()
    target_[mask == 0.0] = 1.0  # for mape
    mask /= torch.mean(mask, dim=agg_dim, keepdim=True)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    score = error_fn(pred, target_)
    score = score*mask
    # score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
    return agg_fn(score)


def masked_MAE(pred, target, null_value=0.0, agg_dim=(0, 1, 2)):
    mae = masked_metric(agg_fn=lambda e: torch.mean(e, dim=agg_dim),
                        error_fn=lambda p, t: torch.absolute(p - t),
                        pred=pred, target=target, null_value=null_value, agg_dim=agg_dim)
    return mae


def masked_MSE(pred, target, null_value=0.0, agg_dim=(0, 1, 2)):
    mse = masked_metric(agg_fn=lambda e: torch.mean(e, dim=agg_dim),
                        error_fn=lambda p, t: (p - t) ** 2,
                        pred=pred, target=target, null_value=null_value, agg_dim=agg_dim)
    return mse


def masked_RMSE(pred, target, null_value=0.0, agg_dim=(0, 1, 2)):
    rmse = masked_metric(agg_fn=lambda e: torch.sqrt(torch.mean(e, dim=agg_dim)),
                         error_fn=lambda p, t: (p - t)**2,
                         pred=pred, target=target, null_value=null_value, agg_dim=agg_dim)
    return rmse


def masked_MAPE(pred, target, null_value=0.0, agg_dim=(0, 1, 2)):
    mape = masked_metric(agg_fn=lambda e: torch.mean(torch.absolute(e) * 100, dim=agg_dim),
                         error_fn=lambda p, t: ((p - t) / (t)),
                         pred=pred, target=target, null_value=null_value, agg_dim=agg_dim)
    return mape


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points=None):
    """
    target: (B, T, V), torch.Tensor
    forecast: (B, n_sample, T, V), torch.Tensor
    eval_points: (B, T, V): which values should be evaluated,
    """
    eval_points = torch.ones_like(target)

    # target = target * scaler + mean_scaler
    # forecast = forecast * scaler + mean_scaler
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


class Metric:
    def __init__(self, criteria="mae", patience=10):
        self.metrics = []
        self.best_metrics = {"epoch": np.inf, "mae": np.inf, "rmse": np.inf, "mape": np.inf, "crps": np.inf, "loss": np.inf}
        self.criteria = criteria
        self.patience = patience
        
    def __call__(self, y_true, y_pred_median, y_pred=None, loss=None, epoch=None, valid=True):
        if y_pred == None:
            y_pred = y_pred_median.unsqueeze(dim=1)

        metrics = {}
        metrics["epoch"] = epoch
        metrics["mae"] = masked_MAE(y_pred_median, y_true)
        metrics["rmse"] = masked_RMSE(y_pred_median, y_true)
        metrics["mape"] = masked_MAPE(y_pred_median, y_true)
        metrics["crps"] = calc_quantile_CRPS(y_true, y_pred)
        metrics["loss"] = loss

        if valid:
            self.metrics.append(metrics)

        if metrics[self.criteria] < self.best_metrics[self.criteria]:
            self.best_metrics = metrics
        
        return metrics

    def check_early_stop(self, epoch):
        if epoch - self.best_metrics["epoch"] > self.patience:
            return True
        return False

    def to_list(self):
        epoch = []
        mae = []
        rmse = []
        mape = []
        crps = []
        loss = []

        for metrics in self.metrics:
            epoch.append(metrics["epoch"])
            mae.append(metrics["mae"])
            rmse.append(metrics["rmse"])
            mape.append(metrics["mape"])
            crps.append(metrics["crps"])
            loss.append(metrics["loss"])
        
        return epoch, mae, rmse, mape, crps, loss
