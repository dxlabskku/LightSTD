import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import Optional

from network import CondNet, NoiseNet


def extract(arr, idx, x_shape):
    batch_size, *_ = idx.shape
    out = arr.gather(-1, idx)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


class CondUpsampler(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim // 2)
        self.linear2 = nn.Linear(output_dim // 2, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.4)
        return x


class LightSTD(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_nodes: int,
            periods: int,
            edge_index: torch.Tensor,
            dropout: float = 0.1,
            hidden_dim: int = 8,
            num_blocks: int = 8,
            diff_steps: int = 200,
            loss_type: str = "l2",
            beta_end: float = 0.1,
            betas: Optional[torch.Tensor] = None,
            beta_schedule: str = "uniform",
            sample_steps: int = 40,
            eta: float = 0.0,
            type: int = 0,
            **kwargs
            ) -> None:
        super(LightSTD, self).__init__()

        self.num_nodes = num_nodes
        self.diff_steps = diff_steps
        self.loss_type = loss_type
        self.edge_index = edge_index
        self.beta_schedule = beta_schedule
        self.sample_steps = sample_steps
        self.eta = eta

        self.cond_net = CondNet(
            input_dim=input_dim,
            output_dim=hidden_dim,
            num_nodes=num_nodes,
            periods=periods,
            num_blocks=num_blocks,
        )

        self.noise_net = NoiseNet(
            num_nodes=num_nodes,
            periods=periods,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
        )

        self.pos = nn.Parameter(torch.randn(num_nodes, hidden_dim))

        if betas is not None:
            betas = (
                betas.detach().cpu().numpy()
                if isinstance(betas, torch.Tensor)
                else betas
            )
        else:
            if beta_schedule == "uniform":
                betas = np.linspace(1e-4, beta_end, diff_steps)
            elif beta_schedule == "quad":
                betas = np.linspace(1e-4 ** 0.5, beta_end ** 0.5, diff_steps) ** 2
            else:
                raise NotImplementedError(beta_schedule)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
    
    @ torch.inference_mode()
    def generalized_steps(self, x, seq, cond):
        B = x.shape[0]
        seq_next = [-1] + list(seq[:-1])

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((B,), i, device=x.device).long()
            t_next = torch.full((B,), j + 1, device=x.device).long()

            at = extract(self.alphas_cumprod, t, x.shape)
            at_next = extract(self.alphas_cumprod_prev, t_next, x.shape)

            noise = self.noise_net(x=x, time=t, cond=cond, pos=self.pos, edge_index=self.edge_index)

            x0 = (x - noise * (1 - at).sqrt()) / at.sqrt()

            c1 = self.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            x = at_next.sqrt() * x0 + c1 * torch.randn_like(x) + c2 * noise
        
        return x

    @torch.inference_mode()
    def p_sample_loop(self, cond: torch.Tensor) -> torch.Tensor:
        B, P = cond.shape[0], cond.shape[1]

        if self.beta_schedule == "uniform":
            skip = self.diff_steps // self.sample_steps
            seq = range(0, self.diff_steps, skip)
        elif self.beta_schedule == "quad":
            seq = (np.linspace(0, np.sqrt(self.diff_steps * 0.8), self.sample_steps) ** 2)
            seq = list(map(int, seq))
        else:
            raise NotImplementedError

        x = torch.randn((B, P, self.num_nodes), device=cond.device)
        return self.generalized_steps(x, seq, cond)

    @torch.inference_mode()
    def sample(self, prev: torch.Tensor, num_samples: int) -> torch.Tensor:
        B, P = prev.shape[0], prev.shape[1]
        cond = self.cond_net(prev, self.edge_index, self.pos)
        cond = cond.unsqueeze(dim=1).repeat(1, num_samples, 1, 1, 1)

        x_hat = self.p_sample_loop(cond=cond.reshape(B * num_samples, P, self.num_nodes, -1))
        return x_hat.reshape(B, num_samples, P, self.num_nodes)

    def q_sample(self, x_start: torch.Tensor, time: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:return (
            extract(self.sqrt_alphas_cumprod, time, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, time, x_start.shape) * noise
        )
    
    def p_losses(self, x_start: torch.Tensor, cond: torch.Tensor, time: torch.Tensor):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, time=time, noise=noise) # (B, P, N)
        noise_pred = self.noise_net(x=x_noisy, time=time, cond=cond, pos=self.pos, edge_index=self.edge_index)

        if self.loss_type == "l1":
            loss = F.l1_loss(noise_pred, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(noise_pred, noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(noise_pred, noise)
        else:
            raise NotImplementedError()
        
        return loss

    def forward(self, x: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        B, P = x.shape[0], x.shape[1]

        time = torch.randint( # (B,)
            low=0,
            high=self.diff_steps,
            size=(B,),
            device=x.device,
        ).long()

        cond = self.cond_net(prev, self.edge_index, self.pos)

        return self.p_losses(
            x_start=x,
            cond=cond,
            time=time,
        )
