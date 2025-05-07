import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, LGConv

import math
from typing import List


class DiffTimeEmbedding(nn.Module):
    def __init__(self, time_dim: List[int], max_steps: int = 500):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(time_dim[0], max_steps), persistent=False
        )
        self.projection1 = nn.Linear(time_dim[0] * 2, time_dim[1])
        self.projection2 = nn.Linear(time_dim[1], time_dim[1])

    def forward(self, time: torch.Tensor):
        x = self.embedding[time]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, time_dim: int, max_steps: int):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        embedding = torch.arange(time_dim).unsqueeze(0)  # [1,dim]
        embedding = steps * 10.0 ** (embedding * 4.0 / time_dim)  # [T,dim]
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
        return embedding


class Projection(nn.Module):
    def __init__(self, in_channels: int, hid_channels: int, out_channels: int):
        super(Projection, self).__init__()
        self.fc1 = nn.Linear(in_channels, hid_channels)
        self.fc2 = nn.Linear(hid_channels, out_channels)
    
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class GatedConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(GatedConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 2 * out_channels, (1, 1))
    
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        gate, x = torch.chunk(x, 2, dim=1)
        x = torch.sigmoid(gate) * x
        return x


class CondBlock(nn.Module):
    def __init__(
            self,
            num_nodes: int,
            periods: int,
            hidden_dim: int,
    ):
        super(CondBlock, self).__init__()

        self.g_conv = LGConv()
        self.g_norm = nn.LayerNorm([num_nodes, hidden_dim])
        self.t_conv = nn.Conv2d(periods, periods, (1, 1))
        self.t_norm = nn.LayerNorm([num_nodes, hidden_dim])
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.g_norm(x)
        x = self.g_conv(x, edge_index)
        x = self.t_norm(x)
        x = self.t_conv(x)
        return x


class NoiseBlock(nn.Module):
    def __init__(
            self,
            num_nodes: int,
            periods: int,
            hidden_dim: int,
            time_dim: int,
    ):
        super(NoiseBlock, self).__init__()

        self.time_projection = Projection(time_dim, hidden_dim, hidden_dim)
        self.cond_projection = Projection(hidden_dim, hidden_dim, hidden_dim)

        self.g_conv = LGConv()
        self.g_norm = nn.LayerNorm([num_nodes, hidden_dim])
        self.t_conv = nn.Conv2d(periods, periods, (1, 1))
        self.t_norm = nn.LayerNorm([num_nodes, hidden_dim])
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, time: torch.Tensor, cond: torch.Tensor):
        time = self.time_projection(time)[:, None, None, :]
        cond = self.cond_projection(cond)

        x += cond
        x = self.g_norm(x)
        x = self.g_conv(x, edge_index)
        x += time
        x = self.t_norm(x)
        x = self.t_conv(x)

        return x


class CondNet(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_nodes: int,
            periods: int,
            num_blocks: int,
    ):
        super(CondNet, self).__init__()

        self.input_projection = Projection(input_dim, output_dim, output_dim)
        self.pos_projection = Projection(output_dim, output_dim, output_dim)
        self.attention = nn.Parameter(torch.randn(num_blocks + 1, 1))

        self.blocks = nn.ModuleList([CondBlock(num_nodes, periods, output_dim) for _ in range(num_blocks)])

        self.output_projection = Projection(output_dim, output_dim, output_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pos: torch.Tensor):
        B, P = x.shape[0], x.shape[1]
        
        x = self.input_projection(x)
        pos = self.pos_projection(pos)
        x += pos.repeat(B, P, 1, 1)

        skip = [x]
        for block in self.blocks:
            x = block(x, edge_index)
            skip.append(x)
        
        x = torch.stack(skip, dim=-1)
        attention = torch.softmax(self.attention, dim=0)
        x = torch.matmul(x, attention).squeeze()

        x = self.output_projection(x)

        return x


class NoiseNet(nn.Module):
    def __init__(
            self,
            num_nodes: int,
            periods: int,
            hidden_dim: int = 8,
            time_dim: List[int] = [16, 64],
            num_blocks: int = 8,
    ):
        super(NoiseNet, self).__init__()
        self.diff_time_embedding = DiffTimeEmbedding(time_dim=time_dim)
        self.input_projection = Projection(1, hidden_dim, hidden_dim)
        self.pos_projection = Projection(hidden_dim, hidden_dim, hidden_dim)
        self.attention = nn.Parameter(torch.randn(num_blocks + 1, 1))
        self.blocks = nn.ModuleList(
            [NoiseBlock(num_nodes, periods, hidden_dim, time_dim[-1]) for _ in range(num_blocks)]
        )
        self.output_projection = Projection(hidden_dim, hidden_dim, 1)
    
    def forward(
            self,
            x: torch.Tensor,
            time: torch.Tensor,
            cond: torch.Tensor,
            pos: torch.Tensor,
            edge_index: torch.Tensor
        ) -> torch.Tensor:
        B, P = x.shape[0], x.shape[1]

        time = self.diff_time_embedding(time)

        x = self.input_projection(x[..., None])
        pos = self.pos_projection(pos)
        x += pos.repeat(B, P, 1, 1)
        
        skip = [x]
        for block in self.blocks:
            x = block(x, edge_index, time, cond)
            skip.append(x)
        
        x = torch.stack(skip, dim=-1)
        attention = torch.softmax(self.attention, dim=0)
        x = torch.matmul(x, attention).squeeze()
        x = self.output_projection(x)
        return x.squeeze()
