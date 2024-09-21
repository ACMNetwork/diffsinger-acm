from typing import Optional

import torch
import torch.nn as nn
from .multiHeadConv import MultiHeadConv


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
            self,
            dim: int,
            intermediate_dim: int,
            layer_scale_init_value: Optional[float] = None, drop_out: float = 0.0

    ):
        super().__init__()
        # self.dwconv0 = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        # self.dwconv1 = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # self.dwconv2 = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.dwconv1 = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.dwconv2 = MultiHeadConv(dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.act1 = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        self.dropout = nn.Dropout(drop_out) if drop_out > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, ) -> torch.Tensor:
        residual = x
        x = self.dwconv1(x)
        x = self.act1(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
		
        x = self.dwconv2(x)
        x = self.dropout(x)
        x = residual + self.drop_path(x)
        return x

class ConvNeXtOriginalBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
            self,
            dim: int,
            intermediate_dim: int,
            layer_scale_init_value: Optional[float] = None, drop_out: float = 0.0

    ):
        super().__init__()
        self.dwconv0 = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.dwconv1 = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.dwconv2 = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        self.dropout = nn.Dropout(drop_out) if drop_out > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, ) -> torch.Tensor:
        residual = x
        x = self.dwconv0(x)
        x = self.act2(x)
        x = self.dwconv1(x)
        x = self.act1(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
		
        x = self.dwconv2(x)
        x = self.dropout(x)
        x = residual + self.drop_path(x)
        return x


class ConvNeXtDecoder(nn.Module):
    def __init__(
            self, in_dims, out_dims, /, *,
            num_channels=512, num_layers=7, kernel_size=6, dropout_rate=0.2, num_layers_mh_front=3, num_layers_mh_end=2
    ):
        super().__init__()
        self.inconv = nn.Conv1d(
            in_dims, num_channels, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2
        )
        self.conv = nn.ModuleList(
            list(
                ConvNeXtBlock(
                    dim=num_channels, intermediate_dim=num_channels * 4,
                    layer_scale_init_value=1e-6, drop_out=dropout_rate
                ) for _ in range(num_layers_mh_front)
            ) + 
            list(
                ConvNeXtOriginalBlock(
                    dim=num_channels, intermediate_dim=num_channels * 4,
                    layer_scale_init_value=1e-6, drop_out=dropout_rate
                ) for _ in range(num_layers)
            ) +
            list(
                ConvNeXtBlock(
                    dim=num_channels, intermediate_dim=num_channels * 4,
                    layer_scale_init_value=1e-6, drop_out=dropout_rate
                ) for _ in range(num_layers_mh_end)
            )
        )
        self.outconv = nn.Conv1d(
            num_channels, out_dims, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2
        )

    # noinspection PyUnusedLocal
    def forward(self, x, infer=False):
        x = x.transpose(1, 2)
        x = self.inconv(x)
        for conv in self.conv:
            x = conv(x)
        x = self.outconv(x)
        x = x.transpose(1, 2)
        return x
