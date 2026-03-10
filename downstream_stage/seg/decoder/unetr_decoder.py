import torch
from dataclasses import dataclass
from torch import nn
from typing import List, Sequence, Tuple, Union

__all__ = [
    "DecoderConfig",
    "ResidualYellowBlock",
    "UpsampleBlueBlock",
    "DeconvGreenBlock",
    "SegmentationUNETRDecoder",
]


@dataclass
class DecoderConfig:
    in_channels: int = 9
    out_channels: int = 6
    img_size: Sequence[int] = (9, 112, 112)
    feature_size: int = 28
    hidden_size: int = 96
    grid_size: Sequence[int] = (9, 4, 14, 14)
    upsample_kernel_sizes: Sequence[Union[List[int], Tuple[int, ...]]] = (
        (1, 2, 2),
        (2, 2, 2),
        (1, 1, 1),
    )


class ResidualYellowBlock(nn.Module):
    """Residual block that matches the original "yellow" stack."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: nn.Module = nn.InstanceNorm3d,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.downsample = in_channels != out_channels
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            normalization(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            normalization(out_channels),
        ]
        self.conv_block = nn.Sequential(*layers)
        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                normalization(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x) if self.downsample else x
        x = self.conv_block(x)
        x += residual 
        return self.activation(x)


class UpsampleBlueBlock(nn.Module):
    """Upsampling block built from transposed conv followed by residual refinement."""

    def __init__(self, in_channels: int, out_channels: int, normalization: nn.Module, layer_num: int) -> None:
        super().__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.blocks = nn.ModuleList(
            [ResidualYellowBlock(out_channels, out_channels, normalization=normalization)]
            + [ResidualYellowBlock(out_channels, out_channels, normalization=normalization) for _ in range(layer_num - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)
        for block in self.blocks:
            x = block(x)
        return x


class DeconvGreenBlock(nn.Module):
    """Single transposed convolution used in the decoder path."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


class SegmentationUNETRDecoder(nn.Module):
    """UNETR decoder that consumes transformer features + skip connections."""

    def __init__(
        self,
        config: DecoderConfig,
    ) -> None:
        super().__init__()
        self.config = config
        feature_size = config.feature_size
        hidden_size = config.hidden_size
        grid_size = tuple(config.grid_size)
        slice_num = config.img_size[0]
        output_channel = config.out_channels # * slice_num (S is handled in Depth)

        self.z0_yellow_block = ResidualYellowBlock(
            in_channels=config.in_channels, out_channels=feature_size, normalization=nn.InstanceNorm3d
        )
        self.z3_block = UpsampleBlueBlock(
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            normalization=nn.InstanceNorm3d,
            layer_num=2,
        )
        self.z6_block = ResidualYellowBlock(
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            normalization=nn.InstanceNorm3d,
        )
        self.z9_block = ResidualYellowBlock(
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            normalization=nn.InstanceNorm3d,
        )
        self.z6_yellow_block = ResidualYellowBlock(
            in_channels=feature_size * 4 * 2,
            out_channels=feature_size * 4,
            normalization=nn.InstanceNorm3d,
        )
        self.z6_green_block = DeconvGreenBlock(feature_size * 4, feature_size * 2)
        self.z3_yellow_block = ResidualYellowBlock(
            in_channels=feature_size * 2 * 2,
            out_channels=feature_size * 2,
            normalization=nn.InstanceNorm3d,
        )
        self.z3_green_block = DeconvGreenBlock(feature_size * 2, feature_size)
        self.output_block = nn.Sequential(
            ResidualYellowBlock(feature_size * 2, feature_size, normalization=nn.InstanceNorm3d),
            nn.Conv3d(feature_size, output_channel, kernel_size=1, stride=1),
        )
        self.grid_size = grid_size
        self.hidden_size = hidden_size
        self.slice_num = slice_num

    @staticmethod
    def _project_features(
        x: torch.Tensor, hidden_size: int, grid_size: Sequence[int]
    ) -> torch.Tensor:
        new_shape = (x.size(0), *grid_size, hidden_size)
        x = x.view(new_shape)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(grid_size)))
        return x.permute(new_axes).contiguous()

    def forward(
        self, x_in: torch.Tensor, x: torch.Tensor, hidden_states_out: List[torch.Tensor]
    ) -> torch.Tensor:
        if x_in.ndim == 5:
             # Downsample Time by 4: [B, S, T/4, H, W] — coarser but cheaper for z0 skip
             # [1,1,144,112,112] -> [1,1,72,112,112], ~50% less 3D conv compute
             x_in_sub = x_in[:, :, ::4, :, :]
             # Flatten S, T: [B, S*T/4, H, W]
             x_in_sub = x_in_sub.flatten(1, 2)
             # Add Channel: [B, 1, D, H, W]
             x_in_sub = x_in_sub.unsqueeze(1)
        else:
             x_in_sub = x_in
             
        z0 = self.z0_yellow_block(x_in_sub)
        
        z3 = hidden_states_out[0]
        z3 = self._project_features(z3, self.hidden_size, self.grid_size)
        z3 = z3.reshape(z3.shape[0], self.hidden_size, -1, *self.grid_size[2:])
        z3 = self.z3_block(z3)
        z6 = hidden_states_out[1]
        z6 = self._project_features(z6, self.hidden_size, self.grid_size)
        z6 = z6.reshape(z6.shape[0], self.hidden_size, -1, *self.grid_size[2:])
        z6 = self.z6_block(z6)

        z9 = x.reshape(hidden_states_out[0].shape[0], -1, self.hidden_size)

        z9 = self._project_features(z9, self.hidden_size, self.grid_size)
        z9 = z9.reshape(z9.shape[0], self.hidden_size, -1, *self.grid_size[2:])
        z9 = self.z9_block(z9)
        y = torch.cat([z9, z6], dim=1)
        y = self.z6_yellow_block(y)
        y = self.z6_green_block(y)
        y = torch.cat([y, z3], dim=1)
        y = self.z3_yellow_block(y)
        y = self.z3_green_block(y)
        # Align z0 depth to y (z0 may be smaller due to temporal stride ::4)
        if z0.shape[2] != y.shape[2]:
            z0 = torch.nn.functional.interpolate(
                z0, size=(y.shape[2], *y.shape[3:]), mode='trilinear', align_corners=False
            )
        y = torch.cat([y, z0], dim=1)
        seg_out = self.output_block(y)
        
        # Upsample back to full T resolution (x2)
        if seg_out.shape[2] != x_in.shape[1] * x_in.shape[2]: # If depth mismatch (S*T)
             target_D = x_in.shape[1] * x_in.shape[2]
             seg_out = torch.nn.functional.interpolate(seg_out, size=(target_D, *seg_out.shape[3:]), mode='trilinear', align_corners=False)

        seg_pred = seg_out.view(seg_out.shape[0], -1, self.slice_num, seg_out.shape[2] // self.slice_num, *seg_out.shape[3:])
        return seg_pred
