from typing import List
import torch
import torch.nn as nn
from timm.layers.squeeze_excite import EffectiveSEModule
from timm.models.layers import DropPath
from timm.models.layers import LayerNorm2d
 
 
__all__ = ['RDNet']
 
 
class RDNetClassifierHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_features = in_features
 
        self.norm = nn.LayerNorm(in_features)
        self.drop = nn.Dropout(drop_rate)
        self.fc = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
 
    def reset(self, num_classes):
        self.fc = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
 
    def forward(self, x, pre_logits: bool = False):
        x = x.mean([-2, -1])
        x = self.norm(x)
        x = self.drop(x)
        if pre_logits:
            return x
        x = self.fc(x)
        return x
 
 
class PatchifyStem(nn.Module):
    def __init__(self, num_input_channels, num_init_features, patch_size=4):
        super().__init__()
 
        self.stem = nn.Sequential(
            nn.Conv2d(
                num_input_channels,
                num_init_features,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            LayerNorm2d(num_init_features),
        )
 
    def forward(self, x):
        return self.stem(x)
 
 
class Block(nn.Module):
    def __init__(self, in_chs, inter_chs, out_chs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_chs, in_chs, groups=in_chs, kernel_size=7, stride=1, padding=3
            ),
            LayerNorm2d(in_chs, eps=1e-6),
            nn.Conv2d(in_chs, inter_chs, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(inter_chs, out_chs, kernel_size=1, stride=1, padding=0),
        )
 
    def forward(self, x):
        return self.layers(x)
 
 
class BlockESE(nn.Module):
    def __init__(self, in_chs, inter_chs, out_chs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_chs, in_chs, groups=in_chs, kernel_size=7, stride=1, padding=3
            ),
            LayerNorm2d(in_chs, eps=1e-6),
            nn.Conv2d(in_chs, inter_chs, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(inter_chs, out_chs, kernel_size=1, stride=1, padding=0),
            EffectiveSEModule(out_chs),
        )
 
    def forward(self, x):
        return self.layers(x)
 
 
class RDNet(nn.Module):
    def __init__(
        self,
        num_input_features,
        growth_rate, # growth_rates=(64, 104, 128, 128, 128, 128, 224)
        bottleneck_width_ratio=64,
        drop_path_rate=4,
        drop_rate=0.0,
        rand_gather_step_prob=0.0,
        block_idx=0,
        block_type="Block",
        ls_init_value=1e-6,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.rand_gather_step_prob = rand_gather_step_prob
        self.block_idx = block_idx
        self.growth_rate = growth_rate
 
        self.gamma = (
            nn.Parameter(ls_init_value * torch.ones(growth_rate))
            if ls_init_value > 0
            else None
        )
        growth_rate = int(growth_rate)
        inter_chs = int(num_input_features * bottleneck_width_ratio / 8) * 8
 
        if self.drop_path_rate > 0:
            self.drop_path = DropPath(drop_path_rate)
 
        self.layers = eval(block_type)(
            in_chs=num_input_features,
            inter_chs=inter_chs,
            out_chs=growth_rate,
        )
 
    def forward(self, x):
        if isinstance(x, List):
            x = torch.cat(x, 1)
        x = self.layers(x)
 
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
 
        return x
 
 
 