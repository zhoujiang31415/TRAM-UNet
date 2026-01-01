import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_utils import up_block, down_block, BasicBlock, BottleneckBlock
from .conv_trans_utils import down_block_trans, up_block_trans


class RegionAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(RegionAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        attention = self.sigmoid(avg_out)
        return x * attention

class TRAM_UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        base_chan=64,
        n_classes=1,
        reduce_size=8,
        block_list='234',
        num_blocks=None,
        projection='interp',
        num_heads=None,
        attn_drop=0.,
        proj_drop=0.,
        bottleneck=False,
        maxpool=True,
        rel_pos=True,
        aux_loss=False,
        use_TR=True,           # ablation switch：Transformer Block
        use_RA=True            # ablation switch：Region Attention
    ):
        super().__init__()
        self.id = "TRAM_UNet"
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_TR = use_TR
        self.use_RA = use_RA

        if num_blocks is None or len(num_blocks) < 4:
            num_blocks = [1, 1, 1, 1]
        if num_heads is None or len(num_heads) < 4:
            num_heads = [2, 4, 8, 16]

        # 1. Define RAM
        if self.use_RA:
            self.region_attn1 = RegionAttentionModule(2 * base_chan)
            self.region_attn2 = RegionAttentionModule(4 * base_chan)
            self.region_attn3 = RegionAttentionModule(8 * base_chan)
            self.region_attn4 = RegionAttentionModule(16 * base_chan)
            self.up_attn1 = RegionAttentionModule(8 * base_chan)
            self.up_attn2 = RegionAttentionModule(4 * base_chan)
            self.up_attn3 = RegionAttentionModule(2 * base_chan)
            self.up_attn4 = RegionAttentionModule(base_chan)

        # 2. Residual Backbone
        self.inc = nn.Sequential(
            BasicBlock(n_channels, base_chan),
            BasicBlock(base_chan, base_chan)
        )

        # 3. Downsampling Path
        if self.use_TR:
            common_params = {"bottleneck": bottleneck, "maxpool": maxpool, "reduce_size": reduce_size, "projection": projection, "rel_pos": rel_pos}
            self.down1 = down_block_trans(base_chan, 2*base_chan, num_blocks[0], heads=num_heads[0], dim_head=(2*base_chan)//num_heads[0], **common_params)
            self.down2 = down_block_trans(2*base_chan, 4*base_chan, num_blocks[1], heads=num_heads[1], dim_head=(4*base_chan)//num_heads[1], **common_params)
            self.down3 = down_block_trans(4*base_chan, 8*base_chan, num_blocks[2], heads=num_heads[2], dim_head=(8*base_chan)//num_heads[2], **common_params)
            self.down4 = down_block_trans(8*base_chan, 16*base_chan, num_blocks[3], heads=num_heads[3], dim_head=(16*base_chan)//num_heads[3], **common_params)
        else:
            self.down1 = down_block(base_chan, 2*base_chan, scale=2, num_block=1, bottleneck=bottleneck, pool=maxpool)
            self.down2 = down_block(2*base_chan, 4*base_chan, scale=2, num_block=1, bottleneck=bottleneck, pool=maxpool)
            self.down3 = down_block(4*base_chan, 8*base_chan, scale=2, num_block=1, bottleneck=bottleneck, pool=maxpool)
            self.down4 = down_block(8*base_chan, 16*base_chan, scale=2, num_block=1, bottleneck=bottleneck, pool=maxpool)

        # 4. Upsampling Path
        if self.use_TR:
            up_params = {"bottleneck": bottleneck, "reduce_size": reduce_size, "projection": projection, "rel_pos": rel_pos}
            self.up1 = up_block_trans(16*base_chan, 8*base_chan, 0, heads=num_heads[3], dim_head=(8*base_chan)//num_heads[3], **up_params)
            self.up2 = up_block_trans(8*base_chan, 4*base_chan, 0, heads=num_heads[2], dim_head=(4*base_chan)//num_heads[2], **up_params)
            self.up3 = up_block_trans(4*base_chan, 2*base_chan, 0, heads=num_heads[1], dim_head=(2*base_chan)//num_heads[1], **up_params)
            self.up4 = up_block_trans(2*base_chan, base_chan, 0, heads=num_heads[0], dim_head=(base_chan)//num_heads[0], **up_params)
        else:
            self.up1 = up_block(16*base_chan, 8*base_chan, num_block=1, scale=(2,2), bottleneck=bottleneck)
            self.up2 = up_block(8*base_chan, 4*base_chan, num_block=1, scale=(2,2), bottleneck=bottleneck)
            self.up3 = up_block(4*base_chan, 2*base_chan, num_block=1, scale=(2,2), bottleneck=bottleneck)
            self.up4 = up_block(2*base_chan, base_chan, num_block=1, scale=(2,2), bottleneck=bottleneck)

        self.outc = nn.Conv2d(base_chan, n_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x1 = self.inc(x)
        
        # Encoding Phase
        d1 = self.down1(x1)
        x2 = self.region_attn1(d1) if self.use_RA else d1
        
        d2 = self.down2(x2)
        x3 = self.region_attn2(d2) if self.use_RA else d2
        
        d3 = self.down3(x3)
        x4 = self.region_attn3(d3) if self.use_RA else d3
        
        d4 = self.down4(x4)
        x5 = self.region_attn4(d4) if self.use_RA else d4

        # Decoding Phase
        u1 = self.up1(x5, x4)
        out = self.up_attn1(u1) if self.use_RA else u1
        
        u2 = self.up2(out, x3)
        out = self.up_attn2(u2) if self.use_RA else u2
        
        u3 = self.up3(out, x2)
        out = self.up_attn3(u3) if self.use_RA else u3
        
        u4 = self.up4(out, x1)
        out = self.up_attn4(u4) if self.use_RA else u4

        return self.outc(out)