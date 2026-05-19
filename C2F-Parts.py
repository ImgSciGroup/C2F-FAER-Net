# Deformconv_v2
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mmcv.ops import ModulatedDeformConv2d as DeformConv2d
    use_modulated = True
except ImportError:
    from torchvision.ops import DeformConv2d  # DCNv1 fallback
    use_modulated = False

class DeformableAlign(nn.Module):
    def __init__(self, channels):
        super(DeformableAlign, self).__init__()
        if use_modulated:
            self.offset_mask_conv = nn.Sequential(
                nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 27, kernel_size=3, padding=1)  # 18 offset + 9 mask
            )
            # 初始化最后一层为 0：表示初始偏移为 0，mask 为 0.5（sigmoid(0)）
            nn.init.constant_(self.offset_mask_conv[-1].weight, 0)
            nn.init.constant_(self.offset_mask_conv[-1].bias, 0)

            self.deform_conv = DeformConv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1
            )
        else:
            self.offset_conv = nn.Sequential(
                nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 18, kernel_size=3, padding=1)
            )
            nn.init.constant_(self.offset_conv[-1].weight, 0)
            nn.init.constant_(self.offset_conv[-1].bias, 0)

            self.deform_conv = DeformConv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1
            )

    def forward(self, x_src, x_ref):
        x_cat = torch.cat([x_src, x_ref], dim=1)
        if use_modulated:
            offset_mask = self.offset_mask_conv(x_cat)
            offset, mask = torch.split(offset_mask, [18, 9], dim=1)
            mask = torch.sigmoid(mask)
            return self.deform_conv(x_src, offset, mask)
        else:
            offset = self.offset_conv(x_cat)
            return self.deform_conv(x_src, offset)



class PyramidAlignBlock(nn.Module):
    def __init__(self, channels_list=[512, 256, 128, 64]):
        super(PyramidAlignBlock, self).__init__()
        self.align5 = DeformableAlign(channels_list[0])  # 8x8
        self.align4 = DeformableAlign(channels_list[1])  # 16x16
        self.align3 = DeformableAlign(channels_list[2])  # 32x32
        self.align2 = DeformableAlign(channels_list[3])  # 64x64

        # 通道对齐（防止相加报错）
        self.reduce_conv_5to4 = nn.Conv2d(512, 256, kernel_size=1)
        self.reduce_conv_4to3 = nn.Conv2d(256, 128, kernel_size=1)
        self.reduce_conv_3to2 = nn.Conv2d(128, 64, kernel_size=1)

    def forward(self, x2_feats, x1_feats):
        x2_5, x2_4, x2_3, x2_2 = x2_feats
        x1_5, x1_4, x1_3, x1_2 = x1_feats

        # === Level 5 对齐
        x2_5_aligned = self.align5(x2_5, x1_5)

        # === Level 4 对齐
        x2_5_up = F.interpolate(x2_5_aligned, size=x2_4.shape[2:], mode='bilinear', align_corners=True)
        x2_5_up = self.reduce_conv_5to4(x2_5_up)
        x2_4_input = x2_4 + x2_5_up
        x2_4_aligned = self.align4(x2_4_input, x1_4)

        # === Level 3 对齐
        x2_4_up = F.interpolate(x2_4_aligned, size=x2_3.shape[2:], mode='bilinear', align_corners=True)
        x2_4_up = self.reduce_conv_4to3(x2_4_up)
        x2_3_input = x2_3 + x2_4_up
        x2_3_aligned = self.align3(x2_3_input, x1_3)

        # === Level 2 对齐
        x2_3_up = F.interpolate(x2_3_aligned, size=x2_2.shape[2:], mode='bilinear', align_corners=True)
        x2_3_up = self.reduce_conv_3to2(x2_3_up)
        x2_2_input = x2_2 + x2_3_up
        x2_2_aligned = self.align2(x2_2_input, x1_2)

        return [x2_5_aligned, x2_4_aligned, x2_3_aligned, x2_2_aligned], [x1_5, x1_4, x1_3, x1_2]


import torch
import torch.nn as nn
import torch.nn.functional as F

class GloballocalChangeAwareBlock(nn.Module):
    def __init__(self, channels):
        super(GloballocalChangeAwareBlock, self).__init__()
        self.channels = channels

        # qkv 线性变换
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)

        # 输出变换
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.diff_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat1, feat2):
        # === Step 1 ===
        # proj Q, K, V
        q1 = self.q_proj(feat1)
        k2 = self.k_proj(feat2)
        v2 = self.v_proj(feat2)

        q2 = self.q_proj(feat2)
        k1 = self.k_proj(feat1)
        v1 = self.v_proj(feat1)

        # flatten for attention
        B, C, H, W = q1.shape
        q1_ = q1.view(B, C, -1).transpose(1, 2)    # [B, HW, C]
        k2_ = k2.view(B, C, -1)                    # [B, C, HW]
        v2_ = v2.view(B, C, -1).transpose(1, 2)    # [B, HW, C]

        q2_ = q2.view(B, C, -1).transpose(1, 2)
        k1_ = k1.view(B, C, -1)
        v1_ = v1.view(B, C, -1).transpose(1, 2)

        attn1 = torch.bmm(q1_, k2_) / (C ** 0.5)    # [B, HW, HW]
        attn1 = F.softmax(attn1, dim=-1)
        out1 = torch.bmm(attn1, v2_)                # [B, HW, C]
        out1 = out1.transpose(1, 2).view(B, C, H, W)

        attn2 = torch.bmm(q2_, k1_) / (C ** 0.5)
        attn2 = F.softmax(attn2, dim=-1)
        out2 = torch.bmm(attn2, v1_)
        out2 = out2.transpose(1, 2).view(B, C, H, W)

        out1 = self.out_proj(out1) + feat1
        out2 = self.out_proj(out2) + feat2

        # === Step 2 ===
        diff = out1 - out2                          # [B, C, H, W]
        gap = F.adaptive_avg_pool2d(diff, (1, 1))   # [B, C, 1, 1]
        gmp = F.adaptive_max_pool2d(diff, (1, 1))
        att = torch.sigmoid(gap + gmp)              # [B, C, 1, 1]

        # 加权增强
        out1 = out1 * att + out1
        out2 = out2 * att + out2

        # 融合压缩回 C 通道
        enhanced = torch.cat([out1, out2], dim=1)   # [B, 2C, H, W]
        fused = self.diff_conv(enhanced)            # [B, C, H, W]
        final = self.fuse_conv(fused)               # [B, C, H, W]

        return final

