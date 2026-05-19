import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import resnet18
from C2F-Parts import PyramidAlignBlock, GloballocalChangeAwareBlock


class TemporalFeatureInteractionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(TemporalFeatureInteractionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv_sub = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_diff_enh1 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_diff_enh2 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(self.in_d * 2, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_dr = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=1, bias=True),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # difference enhance
        x_sub = self.conv_sub(torch.abs(x1 - x2))
        x1 = self.conv_diff_enh1(x1.mul(x_sub) + x1)
        x2 = self.conv_diff_enh2(x2.mul(x_sub) + x2)
        # fusion
        x_f = torch.cat([x1, x2], dim=1)
        x_f = self.conv_cat(x_f)
        x = x_sub + x_f
        x = self.conv_dr(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ChangeInformationExtractionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(ChangeInformationExtractionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.ca = ChannelAttention(self.in_d * 4, ratio=16)
        self.conv_dr = nn.Sequential(
            nn.Conv2d(self.in_d * 4, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.pools_sizes = [2, 4, 8]
        self.conv_pool1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[0], stride=self.pools_sizes[0]),
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_pool2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[1], stride=self.pools_sizes[1]),
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_pool3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[2], stride=self.pools_sizes[2]),
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, d5, d4, d3, d2):
        # upsampling
        d5 = F.interpolate(d5, d2.size()[2:], mode='bilinear', align_corners=True)
        d4 = F.interpolate(d4, d2.size()[2:], mode='bilinear', align_corners=True)
        d3 = F.interpolate(d3, d2.size()[2:], mode='bilinear', align_corners=True)
        # fusion
        x = torch.cat([d5, d4, d3, d2], dim=1)
        x_ca = self.ca(x)
        x = x * x_ca
        x = self.conv_dr(x)

        # feature = x[0:1, 0:64, 0:64, 0:64]
        # vis.visulize_features(feature)

        # pooling
        d2 = x
        d3 = self.conv_pool1(x)
        d4 = self.conv_pool2(x)
        d5 = self.conv_pool3(x)

        return d5, d4, d3, d2


class GuidedRefinementModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(GuidedRefinementModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv_d5 = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
        self.conv_d4 = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
        self.conv_d3 = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
        self.conv_d2 = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p):
        # feature refinement
        d5 = self.conv_d5(d5_p + d5)
        d4 = self.conv_d4(d4_p + d4)
        d3 = self.conv_d3(d3_p + d3)
        d2 = self.conv_d2(d2_p + d2)

        return d5, d4, d3, d2


import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDownsampler(nn.Module):
    def __init__(self):
        super(MultiScaleDownsampler, self).__init__()

        # d5: 512 -> 64 (保持8x8)
        self.d5_reduce = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # d4: 256 -> 64 (保持16x16)
        self.d4_reduce = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # d3: 128 -> 64 (保持32x32)
        self.d3_reduce = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # d2: 64 -> 64 (保持64x64，实际无需处理)
        self.d2_identity = nn.Identity()  # 通道数已是64，直接恒等映射

    def forward(self, d5, d4, d3, d2):
        """
        输入:
            d5: [1, 512, 8, 8]
            d4: [1, 256, 16, 16]
            d3: [1, 128, 32, 32]
            d2: [1, 64, 64, 64]
        输出:
            同输入空间维度，通道数统一为64
        """
        return (
            self.d5_reduce(d5),  # -> [1,64,8,8]
            self.d4_reduce(d4),  # -> [1,64,16,16]
            self.d3_reduce(d3),  # -> [1,64,32,32]
            self.d2_identity(d2)  # -> [1,64,64,64] (保持不变)
        )

import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, out_d):
        super(Decoder, self).__init__()

        self.up4 = self._up_block(512, 256)  # d5 + d4
        self.up3 = self._up_block(256, 128)  # 上一步 + d3
        self.up2 = self._up_block(128, 64)   # 上一步 + d2

        # 额外两级上采样
        self.up1 = self._up_block(64, 32, use_skip=False)    # 64x64 → 128x128
        self.up0 = self._up_block(32, 16, use_skip=False)    # 128x128 → 256x256

        self.final_conv = nn.Conv2d(16, out_d, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _up_block(self, in_channels, out_channels, use_skip=True):
        if use_skip:
            conv_in = out_channels * 2
        else:
            conv_in = out_channels
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self._conv_block(conv_in, out_channels)
        )

    def forward(self, d5, d4, d3, d2):
        # 8x8 → 16x16
        x = self.up4[0](d5)
        x = torch.cat([x, d4], dim=1)
        x = self.up4[1](x)

        # 16x16 → 32x32
        x = self.up3[0](x)
        x = torch.cat([x, d3], dim=1)
        x = self.up3[1](x)

        # 32x32 → 64x64
        x = self.up2[0](x)
        x = torch.cat([x, d2], dim=1)
        x = self.up2[1](x)

        # 64x64 → 128x128（无 skip）
        x = self.up1[0](x)
        x = self.up1[1](x)

        # 128x128 → 256x256（无 skip）
        x = self.up0[0](x)
        x = self.up0[1](x)

        # 输出结果
        mask = self.final_conv(x)
        return mask


class LaplacianConv(nn.Module):
    def __init__(self, in_channels):
        super(LaplacianConv, self).__init__()
        kernel = torch.tensor([[[[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]]]], dtype=torch.float32)
        self.register_buffer('weight', kernel.repeat(in_channels, 1, 1, 1))
        self.in_channels = in_channels

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=1, groups=self.in_channels)

class EdgeDecoder(nn.Module):
    def __init__(self, out_d):
        super(EdgeDecoder, self).__init__()

        # Laplacian Edge Convs
        self.lap5 = LaplacianConv(512)
        self.lap4 = LaplacianConv(256)
        self.lap3 = LaplacianConv(128)
        self.lap2 = LaplacianConv(64)

        # 上采样 + 卷积块
        self.up4 = self._up_block(512, 256)  # d5 → d4
        self.up3 = self._up_block(256, 128)
        self.up2 = self._up_block(128, 64)
        self.up1 = self._up_block(64, 32, use_skip=False)   # no skip
        self.up0 = self._up_block(32, 16, use_skip=False)   # no skip

        self.final_conv = nn.Conv2d(16, out_d, kernel_size=1)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def _up_block(self, in_c, out_c, use_skip=True):
        if use_skip:
            cat_c = out_c * 2
        else:
            cat_c = out_c
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            self._conv_block(cat_c, out_c)
        )

    def forward(self, d5, d4, d3, d2):
        # Step 1: Edge maps
        d5 = self.lap5(d5)
        d4 = self.lap4(d4)
        d3 = self.lap3(d3)
        d2 = self.lap2(d2)

        # 8×8 → 16×16
        x = self.up4[0](d5)
        x = torch.cat([x, d4], dim=1)
        x = self.up4[1](x)

        # 16×16 → 32×32
        x = self.up3[0](x)
        x = torch.cat([x, d3], dim=1)
        x = self.up3[1](x)

        # 32×32 → 64×64
        x = self.up2[0](x)
        x = torch.cat([x, d2], dim=1)
        x = self.up2[1](x)

        # 64×64 → 128×128
        x = self.up1[0](x)
        x = self.up1[1](x)

        # 128×128 → 256×256
        x = self.up0[0](x)
        x = self.up0[1](x)

        # Final edge prediction
        edge_mask = self.final_conv(x)
        return edge_mask





class BaseNet(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(BaseNet, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.mid_d = 64

        self.Align = PyramidAlignBlock(channels_list=[512, 256, 128, 64])
        self.GLcam5 = GloballocalChangeAwareBlock(channels=512)
        self.GLcam4 = GloballocalChangeAwareBlock(channels=256)
        self.GLcam3 = GloballocalChangeAwareBlock(channels=128)
        self.GLcam2 = GloballocalChangeAwareBlock(channels=64)


        self.down5 = SimplifiedTFIM(512, self.mid_d)
        self.down4 = SimplifiedTFIM(256, self.mid_d)
        self.down3 = SimplifiedTFIM(128, self.mid_d)
        self.down2 = SimplifiedTFIM(64, self.mid_d)

        self.main_decoder = Decoder(output_nc)
        self.edge_decoder = EdgeDecoder(output_nc)

    def forward(self, x1, x2):
        # forward backbone resnet
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone.base_forward(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone.base_forward(x2)
        print(x1_1.shape)
        print(x1_2.shape)
        print(x1_3.shape)
        print(x1_4.shape)
        print(x2_5.shape)


        # PAM
        x2_feats = [x2_5, x2_4, x2_3, x2_2]
        x1_feats = [x1_5, x1_4, x1_3, x1_2]
        x2_feats_aligned, x1_feats = self.Align(x2_feats, x1_feats)
        x2_5, x2_4, x2_3, x2_2 = x2_feats_aligned
        x1_5, x1_4, x1_3, x1_2 = x1_feats

        # Enhanced
        d5 = self.Scam5(x2_5, x1_5)
        d4 = self.Scam4(x2_4, x1_4)
        d3 = self.Scam3(x2_3, x1_3)
        d2 = self.Scam2(x2_2, x1_2)

        # decoder
        main_mask = self.main_decoder(d5, d4, d3, d2)
        edge_mask = self.edge_decoder(d5, d4, d3, d2)
        # print("mask_1 = ", mask.shape)
        main_mask = F.interpolate(main_mask, x1.size()[2:], mode='bilinear', align_corners=True)
        edge_mask = F.interpolate(edge_mask, x1.size()[2:], mode='bilinear', align_corners=True)
        # print("mask_2 = ", mask.shape)
        main_mask = torch.sigmoid(main_mask)
        edge_mask = torch.sigmoid(edge_mask)

        return main_mask, edge_mask

if __name__ == '__main__':
    # Create a test case with simulated images
    batch_size = 1
    channels = 3
    height = 256
    width = 256

    # Create random input tensors (simulating two temporal images)
    x1 = torch.randn(batch_size, channels, height, width)
    x2 = torch.randn(batch_size, channels, height, width)
    # print(x1.shape)
    # print(x2.shape)

    # Initialize the model
    input_nc = channels  # input channels
    output_nc = 1  # binary change detection output
    model = BaseNet(input_nc, output_nc)

    # Print model summary
    # print(model)

    # Test forward pass
    # print("\nInput shape - x1:", x1.shape, "x2:", x2.shape)
    with torch.no_grad():
        output1, output2 = model(x1, x2)

    # Check output
    print("Output shape:", output1.shape, output2.shape)
    # print("Output min/max:", output.min().item(), output.max().item())

    # Verify the output makes sense
    # assert output.shape == (batch_size, output_nc, height, width), "Output shape mismatch"
    # assert 0 <= output.min().item() <= output.max().item() <= 1, "Output should be between 0 and 1 (sigmoid activated)"

    # print("\nTest passed successfully!")