import torch.nn as nn
import torch
from blocks import residualBlock, downSampleBlock, upSampleBlock, attentionBlock


class UNet(nn.Module):

    def __init__(self, in_ch, out_ch, time_dim=256, device='cuda'):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        # encoder layers
        self.res_in = residualBlock(in_ch, 64)
        self.down1 = downSampleBlock(64, 128)
        self.sha1 = attentionBlock(128, 32)
        self.down2 = downSampleBlock(128, 256)
        self.sha2 = attentionBlock(256, 16)
        self.down3 = downSampleBlock(256, 256)
        self.sha3 = attentionBlock(256, 8)

        # bottleneck layers
        self.bot1 = residualBlock(256, 512)
        self.bot2 = residualBlock(512, 512)
        self.bot3 = residualBlock(512, 256)

        # decoder layers
        self.up1 = upSampleBlock(512, 128)
        self.sha4 = attentionBlock(128, 16)
        self.up2 = upSampleBlock(256, 64)
        self.sha5 = attentionBlock(64, 32)
        self.up3 = upSampleBlock(128, 64)
        self.sha6 = attentionBlock(64, 64)
        self.out = nn.Conv2d(64, out_ch, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)  # add an extra dimension to t
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.res_in(x)
        x2 = self.down1(x1, t)
        x2 = self.sha1(x2)
        x3 = self.down2(x2, t)
        # x3 = self.sha2(x3)
        x4 = self.down3(x3, t)
        # x4 = self.sha3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # upSample blocks with added skip connections from the downSample blocks
        x = self.up1(x4, x3, t)
        x = self.sha4(x)
        x = self.up2(x, x2, t)
        # x = self.sha5(x)
        x = self.up3(x, x1, t)
        # x = self.sha6(x)
        output = self.out(x)
        return output
