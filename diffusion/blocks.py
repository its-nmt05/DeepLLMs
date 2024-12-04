import torch
import torch.nn as nn
import torch.nn.functional as F

# residual(ResNet) block with optional skip connection
class residualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, mid_ch=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class attentionBlock(nn.Module):

    def __init__(self, channels, size):
        super().__init__()
        self.channels = channels
        self.size = size
        self.ln = nn.LayerNorm([channels])
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ffwd = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        # flatten spatial dims for MHA
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_scores, _ = self.mha(x_ln, x_ln, x_ln)
        attention_scores = attention_scores + x
        attention_scores = self.ffwd(attention_scores) + attention_scores
        # unflatten spatial dims
        return attention_scores.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class downSampleBlock(nn.Module):

    def __init__(self, in_ch, out_ch, emb_dim=256):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.res1 = residualBlock(in_ch, in_ch, residual=True)
        self.res2 = residualBlock(in_ch, out_ch)

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_ch)
        )

    def forward(self, x, t):
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(
            # expand emb spatially to match dims of x
            1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class upSampleBlock(nn.Module):

    def __init__(self, in_ch, out_ch, emb_dim=256):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.res1 = residualBlock(in_ch, in_ch, residual=True)
        self.res2 = residualBlock(in_ch, out_ch, in_ch // 2)

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_ch)
        )

    def forward(self, x, skip_x, t):
        x = self.upsample(x)
        x = torch.cat([x, skip_x], dim=1)
        x = self.res1(x)
        x = self.res2(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(
            # expand emb spatially to match dims of x
            1, 1, x.shape[-2], x.shape[-1])
        return x + emb
