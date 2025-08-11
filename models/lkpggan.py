from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Stem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.stem(x)


def drop_path(x, drop_prob, training=False, scale_by_keep=True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class RepLKBlock(nn.Module):
    def __init__(self, in_channels, dw_channels, block_lk_size, drop_path=0.):
        super(RepLKBlock, self).__init__()
        self.pw1 = nn.Sequential(
            nn.Conv2d(in_channels, dw_channels, 1, 1, 0, groups=1),
            nn.InstanceNorm2d(dw_channels),
            nn.GELU(),
        )
        self.pw2 = nn.Sequential(
            nn.Conv2d(dw_channels, in_channels, 1, 1, 0, groups=1),
            nn.InstanceNorm2d(in_channels),
            nn.GELU(),
        )
        self.large_kernel = nn.Conv2d(in_channels=dw_channels, out_channels=dw_channels, kernel_size=block_lk_size,
                                      stride=1, padding=block_lk_size // 2, groups=dw_channels, bias=True)
        self.lk_nonlinear = nn.GELU()
        self.prelkb_bn = nn.InstanceNorm2d(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # print('drop path:', self.drop_path)

    def forward(self, x):
        out = self.prelkb_bn(x)
        out = self.pw1(out)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class ConvFFN(nn.Module):
    def __init__(self, in_channels, internal_channels, out_channels, drop_path=0.):
        super(ConvFFN, self).__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.preffn_bn = nn.InstanceNorm2d(in_channels)
        self.pw1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, 1, 1, 0, groups=1),
            nn.InstanceNorm2d(internal_channels),
        )
        self.pw2 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, 1, 1, 0, groups=1),
            nn.InstanceNorm2d(out_channels),
        )
        self.nonlinear = nn.GELU()

    def forward(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class ECANet(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECANet, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()

    def forward(self, x):
        output = self.fgp(x)
        output = output.squeeze(-1).transpose(-1, -2)
        output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
        output = self.act1(output)
        output = torch.multiply(x, output)
        return output


# Simple Channel Attention
class SCA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SCA, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
        )

    def forward(self, x):
        return self.ca(x)


# Attention Gate
class AttGate(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AttGate, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.att = SCA(in_ch, out_ch)

    def forward(self, x_low, x_high):  # x_low-encoder, x_high-decoder
        z1 = torch.cat([x_low, x_high], 1)
        z2 = self.conv(z1) + x_high
        z3 = F.leaky_relu(z2, negative_slope=0.2, inplace=True)
        z4 = self.att(z1)
        z5 = z4 * z3
        out = z5 + z3
        return out


class SCConv(nn.Module):
    def __init__(self, planes, pooling_r):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(planes, planes, 3, 1, 1),
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(
            torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))  # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out)  # k3 * sigmoid(identity + k2)
        out = self.k4(out)  # k4

        return out


class SCBottleneck(nn.Module):
    # expansion = 4
    pooling_r = 4  # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, in_planes, planes):
        super(SCBottleneck, self).__init__()
        planes = int(planes / 2)

        self.conv1_a = nn.Conv2d(in_planes, planes, 1, 1)
        self.k1 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

        self.conv1_b = nn.Conv2d(in_planes, planes, 1, 1)

        self.scconv = SCConv(planes, self.pooling_r)

        self.conv3 = nn.Conv2d(planes * 2, planes * 2, 1, 1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)
        out_a = self.relu(out_a)

        out_a = self.k1(out_a)

        out_b = self.conv1_b(x)
        out_b = self.relu(out_b)

        out_b = self.scconv(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))

        out += residual
        out = self.relu(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      SCBottleneck(in_features, in_features)
                      ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class EndConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EndConv, self).__init__()
        self.end = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        return self.end(x)


class Encoder(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Encoder, self).__init__()
        self.encoder_layers1 = nn.Sequential(
            RepLKBlock(in_planes, out_planes, 13),
            ConvFFN(out_planes, out_planes * 2, out_planes),
            SCBottleneck(out_planes, out_planes)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_planes * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            SCBottleneck(out_planes * 2, out_planes * 2)
        )

        self.encoder_layers2 = nn.Sequential(
            RepLKBlock(out_planes * 2, out_planes * 2, 13),
            ConvFFN(out_planes * 2, out_planes * 4, out_planes * 2),
            SCBottleneck(out_planes * 2, out_planes * 2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(out_planes * 2, out_planes * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_planes * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            SCBottleneck(out_planes * 4, out_planes * 4)
        )

        self.encoder_layers3 = nn.Sequential(
            RepLKBlock(out_planes * 4, out_planes * 4, 13),
            ConvFFN(out_planes * 4, out_planes * 8, out_planes * 4),
            SCBottleneck(out_planes * 4, out_planes * 4)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(out_planes * 4, out_planes * 8, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_planes * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            SCBottleneck(out_planes * 8, out_planes * 8)
        )

        self.encoder_layers4 = nn.Sequential(
            RepLKBlock(out_planes * 8, out_planes * 8, 13),
            ConvFFN(out_planes * 8, out_planes * 16, out_planes * 8),
            SCBottleneck(out_planes * 8, out_planes * 8)
        )

    def forward(self, x):
        x1 = self.encoder_layers1(x)
        x2 = self.down1(x1)
        x3 = self.encoder_layers2(x2)
        x4 = self.down2(x3)
        x5 = self.encoder_layers3(x4)
        x6 = self.down3(x5)
        x7 = self.encoder_layers4(x6)
        return x7, x5, x3, x1


class Decoder(nn.Module):
    def __init__(self, out_planes):
        super(Decoder, self).__init__()
        self.out_planes = out_planes
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(out_planes * 8, out_planes * 4, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_planes * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ECANet(out_planes * 4),
            # RepLKBlock(out_planes * 4, out_planes * 4, 13),
            # ConvFFN(out_planes * 4, out_planes * 8, out_planes * 4),
            SCBottleneck(out_planes * 4, out_planes * 4)
        )
        self.att1 = AttGate(out_planes * 4 + out_planes * 4, out_planes * 4)

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(out_planes * 4, out_planes * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_planes * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ECANet(out_planes * 2),
            # RepLKBlock(out_planes * 2, out_planes * 2, 13),
            # ConvFFN(out_planes * 2, out_planes * 4, out_planes * 2),
            SCBottleneck(out_planes * 2, out_planes * 2)
        )
        self.att2 = AttGate(out_planes * 2 + out_planes * 2, out_planes * 2)

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(out_planes * 2, out_planes, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_planes),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ECANet(out_planes),
            # RepLKBlock(out_planes, out_planes, 13),
            # ConvFFN(out_planes, out_planes * 2, out_planes),
            SCBottleneck(out_planes, out_planes)
        )
        self.att3 = AttGate(out_planes + out_planes, out_planes)

    def forward(self, res_feature, encoder_feature):
        x1 = self.att1(encoder_feature[1], self.up1(res_feature))
        x2 = self.att2(encoder_feature[2], self.up2(x1))
        x3 = self.att3(encoder_feature[3], self.up3(x2))
        return x3


class LKPGGAN(nn.Module):
    def __init__(self, in_ch, out_ch, ngf, n_blocks=9):
        super(LKPGGAN, self).__init__()
        self.stem = Stem(in_ch, ngf)
        self.encoder = Encoder(ngf, ngf)
        res = []
        for _ in range(n_blocks):
            res += [ResidualBlock(ngf * 8)]
        self.res = nn.Sequential(*res)
        self.decoder = Decoder(ngf)
        self.end = EndConv(ngf, out_ch)

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.encoder(x1)
        x3 = self.res(x2[0])
        x4 = self.decoder(x3, x2)
        out = self.end(x4)
        return out


if __name__ == "__main__":
    input = torch.Tensor(1, 3, 256, 256).cuda()
    model = LKPGGAN(3, 3, 64, 9).cuda()
    model.eval()
    print(model)
    output = model(input)
    summary(model, (3, 256, 256))
    print(output.shape)
