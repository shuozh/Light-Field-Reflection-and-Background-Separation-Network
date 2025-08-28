import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.commom import DownSample4D, UpSample, ca_layer, Spacial2D, EncoderDecoder, Spacial3D

def activate(act='prelu'):
    if act == 'prelu':
        return nn.PReLU()


class PrepareInput(nn.Module):
    def __init__(self, channels, if_bn=False, bias=True, act='prelu'):
        super(PrepareInput, self).__init__()
        m = []
        m.append(nn.Conv2d(3, channels, 3, 1, 1, bias=bias))
        if if_bn:
            m.append(nn.BatchNorm2d(channels))
        if act is not None:
            m.append(activate(act))
        self.conv = nn.Sequential(*m)
        self.channels = channels

    def forward(self, x):
        batch, height_view, width_view, height, width, channel = x.shape
        x = x.permute(0, 1, 2, 5, 3, 4)
        x = x.reshape(batch * height_view * width_view, channel, height, width)
        x = self.conv(x)
        x = x.reshape(batch, height_view, width_view, self.channels, height, width)
        x = x.permute(0, 3, 1, 2, 4, 5)
        return x


class SA(nn.Module):
    def __init__(self, channels, if_bn=False, bias=True, act='prelu'):
        super(SA, self).__init__()
        m = []
        m.append(nn.Conv2d(channels, channels, 3, 1, 1, bias=bias))
        if if_bn:
            m.append(nn.BatchNorm2d(channels))
        if act is not None:
            m.append(activate(act))
        self.spaconv = nn.Sequential(*m)

        m = []
        m.append(nn.Conv2d(channels, channels, 3, 1, 1, bias=bias))
        if if_bn:
            m.append(nn.BatchNorm2d(channels))
        if act is not None:
            m.append(activate(act))
        self.angconv = nn.Sequential(*m)

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = x.shape

        x = x.permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(batch * height_view * width_view, channel, height, width)
        x = self.spaconv(x)
        x = x.reshape(batch, height_view, width_view, channel, height, width)

        x = x.permute(0, 4, 5, 3, 1, 2)
        x = x.reshape(batch * height * width, channel, height_view, width_view)
        x = self.angconv(x)
        x = x.reshape(batch, height, width, channel, height_view, width_view)

        x = x.permute(0, 3, 4, 5, 1, 2)

        return x


class SAD(nn.Module):
    def __init__(self, channels, if_bn=False, bias=True, act='prelu'):
        super(SAD, self).__init__()
        m = []
        m.append(nn.Conv2d(channels, channels, 3, 1, 1, bias=bias))
        if if_bn:
            m.append(nn.BatchNorm2d(channels))
        if act is not None:
            m.append(activate(act))
        self.spaconv = nn.Sequential(*m)

        m = []
        m.append(nn.Conv2d(channels, channels, 3, 1, 0, bias=bias))
        if if_bn:
            m.append(nn.BatchNorm2d(channels))
        if act is not None:
            m.append(activate(act))
        self.angconv = nn.Sequential(*m)

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = x.shape

        x = x.permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(batch * height_view * width_view, channel, height, width)
        x = self.spaconv(x)
        x = x.reshape(batch, height_view, width_view, channel, height, width)

        x = x.permute(0, 4, 5, 3, 1, 2)
        x = x.reshape(batch * height * width, channel, height_view, width_view)
        x = self.angconv(x)
        x = x.reshape(batch, height, width, channel, height_view - 2, width_view - 2
                      )

        x = x.permute(0, 3, 4, 5, 1, 2)

        return x


class MultiSAD(nn.Module):
    def __init__(self, num=2, channels=64, if_bn=False, bias=True, act='prelu'):
        super(MultiSAD, self).__init__()
        m = []
        for i in range(num):
            m.append(SAD(channels, if_bn, bias, act))
        self.SAD = nn.Sequential(*m)

    def forward(self, x):
        x = self.SAD(x)
        return x


class MultiSA(nn.Module):
    def __init__(self, num=2, channels=64, if_bn=False, bias=True, act='prelu'):
        super(MultiSA, self).__init__()
        m = []
        for i in range(num):
            m.append(SA(channels, if_bn, bias, act))
        self.SA = nn.Sequential(*m)

    def forward(self, x):
        x = self.SA(x)
        return x


class MultiSAS(nn.Module):
    def __init__(self, channels, sa_num=2, sad_num=2, if_bn=False, bias=True, act='prelu'):
        super(MultiSAS, self).__init__()

        self.down_0_1 = DownSample4D(channels, 2, 2, if_bn, bias, act)
        self.down_0_2 = DownSample4D(channels, 4, 2, if_bn, bias, act)

        self.up_1_0 = UpSample(2 * channels, 2, 2, if_bn, bias, act)
        self.up_2_0 = UpSample(4 * channels, 4, 2, if_bn, bias, act)

        self.SAS_0 = MultiSA(sa_num, channels, if_bn, bias, act)
        self.SAS_1 = MultiSA(sa_num, 2 * channels, if_bn, bias, act)
        self.SAS_2 = MultiSA(sa_num, 4 * channels, if_bn, bias, act)


        self.SASD_0 = MultiSAD(sad_num, channels, if_bn, bias, act)
        self.SASD_1 = MultiSAD(sad_num, 2 * channels, if_bn, bias, act)
        self.SASD_2 = MultiSAD(sad_num, 4 * channels, if_bn, bias, act)

        self.ca = ca_layer(3 * channels, 8, bias, act)
        self.down = Spacial2D(3 * channels, channels, if_bn=if_bn, bias=bias, act=act)

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = x.shape

        x0 = x
        x1 = self.down_0_1(x)
        x2 = self.down_0_2(x)

        x0 = self.SAS_0(x0)
        x1 = self.SAS_1(x1)
        x2 = self.SAS_2(x2)

        x0 = self.SASD_0(x0).reshape(batch, channel, height, width)
        x1 = self.SASD_1(x1).reshape(batch, 2 * channel, height // 2, width // 2)
        x2 = self.SASD_2(x2).reshape(batch, 4 * channel, height // 4, width // 4)

        x1 = self.up_1_0(x1)
        x2 = self.up_2_0(x2)


        x_g = torch.cat((x0, x1, x2), dim=1)
        x = self.down(self.ca(x_g))
        del x_g, x0, x1, x2

        return x


class FocusFeature(nn.Module):
    def __init__(self, channels, if_bn=False, bias=True, act='prelu'):
        super(FocusFeature, self).__init__()
        self.prepare = Spacial3D(3, channels, if_bn=if_bn, bias=bias, act=act)
        m = []
        num = 5
        for i in range(num):
            m.append(Spacial3D(channels, channels, 3, 1, 1, if_bn=if_bn, bias=bias, act=act))
        m.append(Spacial3D(channels, 1, 3, 1, 1, if_bn=if_bn, bias=bias, act=act))
        self.feature = nn.Sequential(*m)
        self.channels = channels

    def forward(self, x):
        b, d, h, w, c = x.shape
        x = x.permute(0, 4, 1, 2, 3)
        x = self.prepare(x)
        x = self.feature(x)
        x = F.softmax(x, dim=2)
        return x


class CAttention(nn.Module):
    def __init__(self, channels, if_bn=False, bias=True, act='prelu'):
        super(CAttention, self).__init__()
        m = []
        num = 3
        for i in range(num):
            m.append(Spacial2D(2 * channels, 2 * channels, if_bn=if_bn, bias=bias, act=act))
            m.append(ca_layer(2 * channels, 8, bias, act))
        self.ca = nn.Sequential(*m)
        self.down = Spacial2D(2 * channels, channels, if_bn=if_bn, bias=bias, act=act)

    def forward(self, x):
        x = self.ca(x)
        x = self.down(x)
        return x

class Finaloutput(nn.Module):
    def __init__(self, channels, if_bn=False, bias=True, act='prelu'):
        super(Finaloutput, self).__init__()
        self.ED = EncoderDecoder(channels, if_bn, bias, act)
        m = []
        m.append(nn.Conv2d(channels, 3, 3, 1, 1, 1, bias=bias))
        m.append(nn.ReLU())
        self.conv_final = nn.Sequential(*m)

    def forward(self, x):
        # batch, channels, height, width = x.shape
        x = self.ED(x)
        x = self.conv_final(x)

        return x


class GFinaloutput(nn.Module):
    def __init__(self, channels, if_bn=False, bias=True, act='prelu'):
        super(GFinaloutput, self).__init__()
        m = []
        m.append(Spacial2D(3, channels, if_bn=if_bn, bias=bias, act=act))
        num = 3
        for i in range(num):
            m.append(Spacial2D(channels, channels, if_bn=if_bn, bias=bias, act=act))
        self.conv = nn.Sequential(*m)
        self.down = Spacial2D(channels, 1, if_bn=if_bn, bias=bias, act='relu')

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.down(x1)
        return x1,x2


class ca(nn.Module):
    def __init__(self, channel, reduction=8, bias=True, act='prelu'):
        super(ca, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias),
            activate(act),
            nn.Linear(channel // reduction, channel, bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channel, height, width = x.shape
        y = self.avg_pool(x).reshape(batch, channel)
        # print(y.shape)
        y = self.mlp(y).reshape(batch, channel, 1, 1)
        return y


class sa(nn.Module):
    def __init__(self, kernel_size=7):
        super(sa, self).__init__()

        assert kernel_size in (3, 5, 7), 'kernel size must be 3 or 7'
        if kernel_size == 7:
            padding = 3
        if kernel_size == 5:
            padding = 2
        if kernel_size == 3:
            padding = 1
        # padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return self.sigmoid(y)


class DeRefLF_res(nn.Module):
    def __init__(self, opt):
        super(DeRefLF_res, self).__init__()

        self.channels = opt.channel
        self.height_view = opt.height_view
        self.width_view = opt.width_view
        self.sa_num = opt.sa_num
        self.sad_num = opt.sad_num
        self.if_bn = opt.if_bn
        self.bias = False if self.if_bn else True
        self.act = 'prelu'

        self.PrepareInput = PrepareInput(self.channels, self.if_bn, self.bias, self.act)
        self.MultiSAS = MultiSAS(self.channels, self.sa_num, self.sad_num, self.if_bn, self.bias,
                                 self.act)

        self.B_Focus3D = FocusFeature(self.channels, self.if_bn, self.bias, self.act)
        self.B_G = GFinaloutput(self.channels, self.if_bn, self.bias, self.act)
        self.B_CA0 = ca(self.channels, 8, self.bias, self.act)
        self.B_SA0 = sa(7)
        self.B_CA = CAttention(self.channels, self.if_bn, self.bias, self.act)

        self.B_Finaloutput = Finaloutput(self.channels, self.if_bn, self.bias, self.act)

        self.R_Focus3D = FocusFeature(self.channels, self.if_bn, self.bias, self.act)
        self.R_G = GFinaloutput(self.channels, self.if_bn, self.bias, self.act)
        self.R_CA0 = ca(self.channels, 8, self.bias, self.act)
        self.R_SA0 = sa(7)
        self.R_CA = CAttention(self.channels, self.if_bn, self.bias, self.act)

        self.R_Finaloutput = Finaloutput(self.channels, self.if_bn, self.bias, self.act)

    def filter_convolution(self, dynamic_filter, x):
        # dynamic filter: batch, 1, d, height, width
        # batch, channel, d, height, width = list(x.shape)
        [batch, d, height, width, channel] = x.shape
        x = x.permute(0, 4, 1, 2, 3)
        product = torch.mul(x, dynamic_filter)
        product = torch.sum(product, dim=2)  # batch, channel, 1, height, width
        product = product.reshape(batch, channel, height, width)
        return product

    def forward(self, input, focus_data):
        # [batch, height_view, width_view, height, width, channels] = input.shape
        prepared_input = self.PrepareInput(input)
        del input
        multiSAS_feature = self.MultiSAS(prepared_input)
        del prepared_input

        BG_feature = self.B_Focus3D(focus_data)
        BG_feature = self.filter_convolution(BG_feature, focus_data)
        B_feature,B_G = self.B_G(BG_feature)

        B_feature = torch.cat((B_feature, multiSAS_feature), dim=1)
        B_feature = self.B_CA(B_feature)
        B_CA = self.B_CA0(B_feature)
        B_SA = self.B_SA0(B_G)
        B_weight = B_CA * B_SA
        B_feature = B_feature * (1+B_weight)

        B = self.B_Finaloutput(B_feature)

        RG_feature = self.R_Focus3D(focus_data)
        RG_feature = self.filter_convolution(RG_feature, focus_data)
        R_feature,R_G = self.R_G(RG_feature)

        R_feature = torch.cat((R_feature, multiSAS_feature), dim=1)
        R_feature = self.R_CA(R_feature)
        R_CA = self.R_CA0(R_feature)
        R_SA = self.R_SA0(R_G)
        R_weight = R_CA * R_SA
        R_feature = R_feature * (1+R_weight)
        R = self.R_Finaloutput(R_feature)

        B = B.permute(0, 2, 3, 1)
        R = R.permute(0, 2, 3, 1)
        return B, R, B_G, R_G


class DeRefLF(nn.Module):
    def __init__(self, opt):
        super(DeRefLF, self).__init__()
        self.B = DeRefLF_res(opt)

    def forward(self, input, focus_data):
        B, R, B_G, R_G = self.B(input, focus_data)
        return B, R, B_G, R_G
