import torch
import torch.nn as nn
import numpy as np



def activate(act='prelu'):
    if act == 'prelu':
        return nn.PReLU()
    if act == 'relu':
        return nn.ReLU()


class Spacial2D(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=1, if_bn=False, bias=True, act='prelu'):
        super(Spacial2D, self).__init__()
        if out_channels == None:
            out_channels = in_channels
        m = []
        m.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        if if_bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(activate(act))
        self.spaconv = nn.Sequential(*m)

    def forward(self, x):
        # batch, channel, height, width = x.shape
        x = self.spaconv(x)
        return x


class Spacial3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=1, if_bn=False, bias=True,
                 act='prelu'):
        super(Spacial3D, self).__init__()
        if out_channels == None:
            out_channels = in_channels
        m = []
        m.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        if if_bn:
            m.append(nn.BatchNorm3d(out_channels))
        if act is not None:
            m.append(activate(act))
        self.spaconv = nn.Sequential(*m)

    def forward(self, x):
        # batch, channel, height, width = x.shape
        x = self.spaconv(x)
        return x



class ResidualDownSample(nn.Module):
    def __init__(self, channels, if_bn=False, bias=True, act='prelu'):
        super(ResidualDownSample, self).__init__()

        m = []
        m.append(nn.Conv2d(channels, 2 * channels, 3, 2, 1, bias=bias))
        if if_bn:
            m.append(nn.BatchNorm2d(2 * channels))
        if act is not None:
            m.append(activate(act))


        self.top = nn.Sequential(*m)


    def forward(self, x):
        top = self.top(x)
        return top


class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, if_bn=False, bias=True, act='prelu'):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualDownSample(in_channels, if_bn, bias, act))
            in_channels = int(in_channels * stride)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class ResidualUpSample(nn.Module):
    def __init__(self, channels, if_bn=False, bias=False, act='prelu'):
        super(ResidualUpSample, self).__init__()

        m = []
        m.append(nn.ConvTranspose2d(channels, channels // 2, 3, 2, 1, output_padding=1,
                                    bias=bias), )
        if if_bn:
            m.append(nn.BatchNorm2d(channels // 2))
        if act is not None:
            m.append(activate(act))

        self.top = nn.Sequential(*m)



    def forward(self, x):
        top = self.top(x)
        return top


class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, if_bn=False, bias=True, act='prelu'):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualUpSample(in_channels, if_bn, bias, act))
            in_channels = int(in_channels // stride)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class DownSample4D(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, if_bn=False, bias=True, act='prelu'):
        super(DownSample4D, self).__init__()
        self.down = DownSample(in_channels, scale_factor, stride, if_bn, bias, act)
        self.scale_factor = scale_factor

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = x.shape
        x = x.permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(batch * height_view * width_view, channel, height, width)
        x = self.down(x)
        x = x.reshape(batch, height_view, width_view, channel * self.scale_factor, int(height // self.scale_factor),
                      int(width // self.scale_factor))
        x = x.permute(0, 3, 1, 2, 4, 5)
        return x


class UpSample4D(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, if_bn=False, bias=True, act='prelu'):
        super(UpSample4D, self).__init__()
        self.up = UpSample(in_channels, scale_factor, stride, if_bn, bias, act)
        self.scale_factor = scale_factor

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = x.shape
        x = x.permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(batch * height_view * width_view, channel, height, width)
        x = self.up(x)
        x = x.reshape(batch, height_view, width_view, int(channel // self.scale_factor), height * self.scale_factor,
                      width * self.scale_factor)
        x = x.permute(0, 3, 1, 2, 4, 5)
        return x


class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True, act='prelu'):
        super(ca_layer, self).__init__()
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
        return x * y


class MultiSpacial2D(nn.Module):
    def __init__(self, channels, num, if_bn=False, bias=True, act='prelu'):
        super(MultiSpacial2D, self).__init__()
        m = []
        for i in range(num):
            m.append(Spacial2D(channels, channels, if_bn=if_bn, bias=bias,  act=act))
        self.top = nn.Sequential(*m)

    def forward(self, x):
        x = self.top(x)
        return x


class MultiSpacial3D(nn.Module):
    def __init__(self, channels, num, if_bn=False, bias=True, act='prelu'):
        super(MultiSpacial3D, self).__init__()
        m = []
        for i in range(num):
            m.append(Spacial3D(channels, channels, if_bn=if_bn, bias=bias, act=act))
        self.top = nn.Sequential(*m)

    def forward(self, x):
        x = self.top(x)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, channels, if_bn=False, bias=True, act='prelu'):
        super(EncoderDecoder, self).__init__()
        # self.channels = channels
        m = []
        m.append(MultiSpacial2D(channels, 2, if_bn, bias, act))
        self.encoder_1_0 = nn.Sequential(*m)
        m = []
        m.append(DownSample(channels, 2, 2, if_bn, bias, act))
        self.encoder_1_1 = nn.Sequential(*m)

        m = []
        m.append(MultiSpacial2D(2 * channels, 2, if_bn, bias, act))
        self.encoder_2_0 = nn.Sequential(*m)
        m = []
        m.append(DownSample(2 * channels, 2, 2, if_bn, bias, act))
        self.encoder_2_1 = nn.Sequential(*m)

        m = []
        m.append(MultiSpacial2D(4 * channels, 2, if_bn, bias, act))
        self.encoder_3_0 = nn.Sequential(*m)
        m = []
        m.append(DownSample(4 * channels, 2, 2, if_bn, bias, act))
        self.encoder_3_1 = nn.Sequential(*m)

        self.bottom = Spacial2D(channels * 8, channels * 8, 3,1,1,if_bn, bias, act)

        self.decoder_3_0 = UpSample(channels * 8, 2, 2, if_bn, bias, act)
        m = []
        m.append(Spacial2D(8 * channels, 4 * channels,3,1,1, if_bn, bias, act))
        m.append(MultiSpacial2D(4 * channels, 2, if_bn, bias, act))
        self.decoder_3_1 = nn.Sequential(*m)

        self.decoder_2_0 = UpSample(channels * 4, 2, 2, if_bn, bias, act)
        m = []
        m.append(Spacial2D(4 * channels, 2 * channels, 3,1,1,if_bn, bias, act))
        m.append(MultiSpacial2D(2 * channels, 2, if_bn, bias, act))
        self.decoder_2_1 = nn.Sequential(*m)

        self.decoder_1_0 = UpSample(channels * 2, 2, 2, if_bn, bias, act)
        m = []
        m.append(Spacial2D(2 * channels, channels, 3,1,1,if_bn, bias, act))
        m.append(MultiSpacial2D(channels, 2, if_bn, bias, act))
        self.decoder_1_1 = nn.Sequential(*m)

    def forward(self, x):
        # batch, channel, height, width = x.shape
        encoder_1_res = self.encoder_1_0(x)
        encoder_1 = self.encoder_1_1(encoder_1_res)

        encoder_2_res = self.encoder_2_0(encoder_1)
        encoder_2 = self.encoder_2_1(encoder_2_res)

        encoder_3_res = self.encoder_3_0(encoder_2)
        encoder_3 = self.encoder_3_1(encoder_3_res)

        bottom = self.bottom(encoder_3)

        decoder_3 = self.decoder_3_0(bottom)
        decoder_3 = torch.cat((decoder_3, encoder_3_res), dim=1)
        decoder_3 = self.decoder_3_1(decoder_3)

        decoder_2 = self.decoder_2_0(decoder_3)
        decoder_2 = torch.cat((decoder_2, encoder_2_res), dim=1)
        decoder_2 = self.decoder_2_1(decoder_2)

        decoder_1 = self.decoder_1_0(decoder_2)
        decoder_1 = torch.cat((decoder_1, encoder_1_res), dim=1)
        decoder_1 = self.decoder_1_1(decoder_1)

        return decoder_1


