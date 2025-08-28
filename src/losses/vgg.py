import torch.nn as nn
import torch
import ssl
from torchvision import models
import torch.nn.functional as F
ssl._create_default_https_context = ssl._create_unverified_context

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        # path_checkpoint = f'../pretrained/vgg19-dcbb9e9d.pth'
        # checkpoint = torch.load(path_checkpoint)
        # model=models.vgg19(pretrained=False)
        # model.load_state_dict(checkpoint)
        # self.vgg_pretrained_features = model.features
        self.vgg_pretrained_features = models.vgg19(pretrained=True).features
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
        if indices is None:
            indices = [2, 7, 12, 21, 30]
        out = []
        # indices = sorted(indices)

        # for i in range(indices[-1]):
        #     if (i + 1) in indices:
        #         X = self.vgg_pretrained_features[i](X)
        #         out.append(X)
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            if (i+1) in indices:
                out.append(X)
        return out


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class VGGLoss(nn.Module):
    def __init__(self, device,vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss, self).__init__()
        if vgg is None:
            self.vgg = Vgg19().to(device)
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss().to(device)
        self.weights = weights or [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10 / 1.5]
        self.indices = indices or [2, 7, 12, 21, 30]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).to(device)
        else:
            self.normalize = None

    def forward(self, x, y):
            if self.normalize is not None:
                x = self.normalize(x)
                y = self.normalize(y)

            x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
            loss = 0

            for i in range(len(x_vgg)):
                #loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
                loss = loss+self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
                # print('test start')
                # loss.sum().backward()
                # print('test end')
            #return loss.item()
            return loss

class CRVGGLoss(nn.Module):
    def __init__(self, device,vgg=None, weights=None, indices=None, normalize=True):
        super(CRVGGLoss, self).__init__()
        if vgg is None:
            self.vgg = Vgg19().to(device)
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss().to(device)
        self.weights = weights or [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10 / 1.5]
        self.indices = indices or [2, 7, 12, 21, 30]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).to(device)
        else:
            self.normalize = None

    def forward(self, positive, anchor,negative):
            if self.normalize is not None:
                positive = self.normalize(positive)
                anchor = self.normalize(anchor)
                negative = self.normalize(negative)
            positive_vgg= self.vgg(positive, self.indices)
            anchor_vgg = self.vgg(anchor, self.indices)
            negative_vgg = self.vgg(negative, self.indices)
            loss = 0

            for i in range(len(positive_vgg)):
                loss = loss+self.weights[i] * (self.criterion(positive_vgg[i].detach(), anchor_vgg[i])/self.criterion(negative_vgg [i].detach(), anchor_vgg[i]))

            return loss


class VGGPrepareInput(nn.Module):
    def __init__(self, device,vgg=None):
        super(VGGPrepareInput, self).__init__()
        if vgg is None:
            self.vgg = Vgg19().to(device)
        else:
            self.vgg = vgg

    def forward(self, x):
        hypercolumn = self.vgg(x)
        _, C, H, W = x.shape
        hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for feature in
                       hypercolumn]
        input_i = [x]
        input_i.extend(hypercolumn)
        input_i = torch.cat(input_i, dim=1)
        return input_i
