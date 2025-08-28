import torch
from torch import nn
import torch.nn.functional as F
from losses.vgg import VGGLoss

from math import exp

def L1_loss(pred_data, gt_data, device, opt, loss_count=0):
    criterion = torch.nn.L1Loss().to(device)
    l1_loss = opt.loss_weight[loss_count] * criterion(pred_data, gt_data)
    return l1_loss


def Gradient_loss(pred_data, gt_data, device, opt, loss_count=0):
    grad_loss = opt.loss_weight[loss_count] * gradient_loss(pred_data, gt_data, device)
    return grad_loss


def Vgg_loss(vgg, pred_data, gt_data, device, opt, loss_count=0):
    loss_vgg = VGGLoss(device=device, vgg=vgg, normalize=False)
    pred_data_tmp = pred_data.permute(0, 3, 1, 2).contiguous()
    gt_data_tmp = gt_data.permute(0, 3, 1, 2).contiguous()
    vgg_loss = opt.loss_weight[loss_count] * loss_vgg(pred_data_tmp, gt_data_tmp)

    return vgg_loss


def MY_LOSS_BR(B_pred_data, B_gt_data, R_pred_data, R_gt_data, B_pred_gradient, R_pred_gradient, factor, device, opt,
               vgg):
    B_L1_loss = L1_loss(B_pred_data, B_gt_data, device, opt, loss_count=0)
    B_Gradient_loss = Gradient_loss(B_pred_data, B_gt_data, device, opt, loss_count=1)
    B_Vgg_loss = Vgg_loss(vgg, B_pred_data, B_gt_data, device, opt, loss_count=2)

    B_gt_gradient = gradient(B_gt_data, device)
    B_Gradient_loss0 = L1_loss(B_pred_gradient, B_gt_gradient, device, opt, loss_count=6)
    B_loss = B_L1_loss + B_Gradient_loss + B_Vgg_loss+B_Gradient_loss0

    R_L1_loss = L1_loss(R_pred_data, R_gt_data, device, opt, loss_count=3)
    R_Gradient_loss = Gradient_loss(R_pred_data, R_gt_data, device, opt, loss_count=4)
    R_Vgg_loss = Vgg_loss(vgg, R_pred_data, R_gt_data, device, opt, loss_count=5)

    R_gt_gradient = gradient(R_gt_data, device)
    R_Gradient_loss0 = L1_loss(R_pred_gradient, R_gt_gradient, device, opt, loss_count=7)
    R_loss = R_L1_loss + R_Gradient_loss + R_Vgg_loss+R_Gradient_loss0

    factor = factor.reshape((len(factor), 1, 1, 1))

    I_pred_data = B_pred_data * factor + R_pred_data * (1 - factor)
    I_gt_data = B_gt_data * factor + R_gt_data * (1 - factor)
    I_L1_loss = L1_loss(I_pred_data, I_gt_data, device, opt, loss_count=6)

    loss = B_loss + R_loss + I_L1_loss

    return loss, B_L1_loss, B_Gradient_loss, B_Vgg_loss, R_L1_loss, R_Gradient_loss, R_Vgg_loss, I_L1_loss,B_Gradient_loss0,R_Gradient_loss0


class Gradient_Net(nn.Module):
    def __init__(self, device):
        super(Gradient_Net, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x, padding=1)
        grad_y = F.conv2d(x, self.weight_y, padding=1)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient


def gradient(x, device):
    b, h, w, c = x.shape
    #  rgb2gray       rgb 0.3 , 0.59 , 0.11
    if c == 3:
        x = 0.3 * x[:, :, :, 0] + 0.59 * x[:, :, :, 1] + 0.11 * x[:, :, :, 2]
    x = x.unsqueeze(1)
    gradient_model = Gradient_Net(device).to(device)
    g = gradient_model(x)
    return g


def gradient_loss(pred_data, gt_data, device):
    pred_data_gradient = gradient(pred_data, device)
    gt_data_gradient = gradient(gt_data, device)
    criterion = torch.nn.L1Loss().to(device)
    loss = criterion(pred_data_gradient, gt_data_gradient)
    return loss











def regularization_loss(model, weight_decay, p=2):
    reg_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            l2_reg = torch.norm(param, p=p)
            reg_loss = reg_loss + l2_reg

    reg_loss = weight_decay * reg_loss
    return reg_loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


