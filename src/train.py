import torch
from torch.utils.data import DataLoader
from evaluation_index.evaluation_index import ProgressMeter, Meter, StrMeter, AverageMeterJustValue, \
    AverageMeterJustAVG, visdomMeter
import time
import os
import sys
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from losses.vgg import Vgg19
from initializers.initializers import weights_init
from scheduler.myscheduler import myLR, clip_gradient
from utils import utils
from math import log10
from option.train_options import TrainOptions
from losses.myloss import MY_LOSS_BR
from dataload.dataload import DeRefLF_Train_Dataset, DeRefLF_Test_Dataset
from model.model import DeRefLF


def train_main():
    # Train parameters
    opt = TrainOptions().parse()
    device_ids = opt.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    ''' Define Model(set parameters)'''
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    start_time = time.strftime('%m%d%H', time.localtime(time.time()))

    model_name = opt.model_name
    model = globals()[model_name](opt)

    utils.get_parameter_number(model)
    model.apply(weights_init('xavier'))

    if len(opt.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.base_lr)
    scheduler = myLR(optimizer)

    ''' Create the save path for training models and record the training and test loss'''
    if opt.dir_model is not None:
        ''' Loading the trained model'''
        path_checkpoint = opt.dir_model + f'{opt.model_name}_{opt.current_iter}.pkl'
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        opt.current_iter = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print('Loading the trained model',
              opt.dir_model + f'{opt.model_name}_{opt.current_iter}.pkl')
    else:
        task_name = f'{opt.model_name}_{opt.base_lr}_{opt.batch_size}_{opt.crop_size}_{opt.up_size}_{opt.channel}_{start_time}'
        dir_model = f'{opt.NetworkSave}/{task_name}'

        print('dir_model:', dir_model)
        if not os.path.exists(dir_model):
            os.makedirs(dir_model)
        else:
            print('Exist!')

    best_psnr = 0
    test_psnr = 0
    best_psnr_new = 0
    LOSS = visdomMeter('loss', 1)
    PSNR = visdomMeter('PSNR', 10)

    # load training dataset
    train_dataload_name = opt.train_dataload_name
    train_dataset = globals()[train_dataload_name](opt)

    # load test dataset
    test_dataload_name = opt.test_dataload_name
    test_dataset = globals()[test_dataload_name](opt)

    vgg = Vgg19(requires_grad=False).to(device)

    for epoch in range(opt.current_iter, opt.MAX_EPOCH):

        ''' Validation during the training process'''
        if epoch % 10 == 0:
            best_psnr_new, test_psnr = test_res(test_dataset, model, best_psnr, device, opt)
            if len(opt.gpu_ids) > 1:
                checkpoint = {
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch
                }
            else:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch
                }
            torch.save(checkpoint, dir_model + f'/{opt.model_name}_{epoch}.pkl')

            if best_psnr_new > best_psnr:
                torch.save(checkpoint, dir_model + f'/{opt.model_name}_best.pkl')

            best_psnr = best_psnr_new
            torch.cuda.empty_cache()
        ''' Training begin'''
        train_loss = train_res(train_dataset, model, epoch, optimizer, device, opt, vgg)
        LOSS.update(train_loss)
        PSNR.update(test_psnr)
        scheduler.step()


def train_res(train_dataset, model, epoch, optimizer, device, opt, vgg):
    time_start = time.time()
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=False)

    epoch_time = Meter('Time', ':6.3f')
    LOSS = AverageMeterJustAVG('Loss', ':3.6f')
    B_PSNR = AverageMeterJustAVG('B_PSNR', ':6.3f')
    R_PSNR = AverageMeterJustAVG('R_PSNR', ':6.3f')
    LR = Meter('LR', ':.2e')

    B_L1_LOSS = AverageMeterJustAVG('B_L1_loss', ':3.6f')
    B_Gradient_LOSS = AverageMeterJustAVG('B_Gradient_loss', ':3.6f')
    B_Vgg_LOSS = AverageMeterJustAVG('B_Vgg_loss', ':3.6f')
    R_L1_LOSS = AverageMeterJustAVG('R_L1_loss', ':3.6f')
    I_L1_LOSS = AverageMeterJustAVG('I_L1_loss', ':3.6f')
    R_Gradient_LOSS = AverageMeterJustAVG('R_Gradient_loss', ':3.6f')
    R_Vgg_LOSS = AverageMeterJustAVG('R_Vgg_loss', ':3.6f')

    B_Gradient_LOSS0 = AverageMeterJustAVG('B_Gradient_loss0', ':3.6f')
    R_Gradient_LOSS0 = AverageMeterJustAVG('R_Gradient_loss0', ':3.6f')

    progress = ProgressMeter(opt.MAX_EPOCH,
                             [LR, LOSS, B_PSNR, R_PSNR, epoch_time, B_L1_LOSS, B_Gradient_LOSS, B_Vgg_LOSS, R_L1_LOSS,
                              R_Gradient_LOSS, R_Vgg_LOSS, I_L1_LOSS, B_Gradient_LOSS0, R_Gradient_LOSS0],
                             prefix=f'Epoch: [{epoch}]')

    optimizer.zero_grad()
    for i, (train_data, B_gt_data, R_gt_data, focus_data, factor) in enumerate(train_loader):
        train_data, B_gt_data, R_gt_data, focus_data, factor = train_data.to(device), B_gt_data.to(
            device), R_gt_data.to(
            device), focus_data.to(device), factor.to(device)

        # Forward pass: Compute predicted y by passing x to the model
        B_pred_data, R_pred_data, B_pred_gradient, R_pred_gradient = model(train_data, focus_data)

        loss, B_L1_loss, B_Gradient_loss, B_Vgg_loss, R_L1_loss, R_Gradient_loss, R_Vgg_loss, I_L1_loss, B_Gradient_loss0, R_Gradient_loss0 = MY_LOSS_BR(
            B_pred_data, B_gt_data, R_pred_data, R_gt_data, B_pred_gradient, R_pred_gradient, factor, device, opt, vgg)

        B_loss_mse = torch.nn.MSELoss().to(device)
        B_loss_mse2 = B_loss_mse(B_pred_data, B_gt_data)
        B_psnr = 10 * log10(1 / B_loss_mse2.item())
        B_psnr = torch.from_numpy(np.array(B_psnr))

        R_loss_mse = torch.nn.MSELoss().to(device)
        R_loss_mse2 = R_loss_mse(R_pred_data, R_gt_data)
        R_psnr = 10 * log10(1 / R_loss_mse2.item())
        R_psnr = torch.from_numpy(np.array(R_psnr))

        LOSS.update(loss.item(), train_data.size(0))

        B_PSNR.update(B_psnr.item(), train_data.size(0))
        R_PSNR.update(R_psnr.item(), train_data.size(0))

        B_L1_LOSS.update(B_L1_loss.item(), train_data.size(0))
        R_L1_LOSS.update(R_L1_loss.item(), train_data.size(0))
        I_L1_LOSS.update(I_L1_loss.item(), train_data.size(0))

        B_Gradient_LOSS.update(B_Gradient_loss.item(), train_data.size(0))
        R_Gradient_LOSS.update(R_Gradient_loss.item(), train_data.size(0))
        B_Gradient_LOSS0.update(B_Gradient_loss0.item(), train_data.size(0))
        R_Gradient_LOSS0.update(R_Gradient_loss0.item(), train_data.size(0))

        B_Vgg_LOSS.update(B_Vgg_loss.item(), train_data.size(0))
        R_Vgg_LOSS.update(R_Vgg_loss.item(), train_data.size(0))

        loss = loss / opt.accumulation_steps
        loss.backward()

        if (i + 1) % opt.accumulation_steps == 0:
            if opt.clip_value is not None:
                clip_gradient(optimizer, opt.clip_value)
            optimizer.step()
            optimizer.zero_grad()

    LR.update(optimizer.state_dict()['param_groups'][0]['lr'])
    epoch_time.update(time.time() - time_start)

    progress.display(epoch)
    return LOSS.avg


def test_res(test_dataset, model, best_psnr, device, opt):
    time_start = time.time()
    model.eval()
    border_crop = 15
    [train_data_list, B_gt_data_list, R_gt_data_list, image_path_list] = test_dataset
    num = len(image_path_list)

    B_PSNR = AverageMeterJustValue('B_PSNR', ':6.4f')
    B_SSIM = AverageMeterJustValue('B_SSIM', ':6.4f')
    R_PSNR = AverageMeterJustValue('R_PSNR', ':6.4f')
    R_SSIM = AverageMeterJustValue('R_SSIM', ':6.4f')
    IMG_NAME = StrMeter('IMG_NAME', ':s')
    progress = ProgressMeter(num, [B_PSNR, B_SSIM, R_PSNR, R_SSIM, IMG_NAME], prefix='Test: ')
    for idx in range(num):
        train_data = train_data_list[idx]
        B_gt_data = B_gt_data_list[idx]
        R_gt_data = R_gt_data_list[idx]
        image_path = image_path_list[idx]
        train_data = train_data[np.newaxis, :, :, :, :, :]
        train_data = torch.from_numpy(train_data).to(device)
        focus_data = utils.create_F(train_data, opt.min, opt.max, opt.step)
        with torch.no_grad():
            # Forward pass: Compute predicted y by passing x to the model
            b, u, v, h, w, c = train_data.shape
            h_res = h % 8
            w_res = w % 8
            h = h - h_res
            w = w - w_res
            border_crop_h = border_crop - h_res
            border_crop_w = border_crop - w_res

            B_gt_pred, R_gt_pred, _, _ = model(train_data[:, :, :, 0:h, 0:w, :], focus_data[:, :, 0:h, 0:w, :])

            B_gt_pred_tmp = B_gt_pred[0, border_crop:-border_crop_h, border_crop:-border_crop_w,
                            :].cpu().numpy()
            B_gt_data = B_gt_data[border_crop:-border_crop, border_crop:-border_crop, :]
            B_gt_pred_tmp = np.clip(B_gt_pred_tmp, 0, 1)
            B_gt_data = np.clip(B_gt_data, 0, 1)

            R_gt_pred_tmp = R_gt_pred[0, border_crop:-border_crop_h, border_crop:-border_crop_w,
                            :].cpu().numpy()
            R_gt_data = R_gt_data[border_crop:-border_crop, border_crop:-border_crop, :]
            R_gt_pred_tmp = np.clip(R_gt_pred_tmp, 0, 1)
            R_gt_data = np.clip(R_gt_data, 0, 1)

            B_psnr = peak_signal_noise_ratio(B_gt_pred_tmp, B_gt_data, data_range=1)
            B_ssim = structural_similarity(B_gt_pred_tmp, B_gt_data, data_range=1, multichannel=True)
            R_psnr = peak_signal_noise_ratio(R_gt_pred_tmp, R_gt_data, data_range=1)
            R_ssim = structural_similarity(R_gt_pred_tmp, R_gt_data, data_range=1, multichannel=True)
            B_PSNR.update(B_psnr)
            B_SSIM.update(B_ssim)
            R_PSNR.update(R_psnr)
            R_SSIM.update(R_ssim)
            IMG_NAME.update(image_path)
        progress.display(idx)
    PSNR = (B_PSNR.avg + R_PSNR.avg) / 2
    if PSNR > best_psnr:
        best_psnr = PSNR

    print(
        f'===> PSNR:{PSNR:.4f} dB / BEST {best_psnr:.4f} dB '
        f'B_PSNR: {B_PSNR.avg:.4f} dB B_SSIM: {B_SSIM.avg:.4f} R_PSNR: {R_PSNR.avg:.4f} dB R_SSIM: {R_SSIM.avg:.4f} '
        f'Time: {time.time() - time_start:.6f}')
    return best_psnr, PSNR


if __name__ == '__main__':
    train_main()
