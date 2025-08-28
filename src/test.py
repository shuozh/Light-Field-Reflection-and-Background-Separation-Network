import torch
import os
import time
import numpy as np
import pandas as pd
import sys
import cv2
from evaluation_index.evaluation_index import quality_assess, ProgressMeter, StrMeter, AverageMeterJustValue
from utils import utils
import torch.nn as nn
from option.test_options import TestOptions
from model.model import DeRefLF

class Logger:
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def test_main():
    opt = TestOptions().parse()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    if len(opt.gpu_ids) > 1:
        gpu_ids = opt.gpu_ids.split(',')
        for i in range(len(gpu_ids)):
            gpu_ids[i] = int(gpu_ids[i])
        device_ids = gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print('=' * 40)
    print('create save directory...')
    dir_result = os.path.join('../', opt.dir_result, opt.name)
    if not os.path.exists(dir_result):
        os.makedirs(dir_result)
    sys.stdout = Logger(os.path.join(dir_result, f'test_{int(time.time())}.log'), sys.stdout)
    print('done')
    print('=' * 40)
    print('build network...')

    model_name = opt.model_name
    model = globals()[model_name](opt)

    utils.get_parameter_number(model)
    if len(opt.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        model.to(device)

    model.eval()
    print('done')
    print('=' * 40)
    print('load model...')
    path_checkpoint = opt.dir_model + f'{opt.model_name}_best.pkl'
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint)

    print('done')
    print('=' * 40)
    print('predict image...')

    xls_list = []
    border_crop = 15
    files = os.listdir(opt.dir_test_images)
    files.sort()
    num = len(files)

    B_PSNR = AverageMeterJustValue('B_PSNR', ':6.4f')
    B_SSIM = AverageMeterJustValue('B_SSIM', ':6.4f')
    B_LMSE = AverageMeterJustValue('B_LMSE', ':6.4f')
    B_NCC = AverageMeterJustValue('B_NCC', ':6.4f')
    R_PSNR = AverageMeterJustValue('R_PSNR', ':6.4f')
    R_SSIM = AverageMeterJustValue('R_SSIM', ':6.4f')
    R_LMSE = AverageMeterJustValue('R_LMSE', ':6.4f')
    R_NCC = AverageMeterJustValue('R_NCC', ':6.4f')
    TIME = AverageMeterJustValue('TIME', ':6.4f')
    IMG_NAME = StrMeter('IMG_NAME', ':s')
    progress = ProgressMeter(num,
                             [B_PSNR, B_SSIM, B_LMSE, B_NCC, R_PSNR, R_SSIM, R_LMSE, R_NCC, TIME, IMG_NAME],
                             prefix='Test: ')

    for index, image_name in enumerate(files):
        ori_data, B_gt_data, R_gt_data = utils.load_test(opt.dir_test_images + image_name)
        [height_view, width_view, height, width, channels] = ori_data.shape
        ori_data = torch.from_numpy(ori_data).cuda()
        ori_data = ori_data[np.newaxis, :, :, :, :, :]
        focus_data = utils.create_F(ori_data, opt.min, opt.max, opt.step)
        with torch.no_grad():
            time_item_start = time.time()
            b, u, v, h, w, c = ori_data.shape
            h_res = h % 8
            w_res = w % 8
            h = h - h_res
            border_crop_h = border_crop - h_res
            w = w - w_res
            border_crop_w = border_crop - w_res
            B_pred, R_pred, BG_pred, RG_pred = model(ori_data[:, :, :, 0:h, 0:w, :], focus_data[:, :, 0:h, 0:w, :])
            time_ = time.time() - time_item_start
            TIME.update(time_)

            BG_pred = BG_pred.permute(0, 2, 3, 1)
            RG_pred = RG_pred.permute(0, 2, 3, 1)
            BG_pred = BG_pred[0, border_crop:-border_crop_h, border_crop:-border_crop_w, :].cpu().numpy()
            RG_pred = RG_pred[0, border_crop:-border_crop_h, border_crop:-border_crop_w, :].cpu().numpy()

            B_pred_tmp = B_pred[0, border_crop:-border_crop_h, border_crop:-border_crop_w, :].cpu().numpy()
            B_gt_data = B_gt_data[border_crop:-border_crop, border_crop:-border_crop, :]
            R_pred_tmp = R_pred[0, border_crop:-border_crop_h, border_crop:-border_crop_w, :].cpu().numpy()
            R_gt_data = R_gt_data[border_crop:-border_crop, border_crop:-border_crop, :]
            ori_data_center = ori_data[0, int(height_view // 2), int(width_view // 2), border_crop:-border_crop,
                              border_crop:-border_crop, :].cpu().numpy()

            B_pred_tmp = np.clip(B_pred_tmp, 0, 1)
            B_gt_data = np.clip(B_gt_data, 0, 1)
            R_pred_tmp = np.clip(R_pred_tmp, 0, 1)
            R_gt_data = np.clip(R_gt_data, 0, 1)
            ori_data_center = np.clip(ori_data_center, 0, 1)

            dir_result_img = os.path.join(dir_result, 'img/')
            if not os.path.exists(dir_result_img):
                os.makedirs(dir_result_img)
            result_image_path = dir_result_img + image_name

            cv2.imwrite(os.path.join(result_image_path + '_B_gt.png'), B_gt_data[:, :, ::-1] * 255)
            cv2.imwrite(os.path.join(result_image_path + '_B_pred.png'), B_pred_tmp[:, :, ::-1] * 255)
            cv2.imwrite(os.path.join(result_image_path + '_R_gt.png'), R_gt_data[:, :, ::-1] * 255)
            cv2.imwrite(os.path.join(result_image_path + '_R_pred.png'), R_pred_tmp[:, :, ::-1] * 255)
            cv2.imwrite(os.path.join(result_image_path + '_ori.png'), ori_data_center[:, :, ::-1] * 255)
            cv2.imwrite(os.path.join(result_image_path + '_RG.png'), RG_pred * 255)
            cv2.imwrite(os.path.join(result_image_path + '_BG.png'), BG_pred * 255)

            B_psnr, B_ssim, B_lmse, B_ncc = quality_assess(B_pred_tmp, B_gt_data)
            R_psnr, R_ssim, R_lmse, R_ncc = quality_assess(R_pred_tmp, R_gt_data)
            B_PSNR.update(B_psnr)
            B_SSIM.update(B_ssim)
            B_LMSE.update(B_lmse)
            B_NCC.update(B_ncc)
            R_PSNR.update(R_psnr)
            R_SSIM.update(R_ssim)
            R_LMSE.update(R_lmse)
            R_NCC.update(R_ncc)
            IMG_NAME.update(image_name)
            progress.display(index)
            xls_list.append([image_name, B_psnr, B_ssim, B_lmse, B_ncc, R_psnr, R_ssim, R_lmse, R_ncc, time_])
        torch.cuda.empty_cache()

    print(
        f'Average B_PSNR: {B_PSNR.avg:.4f} B_SSIM: {B_SSIM.avg:.4f} B_LMSE: {B_LMSE.avg:.4f} B_NCC: {B_NCC.avg:.4f} R_PSNR: {R_PSNR.avg:.4f} R_SSIM: {R_SSIM.avg:.4f} R_LMSE: {R_LMSE.avg:.4f} R_NCC: {R_NCC.avg:.4f}time: {TIME.avg:.4f}')
    xls_list.append(
        ['average', B_PSNR.avg, B_SSIM.avg, B_LMSE.avg, B_NCC.avg, R_PSNR.avg, R_SSIM.avg, R_LMSE.avg,
         R_NCC.avg, TIME.avg])
    xls_list = np.array(xls_list)
    result = pd.DataFrame(xls_list,
                          columns=['image', 'B_psnr', 'B_ssim', 'B_lmse', 'B_ncc', 'R_psnr', 'R_ssim', 'R_lmse',
                                   'R_ncc', 'time'])
    result.to_csv(os.path.join(dir_result + f'/result_{int(time.time())}.csv'))
    print('all done')

if __name__ == '__main__':
    test_main()
