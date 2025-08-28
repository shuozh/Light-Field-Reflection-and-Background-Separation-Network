import os
import numpy as np
import cv2
import random
import torch
import torch.nn.functional as F


def listdir_nohidden(path, type=False):
    list_nohidden = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            if type == False:
                list_nohidden.append(f)
            elif os.path.splitext(f)[1] == type:
                list_nohidden.append(f)
    return list_nohidden


def load_B_R_gt(B_dir, R_dir):
    if R_dir == None:
        B_gt_list = []
        B_list = listdir_nohidden(B_dir, type=False)
        B_list.sort()
        B_num = len(B_list)
        for i in range(B_num):
            img = load_img(B_dir + B_list[i])
            if img.max() < 0.15:
                print("Invalid file")
                continue
            B_gt_list.append(img)
            # if i == 3:
            #      break
        return B_gt_list
    else:
        R_gt_list = []
        B_gt_list = []
        B_list = listdir_nohidden(B_dir, type=False)
        B_list.sort()
        B_num = len(B_list)
        R_list = listdir_nohidden(R_dir, type=False)
        R_list.sort()
        R_num = len(R_list)
        for i in range(B_num):
            img = load_img(B_dir + B_list[i])
            if img.max() < 0.15:
                print("Invalid file")
                continue
            B_gt_list.append(img)

        for i in range(R_num):
            img = load_img(R_dir + R_list[i])
            if img.max() < 0.15:
                print("Invalid file")
                continue
            R_gt_list.append(img)
        return B_gt_list, R_gt_list



def create_B_R(B_img, R_img, crop_size=256, disparity=None):
    if disparity is not None:
        R_disparity = np.random.uniform(-disparity, -disparity / 2)
        R_img = view_warp(R_img, R_disparity)
        border = np.int(np.abs(np.floor(R_disparity)) * R_img.shape[0])
        R_img = R_img[:, :, border:-border, border:-border, :]

        B_disparity = np.random.uniform(disparity / 2, disparity)
        B_img = view_warp(B_img, B_disparity)
        border = np.int(np.abs(np.ceil(B_disparity)) * B_img.shape[0])
        B_img = B_img[:, :, border:-border, border:-border, :]

    B_img, _, _ = random_crop(B_img, crop_size)
    R_img, _, _ = random_crop(R_img, crop_size)

    factor = np.random.uniform(0.2, 0.8)
    img = B_img * factor + R_img * (1 - factor)
    u, v, h, w, c = img.shape

    return img / 255.0, B_img[u // 2, v // 2, :, :, :] / 255.0,  R_img[u // 2, v // 2, :, :, :] / 255.0, factor


def create_F(lf, min, max, step):
    focus = focus_img(lf, min, max, step)
    return focus


def load_img(dir):
    img_dirs = listdir_nohidden(dir, type=False)
    img_dirs.sort()
    num = np.int(np.sqrt(len(img_dirs)))
    h, w, c = cv2.imread(dir + '/' + img_dirs[0], cv2.IMREAD_COLOR).shape
    img = np.zeros((num, num, h, w, c))

    for i in range(num):
        for j in range(num):
            index = i * num + j
            img[i, j, :, :, :] = cv2.imread(dir + '/' + img_dirs[index], cv2.IMREAD_COLOR)
            img[i, j, :, :, :] = img[i, j, :, :, ::-1]
    return np.float32(img[:,:,50:-5, 50:-5, :])


def save_img(data, dir):
    u, v, h, w, c = data.shape
    num = 0
    for i in range(u):
        for j in range(v):
            cv2.imwrite(dir + '/' + '%03d' % num + '.png', data[i, j, :, :, ::-1])
            num = num + 1


def random_crop(data, crop_hsize, crop_wsize=None, top=None, left=None):
    if crop_wsize == None:
        crop_wsize = crop_hsize
    height, width = data.shape[-3], data.shape[-2]
    if top == None:
        top = np.random.randint(0, height - crop_hsize + 1)
    if left == None:
        left = np.random.randint(0, width - crop_wsize + 1)
    data_crop = data[..., top: top + crop_hsize, left: left + crop_wsize, :]
    return data_crop, top, left


def random_flip(data, random_tmp=None):
    if random_tmp == None:
        random_tmp = np.random.random()
    if len(data.shape) == 6:
        if random_tmp >= (2.0 / 3):
            data = np.flip(data, 3)
            data = np.flip(data, 1)
        elif random_tmp <= (1.0 / 3):
            data = np.flip(data, 4)
            data = np.flip(data, 2)
    if len(data.shape) == 5:
        if random_tmp >= (2.0 / 3):
            data = np.flip(data, 2)
            data = np.flip(data, 0)
        elif random_tmp <= (1.0 / 3):
            data = np.flip(data, 3)
            data = np.flip(data, 1)
    if len(data.shape) == 4:
        if random_tmp >= (2.0 / 3):
            data = np.flip(data, 1)
        elif random_tmp <= (1.0 / 3):
            data = np.flip(data, 2)
    if len(data.shape) == 3:
        if random_tmp >= (2.0 / 3):
            data = np.flip(data, 0)
        elif random_tmp <= (1.0 / 3):
            data = np.flip(data, 1)
    return data, random_tmp


def random_rotation(data, random_tmp=None):
    if random_tmp == None:
        random_tmp = np.random.choice(range(4))
    if len(data.shape) == 6:
        data = np.rot90(data, random_tmp, (1, 2))
        data = np.rot90(data, random_tmp, (3, 4))
    if len(data.shape) == 5:
        data = np.rot90(data, random_tmp, (0, 1))
        data = np.rot90(data, random_tmp, (2, 3))
    if len(data.shape) == 4:
        data = np.rot90(data, random_tmp, (1, 2))
    if len(data.shape) == 3:
        data = np.rot90(data, random_tmp, (0, 1))

    return data, random_tmp


def view_warp(lf, disparity):
    # warp LF with one specific disparity
    [ang_height, ang_width, height, width, channel] = lf.shape

    warped_lf = np.zeros((ang_height, ang_width, height, width, channel), dtype=np.float32)
    center_u = ang_height // 2
    center_v = ang_width // 2
    ang_u_move = list(range(-center_u, center_u + 1))
    ang_v_move = list(range(-center_v, center_v + 1))

    for u in range(ang_height):
        for v in range(ang_width):
            view = lf[u, v, :, :, :]
            u_move, v_move = ang_u_move[u] * disparity, ang_v_move[v] * disparity
            M = np.float32([[1, 0, v_move], [0, 1, u_move]])
            warped_view = cv2.warpAffine(view, M, (width, height))
            warped_lf[u, v, :, :, :] = warped_view

    return warped_lf


def view_warp_different(lf, disparity_num, step=1):
    [height_view, width_view, height, width, channel] = lf.shape
    disparity_max = disparity_num * step
    # disparity_list = range(-disparity_max, disparity_max + step, step)
    disparity_list = range(-disparity_max, 0, step)
    warped_LF = np.zeros((len(disparity_list), height_view, width_view, height, width, channel), dtype=np.float32)
    for i, disparity in enumerate(disparity_list):
        warped_LF[i, :, :, :, :, :] = view_warp(lf, disparity)
    return warped_LF


def view_warp_fast(lf, disparity):
    # warp LF with one specific disparity
    [batch, height_view, width_view, height, width, channel] = lf.shape

    lf_t = lf.reshape((-1, height, width, channel)).permute((0, 3, 1, 2))  # batch*u*v,c,h,w

    center_u = height_view // 2
    center_v = width_view // 2
    grid = []
    hh = torch.arange(0, height).view(1, height, 1).expand(batch, height, width)
    ww = torch.arange(0, width).view(1, 1, width).expand(batch, height, width)
    for u in range(height_view):
        for v in range(width_view):
            dispmap_u = -disparity * (u - center_u)
            dispmap_v = -disparity * (v - center_v)
            h_range = hh + dispmap_u
            w_range = ww + dispmap_v
            h_range = 2. * h_range / (height - 1) - 1
            w_range = 2. * w_range / (width - 1) - 1
            grid_t = torch.stack((w_range, h_range), dim=3)  # [batch,h,w,2]
            grid.append(grid_t)
    grid = torch.cat(grid, 0)  # [batch*u*v,h,w,2]

    warped_lf = F.grid_sample(lf_t, grid.type_as(lf_t), 'bilinear', padding_mode="zeros", align_corners=False)
    warped_lf = warped_lf.reshape((batch, height_view, width_view, channel, height, width)).permute(
        (0, 1, 2, 4, 5, 3))

    return warped_lf


def view_warp_different_fast(lf, min=-1, max=1, step=1):
    # warp LF with one specific disparity

    [batch, height_view, width_view, height, width, channel] = lf.shape
    disparity_list = np.arange(min, max + step, step)
    warped_LF = torch.zeros((batch, len(disparity_list), height_view, width_view, height, width, channel)).type_as(lf)
    for i, disparity in enumerate(disparity_list):
        disparity = round(disparity, 2)
        warped_LF[:, i, :, :, :, :, :] = view_warp_fast(lf, disparity)
    return warped_LF


def focus_img(lf, min, max, step):
    if len(lf.shape) == 5:
        lf = torch.from_numpy(lf[np.newaxis])
        warped_LF = view_warp_different_fast(lf, min, max, step)
        focused_img = warped_LF.mean(axis=2).mean(axis=2)
        focused_img = np.array(focused_img[0])
    else:
        warped_LF = view_warp_different_fast(lf, min, max, step)
        focused_img = warped_LF.mean(axis=2).mean(axis=2)
    return focused_img


def load_test(dir):
    img_dirs = listdir_nohidden(dir, type=False)
    if 'B.png' in img_dirs:
        img_dirs.remove('B.png')
        img_dirs.remove('R.png')
        img_dirs.sort()
        num = np.int(np.sqrt(len(img_dirs)))
        h, w, c = cv2.imread(dir + '/' + img_dirs[0], cv2.IMREAD_COLOR).shape
        img = np.zeros((num, num, h, w, c))
        for i in range(num):
            for j in range(num):
                index = i * num + j
                img[i, j, :, :, :] = cv2.imread(dir + '/' + img_dirs[index], cv2.IMREAD_COLOR) / 255.0
                img[i, j, :, :, :] = img[i, j, :, :, ::-1]
        B_img = cv2.imread(dir + '/B.png', cv2.IMREAD_COLOR) / 255.0
        B_img = B_img[:, :, ::-1]
        R_img = cv2.imread(dir + '/R.png', cv2.IMREAD_COLOR) / 255.0
        R_img = R_img[:, :, ::-1]
        return np.float32(img), np.float32(B_img), np.float32(R_img)
    else:
        img_dirs.sort()
        num = np.int(np.sqrt(len(img_dirs)))
        h, w, c = cv2.imread(dir + '/' + img_dirs[0], cv2.IMREAD_COLOR).shape
        img = np.zeros((num, num, h, w, c))
        for i in range(num):
            for j in range(num):
                index = i * num + j
                img[i, j, :, :, :] = cv2.imread(dir + '/' + img_dirs[index], cv2.IMREAD_COLOR) / 255.0
                img[i, j, :, :, :] = img[i, j, :, :, ::-1]
        return np.float32(img)


def load_test_full(dir):
    train_data_list = []
    B_gt_data_list = []
    R_gt_data_list = []
    image_path_list = []
    files = os.listdir(dir)
    # files.remove('opt.txt')
    files.sort()
    for image_path in files:
        train_data, B_gt_data, R_gt_data = load_test(dir + image_path)
        train_data_list.append(train_data)
        B_gt_data_list.append(B_gt_data)
        R_gt_data_list.append(R_gt_data)
        image_path_list.append(image_path)

    return [train_data_list, B_gt_data_list, R_gt_data_list, image_path_list]


def get_parameter_number(net):
    print(net)
    parameter_list = [p.numel() for p in net.parameters()]
    print(parameter_list)
    total_num = sum(parameter_list)
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

