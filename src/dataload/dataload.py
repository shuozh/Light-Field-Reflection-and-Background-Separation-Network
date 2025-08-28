import numpy as np
import torch.utils.data as data
from utils import utils
import torch

class DeRefLF_Train_Dataset(data.Dataset):
    def __init__(self, opt):
        self.up_size = opt.up_size
        self.crop_size = opt.crop_size
        self.if_flip = opt.if_flip
        self.if_rotation = opt.if_rotation
        self.B_dir_train = opt.B_dir_train
        self.R_dir_train = opt.R_dir_train

        self.height_view = opt.height_view
        self.min = opt.min
        self.max = opt.max
        self.step = opt.step

        self.disparity = opt.disparity

        if self.R_dir_train == None:
            self.B_gt_data = utils.load_B_R_gt(self.B_dir_train, self.R_dir_train)
        else:
            self.B_gt_data, self.R_gt_data = utils.load_B_R_gt(self.B_dir_train, self.R_dir_train)

        self.num = len(self.B_gt_data)

    def __len__(self):
        return self.num * self.up_size

    def __getitem__(self, idx):
        B_idxtmp = idx % self.num
        while (1):
            if self.R_dir_train == None:
                index = np.random.randint(0, self.num)
                while index == B_idxtmp:
                    index = np.random.randint(0, self.num)
                flag = np.random.rand()
                if flag<0.5:
                    B_gt_data = self.B_gt_data[B_idxtmp]
                    R_gt_data = self.B_gt_data[index]
                else:
                    R_gt_data = self.B_gt_data[B_idxtmp]
                    B_gt_data= self.B_gt_data[index]
            else:
                index = np.random.randint(0, self.num)
                B_gt_data = self.B_gt_data[B_idxtmp]
                R_gt_data = self.R_gt_data[index]

            if np.mean(B_gt_data) * 1 / 2 > np.mean(R_gt_data):
                continue
            if np.mean(R_gt_data) * 1 / 2 > np.mean(B_gt_data):
                continue
            train_data, B_gt_data_c, R_gt_data_c, factor = utils.create_B_R(B_gt_data, R_gt_data,
                                                                            crop_size=self.crop_size + 2 * self.max * (
                                                                                    self.height_view - 1),
                                                                            disparity=self.disparity)
            if B_gt_data_c.max() < 0.15 or R_gt_data_c.max() < 0.15 or train_data.max() < 0.1:
                continue
            break

        focus_data = utils.create_F(train_data, self.min, self.max, self.step)

        top = int(self.max * (self.height_view - 1))
        left = top
        focus_data, _, _ = utils.random_crop(focus_data, self.crop_size, self.crop_size, top, left)
        train_data, _, _ = utils.random_crop(train_data, self.crop_size, self.crop_size, top, left)
        B_gt_data_c, _, _ = utils.random_crop(B_gt_data_c, self.crop_size, self.crop_size, top, left)
        R_gt_data_c, _, _ = utils.random_crop(R_gt_data_c, self.crop_size, self.crop_size, top, left)

        if self.if_flip:
            train_data, random_tmp = utils.random_flip(train_data)
            B_gt_data_c, _ = utils.random_flip(B_gt_data_c, random_tmp)
            R_gt_data_c, _ = utils.random_flip(R_gt_data_c, random_tmp)
            focus_data, _ = utils.random_flip(focus_data, random_tmp)
        if self.if_rotation:
            train_data, random_tmp = utils.random_rotation(train_data)
            B_gt_data_c, _ = utils.random_rotation(B_gt_data_c, random_tmp)
            R_gt_data_c, _ = utils.random_rotation(R_gt_data_c, random_tmp)
            focus_data, _ = utils.random_rotation(focus_data, random_tmp)

        train_data = torch.from_numpy(train_data.copy())
        B_gt_data_c = torch.from_numpy(B_gt_data_c.copy())
        R_gt_data_c = torch.from_numpy(R_gt_data_c.copy())
        focus_data = torch.from_numpy(focus_data.copy())
        factor = torch.tensor(factor, dtype=torch.float32)

        return train_data, B_gt_data_c, R_gt_data_c,focus_data, factor



def DeRefLF_Test_Dataset(opt):
    test_dataset = utils.load_test_full(opt.dir_test_images)
    return test_dataset
