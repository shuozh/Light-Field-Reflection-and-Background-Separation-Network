from option.base_option import BaseOptions
from utils.utils import seed_torch, mkdirs
import os


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # dataload
        self.parser.add_argument('--train_dataload_name', type=str, default='DeRefLF_Train_Dataset')
        self.parser.add_argument('--test_dataload_name', type=str, default='DeRefLF_Test_Dataset')

        # data aug
        self.parser.add_argument('--up_size', type=int, default=16)
        self.parser.add_argument('--B_dir_train', type=str, default='../dataset2/Stanford_Lytro77/train/')
        self.parser.add_argument('--R_dir_train', type=str, default=None)
        self.parser.add_argument('--dir_test_images', type=str, default='../test_example/')
        self.parser.add_argument('--dir_model', type=str, default=None)
        self.parser.add_argument('--if_flip', type=int, default=1, choices=[0, 1])
        self.parser.add_argument('--if_rotation', type=int, default=1, choices=[0, 1])
        self.parser.add_argument('--disparity', type=float, default=1)
        self.parser.add_argument('--min', type=float, default=-2)
        self.parser.add_argument('--max', type=float, default=2)
        self.parser.add_argument('--step', type=float, default=0.4)

        # training
        self.parser.add_argument('--current_iter', type=int, default=0)
        self.parser.add_argument('--MAX_EPOCH', type=int, default=4000)
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--base_lr', type=float, default=2e-4)
        self.parser.add_argument('--crop_size', type=int, default=192)

        # loss
        self.parser.add_argument('--loss_weight', type=str, default='1,0.5,0.05,1,0.5,0.05,1,1')
        self.parser.add_argument('--gradient_loss', type=int, default=1, choices=[0, 1])
        self.parser.add_argument('--l1_loss', type=int, default=1, choices=[0, 1])
        self.parser.add_argument('--accumulation_steps', type=int, default=1)
        self.parser.add_argument('--clip_value', type=float, default=0.5)
        self.parser.add_argument('--weight_decay', type=float, default=0)
        self.parser.add_argument('--regular_type', type=int, default=2)
        self.parser.add_argument('--vgg_loss', type=int, default=1, choices=[0, 1])





        # model
        self.parser.add_argument('--model_name', type=str, default='DeRefLF')
        self.parser.add_argument('--channel', type=int, default=32)
        self.parser.add_argument('--height_view', type=int, default=7)
        self.parser.add_argument('--width_view', type=int, default=7)
        self.parser.add_argument('--sa_num', type=int, default=3)
        self.parser.add_argument('--sad_num', type=int, default=3)
        self.parser.add_argument('--if_bn', type=int, default=0)
        self.parser.add_argument('--disp_num', type=int, default=11)


        # result
        self.parser.add_argument('--NetworkSave', type=str, default='../NetworkSave')
        self.parser.add_argument('--dir_result', type=str, default='result_test/')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')


    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        seed_torch(seed=self.opt.seed)


        args = vars(self.opt)

        str_ids = self.opt.loss_weight.split(',')
        self.opt.loss_weight = []
        for str_id in str_ids:
            id = float(str_id)
            if id >= 0:
                self.opt.loss_weight.append(id)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        self.opt.name = self.opt.model_name
        expr_dir = os.path.join('../', self.opt.dir_result, self.opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt

