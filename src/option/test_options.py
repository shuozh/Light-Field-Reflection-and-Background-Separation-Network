from option.base_option import BaseOptions
from utils.utils import mkdirs
import os

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--model_name', type=str, default='DeRefLF')
        self.parser.add_argument('--channel', type=int, default=32)
        self.parser.add_argument('--height_view', type=int, default=7)
        self.parser.add_argument('--width_view', type=int, default=7)
        self.parser.add_argument('--sa_num', type=int, default=3)
        self.parser.add_argument('--sad_num', type=int, default=3)
        self.parser.add_argument('--if_bn', type=int, default=0)
        self.parser.add_argument('--min', type=float, default=-2)
        self.parser.add_argument('--max', type=float, default=2)
        self.parser.add_argument('--step', type=float, default=0.4)
        self.parser.add_argument('--dir_test_images', type=str,default='../test_example/')
        self.parser.add_argument('--dir_model', type=str, default='../NetworkSave/ourmodel/')
        self.parser.add_argument('--dir_result', type=str, default='result/')
        self.parser.add_argument('--gpu_ids', type=str, default='2', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        print(os.getcwd())
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