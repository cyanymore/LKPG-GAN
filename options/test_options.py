from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=1500, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser

# python -m visdom.server
# python train.py --dataroot E:\colorization\kaist_wash_day --name kaist --model sc --gpu_ids 0 --lambda_spatial 10 --lambda_gradient 0 --attn_layers 4,7,9 --loss_mode cos --gan_mode lsgan --display_port 8097 --direction AtoB --patch_size 64
# python test.py --dataroot E:\colorization_paper\Manuscript\Ablation_study\Modification\kaist_night --checkpoints_dir ./checkpoints --name K_night --model sc --num_test 1500
