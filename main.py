import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # Nattr_path='D:/Work/Dataset/Celeba/img_aling_Celeba/list_attr_celeba.txt',
    # batch_size=16, beta1=0.5, beta2=0.999, c2_dim=8, c_dim=5, celeba_crop_size=178,
    # celeba_image_dir='D:/Work/Dataset/Celeba/img_aling_Celeba/img_align_celeba/',
    # d_conv_dim=64, d_lr=0.0001, d_repeat_num=6, dataset='CelebA', g_conv_dim=64,
    # g_lr=0.0001, g_repeat_num=6, image_size=128, lambda_cls=1, lambda_gp=10, lambda_rec=10,
    # log_dir='stargan_celeba/logs', log_step=10, lr_update_step=1000, mode='train',
    # model_save_dir='stargan_celeba/models', model_save_step=10000, n_critic=5,
    # num_iters=200000, num_iters_decay=100000, num_workers=1, rafd_crop_size=256,
    # rafd_image_dir='data/RaFD/train', result_dir='stargan_celeba/results',
    # resume_iters=None, sample_dir='stargan_celeba/samples', sample_step=1000,
    # selected_attrs=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'],
    # test_iters=200000, use_tensorboard=True)

    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    rafd_loader = None

    # celeba_loader 相当于是 data_loader，而data_loader 是 torch.utils.data.dataloader.DataLoader的返回值
    # 其中 里面封装的dataset是CelebA 这个类的对象
    celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)

    # Solver for training and testing ModularGAN.
    solver = Solver(celeba_loader, rafd_loader, config)

    if config.mode == 'train':
        solver.train()

    elif config.mode == 'test':
        solver.test()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--e_conv_dim', type=int, default=64, help='number of conv filters in the first layer of E')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--attr_dims', type=list, nargs='+', default=[3, 1, 1], help='separate attributes into different modules')
    parser.add_argument('--e_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--t_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_cyc', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')

    # Training configuration.
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    parser.add_argument('--batch_size', type=int, default=4, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=400000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    # parser.add_argument('--num_iters', type=int, default=89217, help='number of total iterations for training D')
    # parser.add_argument('--num_iters_decay', type=int, default=45000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=350000, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=350000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Directories.
    # parser.add_argument('--celeba_image_dir', type=str, default='data/celeba/images')
    # parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
    parser.add_argument('--celeba_image_dir', type=str, default='D:/Work/Dataset/Celeba/img_aling_Celeba/img_align_celeba/')
    parser.add_argument('--attr_path', type=str, default='D:/Work/Dataset/Celeba/img_aling_Celeba/list_attr_celeba.txt')
    parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
    parser.add_argument('--log_dir', type=str, default='stargan_celeba/logs')
    parser.add_argument('--model_save_dir', type=str, default='stargan_celeba/models')
    parser.add_argument('--sample_dir', type=str, default='stargan_celeba/samples')
    parser.add_argument('--result_dir', type=str, default='stargan_celeba/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=5000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)