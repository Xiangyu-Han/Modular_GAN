from model_test import Encoder
from model_test import Transformer
from model_test import Reconstructor
from model_test import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.e_conv_dim = config.e_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.attr_dims = config.attr_dims
        self.num_transformer = len(self.attr_dims)
        self.e_repeat_num = config.e_repeat_num
        self.t_repeat_num = config.t_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls  # lambda_cls =1
        self.lambda_rec = config.lambda_rec  # 10
        self.lambda_gp = config.lambda_gp  # 10
        self.lambda_cyc = config.lambda_cyc

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    # 创建网络
    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.E = Encoder(self.e_conv_dim, self.e_repeat_num)
            self.T = torch.nn.ModuleList()
            for c_dim in self.attr_dims:
                self.T.append(Transformer(self.e_conv_dim*4, c_dim, self.t_repeat_num))

            self.R = Reconstructor(self.e_conv_dim*4)
            self.R.to(self.device)
            self.D = torch.nn.ModuleList()
            for c_dim in self.attr_dims:
                self.D.append(Discriminator(self.image_size, self.d_conv_dim, c_dim, self.d_repeat_num))

        # 优化器
        self.g_optimizer = torch.optim.Adam(list(self.E.parameters()) + list(self.T.parameters()) +
                                            list(self.R.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        # 打印网络结构
        self.print_network(self.E, 'E')
        self.print_network(self.T, 'T')
        self.print_network(self.R, 'R')
        self.print_network(self.D, 'D')

        # cuda加速运算
        self.E.to(self.device)
        self.T.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        # 网络参数量
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        if type(model) == torch.nn.modules.container.ModuleList:
            print(model[0])
        else:
            print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    # 保存训练好的模型
    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        E_path = os.path.join(self.model_save_dir, '{}-E.ckpt'.format(resume_iters))
        T_path = os.path.join(self.model_save_dir, '{}-T.ckpt'.format(resume_iters))
        R_path = os.path.join(self.model_save_dir, '{}-R.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.E.load_state_dict(torch.load(E_path, map_location=lambda storage, loc: storage))
        self.T.load_state_dict(torch.load(T_path, map_location=lambda storage, loc: storage))
        self.R.load_state_dict(torch.load(R_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    # 构建TensorBoard
    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)


    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    # 重置梯度
    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    # 反归一化
    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        #  clamp（）函数的功能将输入input张量每个元素的值压缩到区间 [0,1]，并返回结果到一个新张量。
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        # 这一部分来自WGAN的改善工作，主要是为了满足Lipschitz连续这个WGAN推导中需要的数学约束。
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)  # [16, 3, 128, 128] -> [16, 49152]
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def create_labels(self, batch_size):
        """
        generate target domain labels for debugging and testing
        """
        label_list = []
        for c_dim in self.attr_dims:
            if c_dim > 1:
                labels = []
                for i in range(c_dim):
                    label = torch.zeros([batch_size, c_dim]).to(self.device)
                    label[:, i] = 1
                    labels.append(label)
            else:
                labels = [torch.ones([batch_size, 1]).to(self.device), torch.zeros([batch_size, 1]).to(self.device)]
            label_list.append(labels)
        return label_list

    def create_labels_org(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []  # hair_color_indices ：[0, 1, 2]
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                # 如果目标标签需要改变的是头发颜色，就把想得到的颜色对应的索引置1，其余头发颜色置0
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                # 如果目标标签不是头发颜色，那么就取反，比如男性取反为女性，年老取反为年轻。
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def generate_labels(self, label_org):
        """
        generate target domain labels which are different from original
        """
        # reverse
        label_trg = 1 - label_org
        # finetune multi-label
        start = 0
        for i, c_dim in enumerate(self.attr_dims):
            if c_dim > 1:
                for j in range(label_trg.size(0)):
                    label = label_trg[j, start:start+c_dim]
                    # avoid empty label
                    if torch.sum(label) == 0:
                        label[0] = 1
                        label[:] = label[torch.randperm(c_dim)]
                    # only one positive left
                    elif torch.sum(label) > 1:
                        inds = torch.nonzero(label).view(-1)
                        inds = inds[torch.randperm(inds.size(0))]
                        for ind in inds[1:]:
                            label[ind] = 0
            start += c_dim
        # shuffle
        label_trg = label_trg[torch.randperm(label_trg.size(0))]
        return label_trg.detach()

    def label_slice(self, label, ind):
        """
        slice label for different transformers and discriminators
        """
        start = 0
        for i, c_dim in enumerate(self.attr_dims):
            if i >= ind:
                label_slice = label[:, start:start + c_dim]
                break
            else:
                start += c_dim
        return label_slice

    def save_sample(self, x, c_trg_list, save_path, ind):
        with torch.no_grad():
            x_list = [x]
            # E
            feat = self.E(x)
            # R
            x_rec = self.R(feat)
            x_list.append(x_rec)
            # T
            for j in range(self.num_transformer):
                for c_trg in c_trg_list[j]:
                    x_fake = self.R(self.T[j](feat, c_trg))
                    x_list.append(x_fake)
            x_concat = torch.cat(x_list, dim=3)
            sample_path = os.path.join(save_path, '{}-images.jpg'.format(ind))
            save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)


    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, reduction='sum') / logit.size(0)


    def train(self):
        """Train ModularGAN within a single dataset."""
        # Set data loader.
        data_loader = self.celeba_loader

        # Fetch fixed inputs for debugging.
        # 取一个Batch的固定图片，方便后面来看训练效果
        data_iter = iter(data_loader)
        x_fixed, _ = next(data_iter)  # 得到一个batch的图片
        x_fixed = x_fixed.to(self.device)
        batch_size = self.batch_size
        c_trg_list = self.create_labels(batch_size)
        # c_fixed_list = self.create_labels_org(c_org, self.c_dim, self.dataset, self.selected_attrs)
        # x_fixed表示图像像素值  c_org表示真实标签值  tensor([[ 1.,  0.,  0.,  1.,  1.]])

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:  # 参数resume_iters 设置为none
            start_iters = self.resume_iters  # 可以不连续训练，从之前训练好后的结果处开始
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        print('sample images will be saved into {}...'.format(self.sample_dir))
        start_time = time.time()
        g_loss_dict = {'G/loss_src': 0, 'G/loss_cyc': 0}
        for i in range(start_iters, self.num_iters):
            d_loss_dict = {'D/loss_src': 0, 'D/loss_gp': 0}
            if i and i % self.n_critic == 0:
                g_loss_dict = {'G/loss_src': 0, 'G/loss_cyc': 0}

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, c_org_t = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, c_org_t = next(data_iter)

            c_trg_t = self.generate_labels(c_org_t)

            c_org_l = c_org_t.clone()
            c_trg_l = c_trg_t.clone()

            x_real = x_real.to(self.device)         # Input images.
            c_org_t = c_org_t.to(self.device)         # Original domain labels.  c_org_t
            c_trg_t = c_trg_t.to(self.device)         # Target domain labels.   c_trg_t
            c_org_l = c_org_l.to(self.device)     # Labels for computing classification loss.  c_org_l
            c_trg_l = c_trg_l.to(self.device)     # Labels for computing classification loss.  c_trg_l

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            self.reset_grad()
            d_loss = 0

            # Compute loss with real images.
            for j in range(self.num_transformer):
                c_trg_t_j = self.label_slice(c_trg_t, j)
                c_org_l_j = self.label_slice(c_org_l, j)

                out_src, out_cls = self.D[j](x_real)
                # mean()函数的参数：dim=0,按列求平均值，返回的形状是（1，列数）；dim=1,按行求平均值，返回的形状是（行数，1）
                # 默认不设置dim的时候，返回的是所有元素的平均值。
                d_loss_real = - torch.mean(out_src)  # d_loss_real最小，那么 out_src 最大==1 （针对图像）
                # 计算交叉熵loss
                d_loss_cls = self.classification_loss(out_cls, c_org_l_j, self.dataset)

                # Compute loss with fake images.
                # 将真实图像输入x_real和假的标签c_trg输入生成网络,得到生成图像x_fake
                x_fake = self.R(self.T[j](self.E(x_real), c_trg_t_j))
                # detach()返回一个tensor变量，且这个变量永远不会有梯度值。这个变量跟原图上的变量共享一块内存，也就说是同一个家伙。
                out_src, _ = self.D[j](x_fake.detach())
                d_loss_fake = torch.mean(out_src)  # 判定出是假图的概率越大，损失越小

                # Compute loss for gradient penalty.
                # 计算梯度惩罚因子alpha,根据alpha结合x_real,x_fake,输入判别网络,计算梯度,得到梯度损失函数
                # alpha是一个随机数 tensor([[[[ 0.5692]]]])
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                # x_hat是一个图像大小的张量数据，随着alpha的改变而变化
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D[j](x_hat)
                # 最终d_loss_gp 在0.9954～ 0.9956 波动
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                # 损失包含4项：
                # 1.真实图像判定为真
                # 2.真实图像+错误标签记过G网络生成的图像判定为假
                # 3.真实图像经过D网络的生成的标签与真实标签之间的差异损失
                # 4.真实图像和 真实图像+错误标签记过G网络生成的图像 融合的梯度惩罚因子
                d_loss_j = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp

                # Logging.
                d_loss_dict['D/loss_src'] += d_loss_real.item() + d_loss_fake.item()
                d_loss_dict['D/loss_gp'] += d_loss_gp.item()
                d_loss_dict['D/loss_cls{}'.format(j)] = d_loss_cls.item()
                d_loss += d_loss_j

            d_loss.backward()
            # 一次训练更新一次参数空间
            self.d_optimizer.step()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            # 生成网络的作用是,输入original域的图可以生成目标域的图像,输入为目标域的图像,生成original域的图像（重建）
            # 每更新5次判别器再更新一次生成器
            if (i+1) % self.n_critic == 0:

                self.reset_grad()
                # Original-to-target domain.

                # compute l1 loss with cyclic reconstruction for encoder and decoder
                x_rec = self.R(self.E(x_real))
                g_loss_cyc = torch.mean(torch.abs(x_real - x_rec))

                # compute generation loss and backward
                g_loss = self.lambda_cyc * g_loss_cyc
                g_loss.backward()

                g_loss_dict['G/loss_cyc'] += g_loss_cyc.item()

                for j in range(self.num_transformer):
                    c_org_t_j = self.label_slice(c_org_t, j)
                    c_trg_t_j = self.label_slice(c_trg_t, j)
                    c_trg_l_j = self.label_slice(c_trg_l, j)

                    # generate fake images
                    f_trs = self.T[j](self.E(x_real), c_trg_t_j)
                    x_fake = self.R(f_trs)

                    # compute classification loss with fake images
                    out_src, out_cls = self.D[j](x_fake)
                    g_loss_fake = -torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls, c_trg_l_j, self.dataset)

                    # compute l1 loss with cyclic reconstruction for encoded features
                    f_rec = self.E(x_fake)
                    g_loss_cyc = torch.mean(torch.abs(f_trs - f_rec))

                    # compute generation loss and backward
                    g_loss_j = g_loss_fake + self.lambda_cyc * g_loss_cyc + self.lambda_cls * g_loss_cls
                    g_loss_j.backward()
                    g_loss += g_loss_j

                    # logging
                    g_loss_dict['G/loss_src'] += g_loss_fake.item()
                    g_loss_dict['G/loss_cyc'] += g_loss_cyc.item()
                    g_loss_dict['G/loss_cls{}'.format(j)] = g_loss_cls.item()

                self.g_optimizer.step()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in d_loss_dict.items():
                    log += ", {}: {:.4f}".format(tag, value)
                for tag, value in g_loss_dict.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

            # 存效果图
            # Translate fixed images for debugging.
            if i < 1000 and i % 100 == 0:
                self.save_sample(x_fixed, c_trg_list, self.sample_dir, i)
            if i and i % self.sample_step == 0:
                self.save_sample(x_fixed, c_trg_list, self.sample_dir, i)

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                E_path = os.path.join(self.model_save_dir, '{}-E.ckpt'.format(i+1))
                T_path = os.path.join(self.model_save_dir, '{}-T.ckpt'.format(i+1))
                R_path = os.path.join(self.model_save_dir, '{}-R.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.E.state_dict(), E_path)
                torch.save(self.T.state_dict(), T_path)
                torch.save(self.R.state_dict(), R_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))


            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Translate images using ModularGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        data_loader = self.celeba_loader

        c_trg_list = self.create_labels(self.batch_size)

        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                self.save_sample(x_real, c_trg_list, self.result_dir, i)

                print('Saved real and fake images into {}...'.format(self.result_dir))
