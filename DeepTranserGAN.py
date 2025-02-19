import dis
import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torcheval.metrics.functional import peak_signal_noise_ratio
from torchvision import transforms
from tqdm import tqdm

from datasets.data_set import LowLightDataset
from models.base_mode import Generator, Discriminator, Critic
from utils.loss import BCEBlurWithLogitsLoss
from utils.misic import set_random_seed, get_opt, get_loss, ssim, model_structure, save_path


class BaseTrainer:
    def __init__(self, args, generator, discriminator=None, critic=None):
        self.args = args
        self.generator = generator
        self.discriminator = discriminator
        self.critic = critic
        self.device = torch.device('cpu')
        if args.device == 'cuda':
            self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.generator = self.generator.to(self.device)
        if self.discriminator is not None:
            self.discriminator = self.discriminator.to(self.device)
        if self.critic is not None:
            self.critic = self.critic.to(self.device)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.img_size[0], args.img_size[1])),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.train_data = LowLightDataset(
            image_dir=args.data, transform=self.transform, phase="train")
        self.train_loader = DataLoader(self.train_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                       shuffle=True, drop_last=True)
        self.test_data = LowLightDataset(
            image_dir=args.data, transform=self.transform, phase="test")
        self.test_loader = DataLoader(self.test_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                      shuffle=False, drop_last=False)
        self.g_optimizer, self.d_optimizer = get_opt(
            args, self.generator, self.discriminator if self.discriminator else self.critic)
        self.g_loss = get_loss(args.loss).to(
            self.device) if args.loss else nn.MSELoss().to(self.device)
        self.stable_loss = nn.MSELoss().to(self.device)
        self.log = tensorboard.writer.SummaryWriter(log_dir=self.args.save_path, filename_suffix=str(args.epochs),
                                                    flush_secs=180)
        self.path = save_path(args.save_path)
        os.makedirs(os.path.join(self.path, 'generator'), exist_ok=True)
        if self.discriminator is not None:
            os.makedirs(os.path.join(
                self.path, 'discriminator'), exist_ok=True)
        if self.critic is not None:
            os.makedirs(os.path.join(self.path, 'critic'), exist_ok=True)
        self.train_log = self.path + '/log.txt'
        self.args_dict = args.__dict__
        self.epoch = 0
        self.Ssim = [0.]
        self.PSN = [0.]

    def load_checkpoint(self):
        if self.args.resume != '':
            g_path_checkpoint = os.path.join(
                self.args.resume, 'generator/last.pt')
            g_checkpoint = torch.load(g_path_checkpoint)
            self.generator.load_state_dict(g_checkpoint['net'])
            self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
            self.epoch = g_checkpoint['epoch']

            d_path_checkpoint = os.path.join(
                self.args.resume, 'discriminator/last.pt') if self.discriminator else os.path.join(self.args.resume, 'critic/last.pt')
            d_checkpoint = torch.load(d_path_checkpoint)
            if self.discriminator:
                self.discriminator.load_state_dict(d_checkpoint['net'])
            else:
                self.critic.load_state_dict(
                    d_checkpoint['net']) if self.critic else None
            self.d_optimizer.load_state_dict(d_checkpoint['optimizer'])
            print(f'继续第：{self.epoch + 1}轮训练')
            # 继续训练时，不需要再次设置路径
            self.path = self.args.resume
            self.args.resume = ''

    def save_checkpoint(self):
        g_checkpoint = {'net': self.generator.state_dict(
        ), 'optimizer': self.g_optimizer.state_dict(), 'epoch': self.epoch}
        d_checkpoint = {'net': self.discriminator.state_dict() if self.discriminator else self.critic.state_dict() if self.critic else None,
                        'optimizer': self.d_optimizer.state_dict(), 'epoch': self.epoch}
        torch.save(g_checkpoint, os.path.join(self.path, 'generator/last.pt'))
        torch.save(d_checkpoint, os.path.join(self.path, 'discriminator/last.pt')
                   if self.discriminator else os.path.join(self.path, 'critic/last.pt'))

    def log_message(self, message):
        print(message)

    def write_log(self, epoch, gen_loss, dis_loss, d_x, d_g_z1, d_g_z2):
        train_log_txt_formatter = (
            '{time_str} \t [Epoch] \t {epoch:03d} \t [gLoss] \t {gloss_str} \t [dLoss] \t {dloss_str} \t {Dx_str} \t ['
            'Dgz0] \t {Dgz0_str} \t [Dgz1] \t {Dgz1_str}\n')
        to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"), epoch=epoch + 1,
                                                  gloss_str=" ".join(
                                                      ["{:4f}".format(np.mean(gen_loss))]),
                                                  dloss_str=" ".join(
                                                      ["{:4f}".format(np.mean(dis_loss))]),
                                                  Dx_str=" ".join(
                                                      ["{:4f}".format(d_x)]),
                                                  Dgz0_str=" ".join(
                                                      ["{:4f}".format(d_g_z1)]),
                                                  Dgz1_str=" ".join(["{:4f}".format(d_g_z2)]))
        with open(self.train_log, "a") as f:
            f.write(to_write)

    def visualize_results(self, epoch, gen_loss, dis_loss, high_images, fake):
        self.log.add_scalar('generation loss', np.mean(gen_loss), epoch + 1)
        self.log.add_scalar('discrimination loss',
                            np.mean(dis_loss), epoch + 1)
        self.log.add_scalar('learning rate', self.g_optimizer.state_dict()[
                            'param_groups'][0]['lr'], epoch + 1)
        self.log.add_images('real', high_images, epoch + 1)
        self.log.add_images('fake', fake, epoch + 1)

    def evaluate_model(self):
        with torch.no_grad():
            self.log_message("Evaluating the generator model")
            self.generator.eval()
            if self.discriminator:
                self.discriminator.eval()
            else:
                self.critic.eval() if self.critic else None
            Ssim = [0.]
            PSN = [0.]
            for i, (low_images, high_images) in enumerate(self.test_loader):
                low_images = low_images.to(self.device)
                high_images = high_images.to(self.device)
                fake_eval = self.generator(low_images)
                ssim_source = ssim(fake_eval, high_images)
                psn = peak_signal_noise_ratio(fake_eval, high_images)
                if ssim_source.item() > max(Ssim):
                    g_checkpoint = {'net': self.generator.state_dict(
                    ), 'optimizer': self.g_optimizer.state_dict(), 'epoch': self.epoch}
                    d_checkpoint = {'net': self.discriminator.state_dict() if self.discriminator else self.critic.state_dict(),  # type: ignore
                                    'optimizer': self.d_optimizer.state_dict(), 'epoch': self.epoch}
                    torch.save(g_checkpoint, os.path.join(
                        self.path, 'generator/best.pt'))
                    torch.save(d_checkpoint, os.path.join(self.path, 'discriminator/best.pt')
                               if self.discriminator else os.path.join(self.path, 'critic/best.pt'))
                Ssim.append(ssim_source.item())
                PSN.append(psn.item())
            self.log_message("Model SSIM : {}          PSN: {}".format(
                np.mean(Ssim), np.mean(PSN)))

    def train_epoch(self):
        raise NotImplementedError

    def train(self):
        set_random_seed(
            self.args.seed, deterministic=self.args.deterministic, benchmark=self.args.benchmark)
        self.load_checkpoint()
        while self.epoch < self.args.epochs:
            self.train_epoch()
            if (self.epoch + 1) % 100 == 0 and (self.epoch + 1) >= 100:
                self.evaluate_model()
            self.epoch += 1
        self.log.close()


class StandardGANTrainer(BaseTrainer):
    def __init__(self, args, generator, discriminator):
        super().__init__(args, generator, discriminator=discriminator)
        self.d_loss = BCEBlurWithLogitsLoss().to(self.device)

    def train_epoch(self):
        self.discriminator.train()
        self.generator.train()
        source_g = [0.]
        d_g_z2 = 0.
        gen_loss = []
        dis_loss = []
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} {elapsed}', colour='#8762A5')
        for i, (low_images, high_images) in pbar:
            low_images = low_images.to(self.device)
            high_images = high_images.to(self.device)
            with autocast(enabled=self.args.amp):
                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()
                fake = self.generator(low_images)
                fake_inputs = self.discriminator(fake.detach())
                real_inputs = self.discriminator(high_images)

                real_lable = torch.ones_like(
                    fake_inputs.detach(), requires_grad=False)
                fake_lable = torch.zeros_like(
                    fake_inputs.detach(), requires_grad=False)
                d_real_output = self.d_loss(real_inputs, real_lable)
                d_x = real_inputs.mean().item()
                d_fake_output = self.d_loss(fake_inputs, fake_lable)
                d_g_z1 = fake_inputs.mean().item()
                d_output = (d_real_output.item() + d_fake_output.item()) / 2.

                fake_inputs = self.discriminator(fake.detach())
                g_output = self.g_loss(
                    fake_inputs, real_lable) + self.stable_loss(fake, high_images)

                d_real_output.backward()
                d_fake_output.backward()
                g_output.backward()
                d_g_z2 = fake_inputs.mean().item()
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), 500)
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5)
                self.d_optimizer.step()
                self.g_optimizer.step()

                gen_loss.append(g_output.item())
                dis_loss.append(d_output)

                source_g.append(d_g_z2)
                pbar.set_description('||Epoch: [%d/%d]|--|--|Batch: [%d/%d]|--|--|Loss_D: %.4f|--|--|Loss_G: '
                                     '%.4f|--|--|--|D(x): %.4f|--|--|D(G(z)): %.4f / %.4f|'
                                     % (self.epoch + 1, self.args.epochs, i + 1, len(self.train_loader),
                                        d_output, g_output.item(), d_x, d_g_z1, d_g_z2))
                self.save_checkpoint()
        self.write_log(self.epoch, gen_loss, dis_loss, d_x, d_g_z1, d_g_z2)
        high_images = high_images.to(self.device)
        fake = fake.to(self.device)
        self.visualize_results(self.epoch, gen_loss,
                               dis_loss, high_images, fake)


class WGAN_GPTrainer(BaseTrainer):
    def __init__(self, args, generator, critic):
        super().__init__(args, generator, critic=critic)
        self.lambda_gp = 10
        if self.critic is None:
            raise ValueError("Critic model is not initialized.")
        self.c_optimizer, self.g_optimizer = get_opt(
            args, self.generator, self.critic)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand((real_samples.size(0), 1, 1, 1),
                           device=real_samples.device)
        interpolates = (alpha * real_samples + ((1 - alpha)
                        * fake_samples)).requires_grad_(True)
        critic_interpolates = self.critic(interpolates)
        grad_outputs = torch.ones(critic_interpolates.size(
        ), device=real_samples.device, requires_grad=False)
        grad_interpolates = torch.autograd.grad(outputs=critic_interpolates, inputs=interpolates,
                                                grad_outputs=grad_outputs, create_graph=True, retain_graph=True,
                                                only_inputs=True)[0]
        grad_interpolates = grad_interpolates.view(real_samples.size(0), -1)
        grad_norm = grad_interpolates.norm(2, dim=1)
        grad_penalty = ((grad_norm - 1) ** 2).mean()
        return grad_penalty

    def train_epoch(self):
        self.critic.train()
        self.generator.train()
        source_g = [0.]
        g_z = 0.
        gen_loss = []
        critic_loss = []
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} {elapsed}', colour='#8762A5')
        for i, (low_images, high_images) in pbar:
            low_images = low_images.to(self.device)
            high_images = high_images.to(self.device)
            with autocast(enabled=self.args.amp):
                self.c_optimizer.zero_grad()
                self.g_optimizer.zero_grad()
                fake_images = self.generator(low_images)
                critic_real = self.critic(high_images)
                critic_fake = self.critic(fake_images.detach())
                gradient_penalty = self.compute_gradient_penalty(
                    high_images, fake_images.detach())
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)
                                ) + self.lambda_gp * gradient_penalty

                fake_images = self.generator(low_images)
                critic_fake = self.critic(fake_images)
                loss_generator = - \
                    torch.mean(critic_fake) + \
                    self.stable_loss(fake_images, high_images)

                loss_critic.backward()
                self.c_optimizer.step()
                loss_generator.backward()
                self.g_optimizer.step()

                gen_loss.append(loss_generator.item())
                critic_loss.append(loss_critic.item())

                g_z = critic_fake.mean().item()
                source_g.append(g_z)
                pbar.set_description('||Epoch: [%d/%d]|--|--|Batch: [%d/%d]|--|--|Loss_C: %.4f|--|--|Loss_G: '
                                     '%.4f|--|--|--|D(x): N/A|--|--|G(z): %.4f|'
                                     % (self.epoch + 1, self.args.epochs, i + 1, len(self.train_loader),
                                        loss_critic.item(), loss_generator.item(), g_z))
                self.save_checkpoint()
        train_log_txt_formatter = (
            '{time_str} \t [Epoch] \t {epoch:03d} \t [gLoss] \t {gloss_str} \t [cLoss] \t {closs_str} \t [Dx] \t {Dx_str} \t ['
            'Gz] \t {Gz_str}\n')
        to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"), epoch=self.epoch + 1,
                                                  gloss_str=" ".join(
                                                      ["{:4f}".format(np.mean(gen_loss))]),
                                                  closs_str=" ".join(
                                                      ["{:4f}".format(np.mean(critic_loss))]),
                                                  Dx_str="N/A", Gz_str=" ".join(["{:4f}".format(g_z)]))
        with open(self.train_log, "a") as f:
            f.write(to_write)
        high_images = high_images.to(self.device)
        fake_images = fake_images.to(self.device)
        self.log.add_scalar('generation loss',
                            np.mean(gen_loss), self.epoch + 1)
        self.log.add_scalar('critic loss', np.mean(
            critic_loss), self.epoch + 1)
        self.log.add_scalar('learning rate', self.g_optimizer.state_dict()[
                            'param_groups'][0]['lr'], self.epoch + 1)
        self.log.add_images('real', high_images, self.epoch + 1)
        self.log.add_images('fake', fake_images, self.epoch + 1)


def train(args):
    generator = Generator(args.depth, args.weight)
    discriminator = Discriminator(
        batch_size=args.batch_size, img_size=args.img_size[0])
    trainer = StandardGANTrainer(args, generator, discriminator)
    trainer.train()


def train_WGAN(args):
    generator = Generator(args.depth, args.weight)
    critic = Critic()
    trainer = WGAN_GPTrainer(args, generator, critic)
    trainer.train()


def predict(self):
    # 防止同名覆盖
    path = save_path(self.save_path, model='predict')
    # 数据准备
    if self.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    model = Generator()
    model_structure(model, (3, self.img_size[0], self.img_size[1]))
    checkpoint = torch.load(self.model)
    model.load_state_dict(checkpoint['net'])
    model.to(device)

    transform = transforms.Compose([
        transforms.ToPILImage(),  # 先转换为PIL Image, 因为一些transform需要PIL Image作为输入
        transforms.Resize((256, 256)),  # 可选：调整大小
        transforms.ToTensor(),          # 转换为Tensor
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 可选：归一化
    ])  # 图像转换
    test_data = LowLightDataset(
        image_dir=self.data, transform=transform, phase="test")
    img_pil = transforms.ToPILImage()
    test_loader = DataLoader(test_data,
                             batch_size=self.batch_size,
                             num_workers=self.num_workers,
                             drop_last=True)
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), bar_format='{l_bar}{bar:10}| {n_fmt}/{'
                                                                           'total_fmt} {elapsed}')
    model.eval()
    torch.no_grad()
    i = 0
    if not os.path.exists(os.path.join(path, 'predictions')):
        os.makedirs(os.path.join(path, 'predictions'))
    for i, (low_images, high_images) in pbar:

        lamb = 255.  # 取绝对值最大值，避免负数超出索引
        low_images = low_images.to(device) / lamb
        high_images = high_images.to(device) / lamb

        fake = model(low_images)
        for j in range(self.batch_size):

            fake_img = np.array(
                img_pil(fake[j]), dtype=np.float32)

            if i > 10 and i % 10 == 0:  # 图片太多，十轮保存一次
                img_save_path = os.path.join(
                    path, 'predictions', str(i) + '.jpg')
                cv2.imwrite(img_save_path, fake_img)
            i = i + 1
        pbar.set_description('Processed %d images' % i)
    pbar.close()


def predict_live(self):
    if self.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    model = Generator(1, 1)
    model_structure(model, (3, self.img_size[0], self.img_size[1]))
    checkpoint = torch.load(self.model)
    model.load_state_dict(checkpoint['net'])
    model.to(device)
    cap = cv2.VideoCapture('test_dark_countryard.mp4')  # 读取图像
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    write = cv2.VideoWriter()
    write.open(self.save_path + '/fake.mp4', fourcc=fourcc, fps=cap.get(cv2.CAP_PROP_FPS), isColor=True,
               frameSize=[640, 480])

    frame_buffer = []
    batch_size = 4  # 根据显存调整

    model.eval()
    torch.no_grad()
    if not os.path.exists(os.path.join(self.save_path, 'predictions')):
        os.makedirs(os.path.join(self.save_path, 'predictions'))
    while cap.isOpened():
        _, frame = cap.read()
        _, frame = cap.read()
        frame_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        '''# 是否需要resize取决于新图片格式与训练时的是否一致
        frame_pil = cv2.resize(frame_pil, self.img_size)
        # 是否需要resize取决于新图片格式与训练时的是否一致
        frame_pil = cv2.resize(frame_pil, self.img_size)'''

        frame_pil = torch.tensor(np.array(
            frame_pil, np.float32) / 255., dtype=torch.float32).to(device)  # 转为tensor
        frame_pil = torch.unsqueeze(frame_pil, 0).permute(
            0, 3, 1, 2)  # 提升维度--转换维度
        fake = model(frame_pil)
        fake = fake.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        # fake = cv2.resize(fake, (640, 480))  # 维度还没降下来
        cv2.imshow('fake', fake)

        cv2.imshow('origin', frame)

        # 写入文件
        # float转uint8 fake[0,1]转[0,255]
        write.write(fake)

        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
    cap.release()
    write.release()


def compute_gradient_penalty(critic, real_samples, fake_samples):
    """计算梯度惩罚"""
    alpha = torch.rand((real_samples.size(
        0), 1, 1, 1), device=real_samples.device)  # 形状是 (batch_size, 1, 1, 1)  适配图像形状
    interpolates = (alpha * real_samples + ((1 - alpha)
                    * fake_samples)).requires_grad_(True)

    critic_interpolates = critic(interpolates)

    grad_outputs = torch.ones(critic_interpolates.size(
    ), device=real_samples.device, requires_grad=False)

    grad_interpolates = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad_interpolates = grad_interpolates.view(
        real_samples.size(0), -1)  # 展平梯度
    grad_norm = grad_interpolates.norm(2, dim=1)  # 计算梯度的范数
    grad_penalty = ((grad_norm - 1) ** 2).mean()
    return grad_penalty
