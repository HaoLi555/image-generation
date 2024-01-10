import jittor as jt
from jittor import nn
from jittor.dataset.cifar import CIFAR10
import numpy as np
from tqdm import tqdm
import wandb
from PIL import Image
import os


class GeneratorBlock(nn.Module):
    """生成器的残差块，可以进行channel变化以及上采样

    """

    def __init__(self, in_channel, out_channel, upsample=False):
        super(GeneratorBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample

        # 用于残差连接
        if in_channel != out_channel:
            self.shortcut_conv = nn.Conv(
                in_channel, out_channel, kernel_size=1)
        else:
            self.shortcut_conv = None

        self.bn1 = nn.BatchNorm(in_channel)
        self.conv1 = nn.Conv(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(out_channel)
        self.conv2 = nn.Conv(out_channel, out_channel,
                             kernel_size=3, padding=1)

    def execute(self, x):
        if self.upsample:
            shortcut = nn.upsample(
                x, size=(2*x.shape[2], 2*x.shape[3]), mode='nearest')
        else:
            shortcut = x

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)

        x = self.bn1(x)
        x = nn.relu(x)
        if self.upsample:
            x = nn.upsample(
                x, size=(2*x.shape[2], 2*x.shape[3]), mode='nearest')
        x = self.conv1(x)
        x = self.bn2(x)
        x = nn.relu(x)
        x = self.conv2(x)

        return x+shortcut


class Generator(nn.Module):
    """由线性层、残差块、卷积层组成

    """

    def __init__(self, in_dim=128, in_channel=128, out_channel=3):
        super(Generator, self).__init__()
        self.in_dim = in_dim
        self.out_channel = out_channel

        self.linear = nn.Linear(in_dim, 4*4*in_channel)

        # 4*4
        self.block1 = GeneratorBlock(in_channel, in_channel, upsample=True)
        # 8*8
        self.block2 = GeneratorBlock(in_channel, in_channel, upsample=True)
        # 16*16
        self.block3 = GeneratorBlock(in_channel, in_channel, upsample=True)
        # 32*32

        self.bn = nn.BatchNorm(in_channel)
        self.conv = nn.Conv(in_channel, out_channel, kernel_size=3, padding=1)

        self.tanh = nn.Tanh()

        # 参数初始化
        relu_gain = nn.init.calculate_gain('relu')
        for module in self.modules():
            if isinstance(module, (nn.Conv, nn.Linear)):
                gain = relu_gain if module != self.linear else 1.0
                nn.init.xavier_uniform_(module.weight, gain=gain)
                nn.init.zero_(module.bias)

    def execute(self, x):
        x = self.linear(x)
        x = x.reshape(-1, 128, 4, 4)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.bn(x)
        x = nn.relu(x)
        x = self.conv(x)

        x = self.tanh(x)

        return x


class DiscriminatorBlock(nn.Module):
    """辨别器的残差块，可以进行channel变化以及下采样

    """

    def __init__(self, in_channel, out_channel, downsample=False, first=False):
        super(DiscriminatorBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.downsample = downsample
        # 是否是第一层——决定是否relu
        self.first = first

        if in_channel != out_channel:
            self.shortcut_conv = nn.Conv(
                in_channel, out_channel, kernel_size=1)
        else:
            self.shortcut_conv = None

        self.conv1 = nn.Conv(in_channel, out_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv(out_channel, out_channel,
                             kernel_size=3, padding=1)

    def execute(self, x):
        if self.downsample:
            shortcut = nn.pool(x, 2, 'mean')
        else:
            shortcut = x

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)

        if self.first:
            x = nn.relu(x)
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        if self.downsample:
            x = nn.pool(x, 2, 'mean')

        return x+shortcut


class Discriminator(nn.Module):
    """由残差块、线性层组成

    """

    def __init__(self, in_channel=3, channel=128):
        super(Discriminator, self).__init__()
        self.in_channel = in_channel
        self.channel = channel

        # 32*32
        self.block1 = DiscriminatorBlock(
            in_channel, channel, downsample=True, first=True)
        # 16*16
        self.block2 = DiscriminatorBlock(channel, channel, downsample=True)
        # 8*8
        self.block3 = DiscriminatorBlock(channel, channel, downsample=False)
        self.block4 = DiscriminatorBlock(channel, channel, downsample=False)

        self.linear = nn.Linear(channel, 1)

        # 参数初始化
        relu_gain = nn.init.calculate_gain('relu')
        for module in self.modules():
            if isinstance(module, (nn.Conv, nn.Linear)):
                gain = relu_gain if module != self.linear else 1.0
                nn.init.xavier_uniform_(module.weight, gain=gain)
                nn.init.zero_(module.bias)

    def execute(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = nn.relu(x)
        x = jt.mean(x, dims=(2, 3))
        x = self.linear(x)

        return x


def circle(iterable):
    while True:
        for x in iterable:
            yield x


class WGAN_Manager():
    def __init__(self, train_loader, test_loader, wandb_run, num_steps=1, critic_steps=5, k=2, p=6, show_real_image=False) -> None:
        self.netG = Generator()
        self.netD = Discriminator()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_steps = num_steps
        # 生成器每迭代一次，判别器迭代的次数
        self.critic_steps = critic_steps
        self.k = k
        self.p = p
        self.show_real_image = show_real_image

        # 优化器定义
        self.optimizerG = nn.Adam(
            self.netG.parameters(), lr=2e-4, betas=(0, 0.9))
        self.optimizerD = nn.Adam(
            self.netD.parameters(), lr=2e-4, betas=(0, 0.9))

        self.wandb_run = wandb_run

    def train(self):

        self.netD.train()
        self.netG.train()

        train_gen = iter(circle(self.train_loader))

        for i in tqdm(range(self.num_steps), desc='Training'):

            loss_D_list = []
            # 训练辨别器
            for _ in range(0, self.critic_steps):
                real, label = next(train_gen)
                # 注意这里必须全都是32类型，保持类型一致
                real = jt.float32(real)
                real = real*2/255-1
                real = jt.transpose(real, (0, 3, 1, 2))
                noise = jt.float32(np.random.randn(real.shape[0], 128))

                # 获得生成图片
                fake = self.netG(noise)

                # 辨别器给出的得分
                real_score = jt.mean(self.netD(real))
                fake_score = jt.mean(self.netD(fake))

                # 计算梯度惩罚
                real_grad = jt.grad(real_score, real)
                fake_grad = jt.grad(fake_score, fake)

                # 计算梯度惩罚
                real_grad_norm = jt.sum(
                    jt.pow(real_grad, 2), dims=(1, 2, 3)).sqrt()
                fake_grad_norm = jt.sum(
                    jt.pow(fake_grad, 2), dims=(1, 2, 3)).sqrt()
                grad_loss = (self.k/2)*jt.mean(real_grad_norm **
                                               self.p + fake_grad_norm**self.p)

                # 总的损失——散度
                tot_loss = grad_loss+fake_score-real_score

                tot_loss.sync()
                loss_D_list.append(tot_loss.data[0])

                self.optimizerD.step(tot_loss)

            # 展示真实图片
            if self.show_real_image and i % 100 == 0:
                self.wandb_run.log({'real img': [wandb.Image(
                    (real[0].transpose(1, 2, 0).numpy()+1)*255/2)]}, step=i)

            # 训练生成器
            noise = jt.float32(np.random.randn(64, 128))
            gen = self.netG(noise)
            loss = -self.netD(gen).mean()
            loss.sync()
            self.optimizerG.step(loss)

            if i % 100 == 0:
                print(f"    step {i}:")
                print(f"        lossD: {np.mean(loss_D_list)}")
                print(f"        lossG: {loss.data[0]}")

                self.wandb_run.log({'lossD': np.mean(loss_D_list)}, step=i)
                self.wandb_run.log({'lossG': loss.data[0]}, step=i)
                # 随机展示生成图片
                self.wandb_run.log({'img': wandb.Image(np.concatenate([(gen[j].transpose(1, 2, 0).numpy(
                )+1)*255/2 for j in np.random.choice(64, 5, replace=False)],axis=1))}, step=i)

    def test(self):

        self.netG.eval()
        self.netD.eval()

        noise = jt.float32(np.random.randn(64, 128))
        imgs = self.netG(noise).data

        col=8
        row=8
        temp=[]
        for i in range(0,row):
            one_row=[]
            for j in range(0,col):
                index=i*col+j
                one_row.append(imgs[index])
            one_row=np.concatenate(one_row, axis=2)
            temp.append(one_row)
        temp=np.concatenate(temp, axis=1)

        temp=np.transpose(temp, [1,2,0])
        temp=(temp+1)*255/2
        temp=temp.astype(np.uint8)
        img=Image.fromarray(temp)

        if not os.path.exists('results'):
            os.makedirs('results')

        img.save('results/WGAN.png')



if __name__ == "__main__":
    if jt.has_cuda:
        jt.flags.use_cuda = 1

    wandb_run = wandb.init(
        project='image-gen',
    )

    train_loader = CIFAR10(train=True).set_attrs(shuffle=True, batch_size=64)
    test_loader = CIFAR10(train=False).set_attrs(shuffle=True, batch_size=64)

    manager = WGAN_Manager(train_loader, test_loader, wandb_run=wandb_run)
    manager.train()
    manager.test()
