import jittor as jt
from jittor import nn
from jittor.dataset.cifar import CIFAR10
import numpy as np
from tqdm import tqdm
import wandb
from PIL import Image
from jittor import init


class GeneratorBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upsample=False):
        super(GeneratorBlock, self).__init__()
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.upsample=upsample

        if in_channel!=out_channel:
            self.shortcut_conv=nn.Conv(in_channel, out_channel, kernel_size=1)
        else:
            self.shortcut_conv=None

        self.bn1=nn.BatchNorm(in_channel)
        self.conv1=nn.Conv(in_channel, in_channel, kernel_size=3, padding=1)
        self.bn2=nn.BatchNorm(in_channel)
        self.conv2=nn.Conv(in_channel, out_channel, kernel_size=3, padding=1)

    def execute(self, x):
        if self.upsample:
            shortcut=nn.upsample(x,size=(2*x.shape[2],2*x.shape[3]), mode='nearest')
        else:
            shortcut=x

        if self.shortcut_conv is not None:
            shortcut=self.shortcut_conv(shortcut)

        x=self.bn1(x)
        x=nn.relu(x)
        if self.upsample:
            x=nn.upsample(x, size=(2*x.shape[2],2*x.shape[3]), mode='nearest')
        x=self.conv1(x)
        x=self.bn2(x)
        x=nn.relu(x)
        x=self.conv2(x)

        return x+shortcut


class Generator(nn.Module):
    def __init__(self, in_dim=128,in_channel=128, out_channel=3):
        super(Generator, self).__init__()
        self.in_dim=in_dim
        self.out_channel=out_channel

        # self.decoder=nn.Sequential(
        #     nn.ConvTranspose2d(in_channel,hidden_dim*4,4,1,0),
        #     nn.BatchNorm2d(hidden_dim*4),
        #     nn.LeakyReLU(),
       
        #     nn.ConvTranspose2d(hidden_dim*4,hidden_dim*2,4,2,1),
        #     nn.BatchNorm2d(hidden_dim*2),
        #     nn.LeakyReLU(),
        
        #     nn.ConvTranspose2d(hidden_dim*2,hidden_dim,4,2,1),
        #     nn.BatchNorm2d(hidden_dim),
        #     nn.LeakyReLU(),

        #     nn.ConvTranspose2d(hidden_dim,3,4,2,1),
        #     nn.Tanh()
        # )

        self.linear=nn.Linear(in_dim, 4*4*in_channel)

        self.block1=GeneratorBlock(in_channel, in_channel, upsample=True)
        self.block2=GeneratorBlock(in_channel, in_channel, upsample=True)
        self.block3=GeneratorBlock(in_channel, in_channel, upsample=True)

        self.bn=nn.BatchNorm(in_channel)
        self.conv=nn.Conv(in_channel, out_channel, kernel_size=3, padding=1)

        self.tanh=nn.Tanh()

    def execute(self, x):
        x=self.linear(x)
        x=x.reshape(-1,128,4,4)

        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)

        x=self.bn(x)
        x=nn.relu(x)
        x=self.conv(x)
        
        x=self.tanh(x)

        return x

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=False):
        super(DiscriminatorBlock, self).__init__()
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.downsample=downsample

        if in_channel!=out_channel:
            self.shortcut_conv=nn.Conv(in_channel, out_channel, kernel_size=1)
        else:
            self.shortcut_conv=None

        self.bn1=nn.BatchNorm(in_channel)
        self.conv1=nn.Conv(in_channel, in_channel, kernel_size=3, padding=1)
        self.bn2=nn.BatchNorm(in_channel)
        self.conv2=nn.Conv(in_channel, out_channel, kernel_size=3, padding=1)

    def execute(self, x):
        if self.downsample:
            shortcut=nn.pool(x, 2, 'mean')
        else:
            shortcut=x

        if self.shortcut_conv is not None:
            shortcut=self.shortcut_conv(shortcut)

        x=self.bn1(x)
        x=nn.relu(x)
        if self.downsample:
            x=nn.pool(x, 2, 'mean')
        x=self.conv1(x)
        x=self.bn2(x)
        x=nn.relu(x)
        x=self.conv2(x)

        return x+shortcut


class Discriminator(nn.Module):
    def __init__(self, in_channel=3, channel=128):
        super(Discriminator, self).__init__()
        self.in_channel=in_channel
        self.channel=channel

        # self.encoder=nn.Sequential(
        #     nn.Conv2d(in_channel,hidden_dim,4,2,1),
        #     nn.LeakyReLU(),

        #     nn.Conv2d(hidden_dim,hidden_dim*2,4,2,1),
        #     nn.BatchNorm2d(hidden_dim*2),
        #     nn.LeakyReLU(),

        #     nn.Conv2d(hidden_dim*2,hidden_dim*4,4,2,1),
        #     nn.BatchNorm2d(hidden_dim*4),
        #     nn.LeakyReLU(),

        #     nn.Conv2d(hidden_dim*4,1,4,1,0),

        # )
        self.block1=DiscriminatorBlock(in_channel, channel, downsample=True)
        self.block2=DiscriminatorBlock(channel, channel, downsample=True)
        self.block3=DiscriminatorBlock(channel, channel, downsample=False)
        self.block4=DiscriminatorBlock(channel, channel, downsample=False)

        self.linear=nn.Linear(channel, 1)



    def execute(self, x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)

        x=nn.relu(x)
        x=x.mean(-1, keepdims=False).mean(-1, keepdims=False)
        x=x.reshape((-1, self.channel))
        x=self.linear(x)

        return x


def init_weights(m):
    if type(m)==nn.Conv2d:
        init.trunc_normal_(m.weight, std=0.02)
    elif type(m)==nn.ConvTranspose2d:
        init.trunc_normal_(m.weight, std=0.02)
    elif type(m)==nn.BatchNorm2d:
        init.trunc_normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0)


def circle(iterable):
    while True:
        for x in iterable:
            yield x

class WGAN_Manager():
    def __init__(self, train_loader, test_loader, wandb_run, num_steps=5000, critic_steps=5) -> None:
        self.netG=Generator()
        self.netD=Discriminator()
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.num_steps=num_steps
        # 生成器每迭代一次，判别器迭代的次数
        self.critic_steps=critic_steps
        self.optimizerG=nn.RMSprop(self.netG.parameters(), lr=0.0002)
        self.optimizerD=nn.RMSprop(self.netD.parameters(), lr=0.0002)

        self.wandb_run=wandb_run

        self.netD.apply(init_weights)
        self.netG.apply(init_weights)

    def train(self):    

        self.netD.train()
        self.netG.train()

        train_gen=iter(circle(self.train_loader))

        for i in tqdm(range(self.num_steps), desc='Training'):

            loss_D_list=[]
            # 训练辨别器
            for _ in range(0, self.critic_steps):
                real , label=next(train_gen)
                # 注意这里必须全都是32类型
                real=jt.transpose(real, (0, 3, 1, 2)).float32()
                noise=jt.float32(np.random.randn(real.shape[0],128))

                loss_D_real=-self.netD(real).mean()
                loss_D_real.sync()
                self.optimizerD.step(loss_D_real)

                loss_D_fake=self.netD(self.netG(noise)).mean()
                loss_D_fake.sync()
                self.optimizerG.step(loss_D_fake)

                loss_D=loss_D_real.data+loss_D_fake.data
                loss_D_list.append(loss_D)

                for param in  self.netD.parameters():
                    param.safe_clip(-0.01,0.01)

            if i%100==0: 
                self.wandb_run.log({'real img':[wandb.Image(real[0].transpose(1,2,0).numpy())]}, step=i)
            
            
            # 训练生成器
            noise=jt.float32(np.random.randn(64,128))
            gen=self.netG(noise)
            loss=-self.netD(gen).mean()
            loss.sync()
            self.optimizerG.step(loss)

            if i%100==0: 
                print(f"    step {i}:")
                print(f"        lossD: {np.mean(loss_D_list)}")
                print(f"        lossG: {loss.data[0]}")

                self.wandb_run.log({'lossD':np.mean(loss_D_list)}, step=i)
                self.wandb_run.log({'lossG':loss.data[0]}, step=i)
                self.wandb_run.log({'img':[wandb.Image(gen[j].transpose(1,2,0).numpy()) for j in np.random.choice(64,5, replace=False)]}, step=i)


    def test(self):
        
        self.netG.eval()
        self.netD.eval()


        noise=jt.float32(np.random.randn(16,128))
        imgs=self.netG(noise).data
        for index, img in enumerate(imgs):
            img=img.transpose(1,2,0)
            img=img.astype(np.uint8)
            img=Image.fromarray(img)
            img.save(f'results/{index}.png')


if __name__ == "__main__":
    if jt.has_cuda:
        jt.flags.use_cuda=1


    wandb_run=wandb.init(
        project='image-gen',
    )


    train_loader=CIFAR10(train=True).set_attrs(shuffle=True, batch_size=64)
    test_loader=CIFAR10(train=False).set_attrs(shuffle=True, batch_size=64)


    manager=WGAN_Manager(train_loader, test_loader,wandb_run=wandb_run)
    manager.train()