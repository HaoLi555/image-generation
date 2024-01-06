import jittor as jt
from jittor import nn
from jittor.dataset.cifar import CIFAR10
import numpy as np
from tqdm import tqdm
import wandb
from PIL import Image

class Generator(nn.Module):
    def __init__(self, in_channel=16, hidden_dim=16):
        super(Generator, self).__init__()
        self.in_channel=in_channel
        self.hidden_dim=hidden_dim

        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(in_channel,hidden_dim*4,4,1,0),
            nn.BatchNorm2d(hidden_dim*4),
            nn.LeakyReLU(),
       
            nn.ConvTranspose2d(hidden_dim*4,hidden_dim*2,4,2,1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.LeakyReLU(),
        
            nn.ConvTranspose2d(hidden_dim*2,hidden_dim,4,2,1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(hidden_dim,3,4,2,1),
            nn.Tanh()
        )

    def execute(self, x):
        x=self.decoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channel=3, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.in_channel=in_channel
        self.hidden_dim=hidden_dim

        self.encoder=nn.Sequential(
            nn.Conv2d(in_channel,hidden_dim,4,2,1),
            nn.LeakyReLU(),

            nn.Conv2d(hidden_dim,hidden_dim*2,4,2,1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.LeakyReLU(),

            nn.Conv2d(hidden_dim*2,hidden_dim*4,4,2,1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.LeakyReLU(),

            nn.Conv2d(hidden_dim*4,1,4,1,0),

        )

    def execute(self, x):
        x=self.encoder(x)
        return x
    
# def init_weights(m):
#     if type(m)==nn.Conv2d:





def circle(iterable):
    while True:
        for x in iterable:
            yield x

class WGAN_Manager():
    def __init__(self, train_loader, test_loader, wandb_run, num_steps=10000, critic_steps=5) -> None:
        self.netG=Generator()
        self.netD=Discriminator()
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.num_steps=num_steps
        # 生成器每迭代一次，判别器迭代的次数
        self.critic_steps=critic_steps
        self.optimizerG=nn.RMSprop(self.netG.parameters(), lr=0.00005)
        self.optimizerD=nn.RMSprop(self.netD.parameters(), lr=0.00005)

        self.wandb_run=wandb_run

    def train(self):    

        self.netD.train()
        self.netG.train()

        train_gen=iter(circle(self.train_loader))

        for i in tqdm(range(self.num_steps), desc='Training'):

            loss_D=[]
            # 训练辨别器
            for _ in range(0, self.critic_steps):
                real , label=next(train_gen)
                # 注意这里必须全都是32类型
                real=jt.transpose(real, (0, 3, 1, 2)).float32()
                noise=jt.float32(np.random.randn(real.shape[0],16,1,1))

                critic_real_mean=self.netD(real).mean()
                critic_gen_mean=self.netD(self.netG(noise)).mean()

                loss=critic_gen_mean-critic_real_mean
                loss.sync()
                self.optimizerD.step(loss)
                loss_D.append(loss.data)

                for param in  self.netD.parameters():
                    param.safe_clip(-0.01,0.01)

            
            
            # 训练生成器
            noise=jt.float32(np.random.randn(64,16,1,1))
            gen=self.netG(noise)
            loss=-self.netD(gen).mean()
            loss.sync()
            self.optimizerG.step(loss)

            for param in  self.netG.parameters():
                param.safe_clip(-0.01,0.01)

            if i%100==0: 
                print(f"    step {i}:")
                print(f"        lossD: {np.mean(loss_D)}")
                print(f"        lossG: {loss.data[0]}")

                self.wandb_run.log({'lossD':np.mean(loss_D)}, step=i)
                self.wandb_run.log({'lossG':loss.data[0]}, step=i)
                self.wandb_run.log({'img':[wandb.Image(gen[j].transpose(1,2,0).numpy()) for j in np.random.choice(64,5, replace=False)]}, step=i)


    def test(self):
        
        self.netG.eval()
        self.netD.eval()


        noise=jt.float32(np.random.randn(16,16,1,1))
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