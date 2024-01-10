import jittor as jt
from jittor import nn
import jittor.init as init
from tqdm import tqdm
import wandb
import numpy as np
import os 
from PIL import Image

def init_weights(m):
    if type(m)==nn.Conv2d:
        init.trunc_normal_(m.weight, std=0.02)
    elif type(m)==nn.ConvTranspose2d:
        init.trunc_normal_(m.weight, std=0.02)
    elif type(m)==nn.BatchNorm2d:
        init.trunc_normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0)

class Encoder(nn.Module):
    def __init__(self, channel, out_dim) -> None:
        super(Encoder, self).__init__()
        self.channel=channel
        self.out_dim=out_dim

        self.conv_block=nn.Sequential(
            nn.Conv(self.channel, 32, kernel_size=4, stride=2, padding=1), # 32, 16, 16
            nn.ReLU(),
            nn.BatchNorm(32),
            nn.Conv(32, 64, kernel_size=4, stride=2, padding=1), # 64, 8, 8
            nn.ReLU(),
            nn.BatchNorm(64),
            nn.Conv(64, 64, kernel_size=4, stride=2, padding=1), # 64, 4, 4
            nn.ReLU(),
            nn.BatchNorm(64),
            nn.Conv(64, 256, kernel_size=4, stride=1), # 256, 1, 1
            nn.ReLU(),
            nn.BatchNorm(256),
        )
        self.linear=nn.Linear(256, self.out_dim)

    def execute(self, x):
        x=self.conv_block(x)
        x=jt.reshape(x, (-1, 256*1*1))
        x=self.linear(x)
        return x
        
class Decoder(nn.Module):
    def __init__(self, channel, in_dim) -> None:
        super(Decoder, self).__init__()
        self.channel=channel
        self.in_dim=in_dim

        self.linear_block=nn.Sequential(
            nn.Linear(self.in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64*4*4),
            nn.ReLU(),
        )
        self.convtrans_block=nn.Sequential(
            nn.ConvTranspose(64, 64, kernel_size=4, stride=2, padding=1), # 64, 8, 8
            nn.ReLU(),
            nn.BatchNorm(64),
            nn.ConvTranspose(64, 32, kernel_size=4, stride=2, padding=1), # 32, 16, 16
            nn.ReLU(),
            nn.BatchNorm(32),
            nn.ConvTranspose(32, self.channel, kernel_size=4, stride=2, padding=1), # 3, 32, 32
        )

    def execute(self, x):
        x=self.linear_block(x)
        x=jt.reshape(x, (-1, 64, 4, 4))
        x=self.convtrans_block(x)
        return x

class Beta_VAE(nn.Module):
    def __init__(self, z_dim=10, channel=3):
        super(Beta_VAE, self).__init__()
        self.z_dim = z_dim
        self.channel = channel

        self.encoder = Encoder(self.channel, self.z_dim*2)

        self.decoder = Decoder(self.channel, self.z_dim)

    def execute(self, x):
        encoded=self.encoder(x)
        mean=encoded[:,:self.z_dim]
        logvar=encoded[:,self.z_dim:]

        std=jt.exp(logvar).sqrt()

        epsilon=jt.normal(jt.zeros((x.shape[0], self.z_dim)), jt.ones((x.shape[0], self.z_dim)))

        z=mean+std*epsilon

        x_hat=self.decoder(z)

        return x_hat, mean, std
    
def circle(iterable):
    while True:
        for x in iterable:
            yield x

class Beta_VAE_Manager():
    def __init__(self, train_loader, test_loader, wandb_run, num_steps=100000, show_real=True) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.wandb_run = wandb_run
        self.num_steps=num_steps
        self.show_real=show_real

        self.model = Beta_VAE()

        self.model.apply(init_weights)

        self.optimizer = nn.Adam(self.model.parameters(), lr=0.005)

    def train(self):
        self.model.train()

        train_gen = iter(circle(self.train_loader))

        recon_loss_list=[]
        KL_loss_list=[]
        tot_loss_list=[]
        for i in tqdm(range(self.num_steps), desc="Training"):
            input, label=next(train_gen)

            input=jt.float32(input)
            input=jt.transpose(input, (0, 3, 1, 2))
            input=input/255

            output, mean, std=self.model(input)

            # 计算重构损失
            recon_loss=nn.mse_loss(output, input, reduction='mean')
            # 计算KL散度
            KL_loss=(mean**2+std**2-1-2*std.log()).sum()/2
            # print("mean")
            # print(mean)
            # print("std")
            # print(std)
            # print("kl")
            # print(KL_loss)

            tot_loss=recon_loss+KL_loss

            tot_loss.sync()

            self.optimizer.step(tot_loss)

            recon_loss_list.append(recon_loss.data[0])
            KL_loss_list.append(KL_loss.data[0])
            tot_loss_list.append(tot_loss.data[0])

            if i%100==0:
                recon_loss_mean=np.mean(recon_loss_list)
                KL_loss_mean=np.mean(KL_loss_list)
                tot_loss_mean=np.mean(tot_loss_list)

                recon_loss_list.clear()
                KL_loss_list.clear()
                tot_loss_list.clear()

                if self.show_real:
                    real_imgs=jt.misc.make_grid(input, nrow=8)
                    real_imgs=real_imgs.numpy()*255
                    real_imgs=real_imgs.transpose(1,2,0).astype(np.uint8)
                    self.wandb_run.log({"Real imgs":wandb.Image(real_imgs)}, step=i)


                imgs=jt.misc.make_grid(output, nrow=8)
                imgs=imgs.numpy()*255
                imgs=imgs.transpose(1,2,0).astype(np.uint8)

                self.wandb_run.log({
                    "Reconstruction Loss":recon_loss_mean,
                }, step=i)
                self.wandb_run.log({"KL Loss":KL_loss_mean}, step=i)
                self.wandb_run.log({"Total Loss":tot_loss_mean}, step=i)
                self.wandb_run.log({"Imgs": wandb.Image(imgs)}, step=i)


                print(f"    step {i}:")
                print(f"        Reconstruction Loss: {recon_loss_mean}")
                print(f"        KL Loss: {KL_loss_mean}")
                print(f"        Total Loss: {tot_loss_mean}")


    def test(self):
        self.model.eval()

        epsilon=jt.normal(jt.zeros((64, self.model.z_dim)), jt.ones((64, self.model.z_dim)))
        result=self.model.decoder(epsilon)

        result=jt.misc.make_grid(result, nrow=8)
        result=result.numpy()*255
        result=jt.transpose(result, (1,2,0)).astype(np.uint8)

        imgs=Image.fromarray(result.numpy())

        if not os.path.exists('results'):
            os.mkdir('results')

        imgs.save('results/Beta_VAE.png')


