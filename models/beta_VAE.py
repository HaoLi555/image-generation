import jittor as jt
from jittor import nn
import jittor.init as init
from tqdm import tqdm
import wandb
import numpy as np
import os 

def init_weights(m):
    if type(m)==nn.Conv2d:
        init.trunc_normal_(m.weight, std=0.02)
    elif type(m)==nn.ConvTranspose2d:
        init.trunc_normal_(m.weight, std=0.02)
    elif type(m)==nn.BatchNorm2d:
        init.trunc_normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0)

class Beta_VAE(nn.Module):
    def __init__(self, z_dim=10, channel=3):
        super(Beta_VAE, self).__init__()
        self.z_dim = z_dim
        self.channel = channel

        self.encoder = nn.Sequential(
            nn.Conv(channel, 32, kernel_size=4, stride=2, padding=1), # 32, 16, 16
            nn.ReLU(),
            nn.Conv(32, 64, kernel_size=4, stride=2, padding=1), # 64, 8, 8
            nn.ReLU(),
            nn.Conv(64, 64, kernel_size=4, stride=2, padding=1), # 64, 4, 4
            nn.ReLU(),
            nn.Conv(64, 256, kernel_size=4, stride=1), # 256, 1, 1
            nn.ReLU(),
            nn.Reshape((-1, 256*1*1)),
            nn.Linear(256, z_dim*2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64*4*4),
            nn.ReLU(),
            nn.Reshape((-1, 64, 4, 4)),
            nn.ConvTranspose(64, 64, kernel_size=4, stride=2, padding=1), # 64, 8, 8
            nn.ReLU(),
            nn.ConvTranspose(64, 32, kernel_size=4, stride=2, padding=1), # 32, 16, 16
            nn.ReLU(),
            nn.ConvTranspose(32, channel, kernel_size=4, stride=2, padding=1), # 3, 32, 32
        )

    def forward(self, x):
        encoded=self.encoder(x)
        mean=encoded[:,:self.z_dim]
        std=encoded[:,self.z_dim:]

        epsilon=jt.normal(jt.zeros_like(mean), jt.ones_like(std))

        z=mean+std*epsilon

        x_hat=self.decoder(z)

        return x_hat, mean, std
    
def circle(iterable):
    while True:
        for x in iterable:
            yield x

class Beta_VAE_Manager():
    def __init__(self, train_loader, test_loader, wandb_run, num_steps=10000) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.wandb_run = wandb_run

        self.model = Beta_VAE()

        self.model.apply(init_weights)

        self.optimizer = nn.Adam(self.model.parameters(), lr=1e-4)

    def train(self):
        self.model.train()

        train_gen = iter(circle(self.train_loader))

        recon_loss_list=[]
        KL_loss_list=[]
        tot_loss_list=[]
        for i in tqdm(range(self.num_steps), desc="Training"):
            input=next(train_gen)

            input=input*2/255-1
            input=jt.transpose(input, (0, 3, 1, 2))
            input=jt.float32(input)

            output, mean, std=self.model(input)

            # 计算重构损失
            recon_loss=nn.mse_loss(output, input, reduction='mean')
            # 计算KL散度
            KL_loss=-(mean**2+std**2-1-2*std.log()).sum()/2

            tot_loss=recon_loss+KL_loss

            tot_loss.sync()

            self.optimizer.step(tot_loss)

            recon_loss_list.append(recon_loss.data[0])
            KL_loss_list.append(KL_loss.data[0])
            tot_loss_list.append(tot_loss.data[0])

            if i%100==0:
                recon_loss_mean=recon_loss_list.mean()
                KL_loss_mean=KL_loss_list.mean()
                tot_loss_mean=tot_loss_list.mean()

                recon_loss_list.clear()
                KL_loss_list.clear()
                tot_loss_list.clear()

                imgs=jt.misc.make_grid(output, nrow=8)
                imgs=(imgs.numpy()+1)/2*255
                imgs=imgs.transpose(1,2,0).astype(np.uint8)

                self.wandb_run.log({
                    "Reconstruction Loss":recon_loss_list[-1],
                    "KL Loss":KL_loss_list[-1],
                    "Total Loss":tot_loss_list[-1],
                    "Imgs": wandb.Image(imgs)
                }, step=i)

    def test(self):
        self.model.eval()

        epsilon=jt.normal(jt.zeros((64, self.model.z_dim)), jt.ones((64, self.model.z_dim)))
        result=self.model.decoder(epsilon)

        if not os.path.exists('results'):
            os.mkdir('results')

        jt.misc.save_image(result, "results/Beta_VAE.png", nrow=8)


