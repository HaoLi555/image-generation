import jittor as jt
import wandb
from jittor.dataset.cifar import CIFAR10
from models import ModelManger
import argparse




if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='wgan')
    args=parser.parse_args()

    if jt.has_cuda:
        jt.flags.use_cuda = 1

    wandb_run = wandb.init(
        project='image-gen',
    )

    train_loader = CIFAR10(train=True).set_attrs(shuffle=True, batch_size=64)
    test_loader = CIFAR10(train=False).set_attrs(shuffle=True, batch_size=64)


    manager=ModelManger[args.model.lower()](train_loader, test_loader, wandb_run=wandb_run)
    manager.train()
    manager.test()