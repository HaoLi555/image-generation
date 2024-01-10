from .beta_VAE import Beta_VAE_Manager
from .WGAN import WGAN_Manager

ModelManger = {
    'beta_vae': Beta_VAE_Manager,
    'wgan': WGAN_Manager
}
