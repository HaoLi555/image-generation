from PIL import Image
import imageio
import os
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='wgan')
    parser.add_argument('--beta', default=2, type=int)
    args = parser.parse_args()

    # 生成的图片所在的文件夹路径
    image_folder = 'train_wgan' if args.model.lower() == 'wgan' else 'train_beta_vae'

    if not os.path.exists('gifs'):
        os.makedirs('gifs')

    # 输出的 GIF 文件路径
    output_gif_path = 'gifs/wgan.gif' if args.model.lower(
    ) == 'wgan' else f'gifs/{args.beta}_vae.gif'

    # 获取图片文件列表
    images = [f"wgan_{i}.png" for i in range(0,100)] if args.model.lower() =='wgan' else [f"{args.beta}_vae{i}.png" for i in range(0, 50)]


    # 创建一个空白的图像列表
    image_list = []

    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        img = Image.open(img_path)
        image_list.append(img)

    # 保存图像列表为 GIF
    imageio.mimsave(output_gif_path, image_list, duration=0.2)

    print(f"动图已保存至: {output_gif_path}")
