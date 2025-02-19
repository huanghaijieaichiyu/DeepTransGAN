import argparse
from DeepTranserGAN import train, train_WGAN
import random


def args():
    parser = argparse.ArgumentParser()  # 命令行选项、参数和子命令解析器
    parser.add_argument("--data", type=str,
                        default='../datasets/kitti_LOL', help="path to dataset")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="number of epochs of training")  # 迭代次数
    parser.add_argument("--batch_size", type=int, default=16,
                        help="size of the batches")  # batch大小
    parser.add_argument("--img_size", type=tuple,
                        default=(256, 256), help="size of the image")
    parser.add_argument("--optimizer", type=str, default='Adam',
                        choices=['AdamW', 'SGD', 'Adam', 'lion', 'rmp'])
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of data loading workers, if in windows, must be 0"
                        )
    parser.add_argument("--seed", type=int,
                        default=random.randint(1, 100000), help="random seed")
    parser.add_argument("--resume", type=str,
                        default='', help="path to two latest checkpoint.")
    parser.add_argument("--amp", type=bool, default=True,
                        help="Whether to use amp in mixed precision")
    parser.add_argument("--cuDNN", type=bool, default=False,
                        help="Wether use cuDNN to celerate your program")
    parser.add_argument("--loss", type=str, default='BCEBlurWithLogitsLoss',
                        choices=['BCEBlurWithLogitsLoss', 'mse', 'bce',
                                 'FocalLoss'], help="loss function")
    parser.add_argument("--lr", type=float, default=0.5e-4,
                        help="learning rate, for adam is 1-e3, SGD is 1-e2")  # 学习率
    parser.add_argument("--momentum", type=float, default=0.5,
                        help="momentum for adam and SGD")
    parser.add_argument("--depth", type=float, default=1,
                        help="depth of the generator")
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of the generator")
    parser.add_argument("--model", type=str, default="train",
                        help="train or test model")
    parser.add_argument("--b1", type=float, default=0.9,
                        help="adam: decay of first order momentum of gradient")  # 动量梯度下降第一个参数
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")  # 动量梯度下降第二个参数
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                        help="~~")  # 每次迭代的训练步数
    parser.add_argument("--wgan", type=bool, default=False,
                        help="whether to use wgan to train")
    parser.add_argument("--device", type=str, default='cuda', choices=['cpu', 'cuda'],
                        help="select your device to train, if you have a gpu, use 'cuda:0'!")  # 训练设备
    parser.add_argument("--save_path", type=str, default='runs/',
                        help="where to save your data")  # 保存位置
    parser.add_argument("--benchmark", type=bool, default=False, help="whether using torch.benchmark to accelerate "
                                                                      "training(not working in interactive mode)")
    parser.add_argument("--deterministic", type=bool, default=True,
                        help="whether to use deterministic initialization")
    parser.add_argument("--draw_model", type=bool, default=False,
                        help="whether to draw model graph to tensorboard")

    # 新增WGAN-GP参数
    parser.add_argument("--critic_iters", type=int, default=5,
                        help="Number of critic iterations per generator iteration (WGAN-GP)")
    parser.add_argument("--lambda_gp", type=float, default=10.0,
                        help="Gradient penalty coefficient (WGAN-GP)")
    parser.add_argument("--clip_value", type=float, default=0.01,
                        help="Weight clipping value (WGAN without GP)")

    #  此处开始训练
    arges = parser.parse_args()
    return arges


if __name__ == '__main__':
    arges = args()
    if arges.wgan:
        train_WGAN(arges)
    else:
        train(arges)
