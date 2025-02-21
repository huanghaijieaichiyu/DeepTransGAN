import argparse
from DeepTranserGAN import predict


def parse_args():
    parser = argparse.ArgumentParser()  # 命令行选项、参数和子命令解析器
    parser.add_argument("--data", type=str, default='examples/dark_video.mp4',
                        help='path to dataset, and 0 is to open your camara')
    parser.add_argument(
        "--model", type=str, default='runs/train(1)/generator/best.pt', help="path to model")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="size of the batches")  # batch大小
    parser.add_argument("--img_size", type=tuple,
                        default=(256, 256), help="size of the image")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of data loading workers, if in Windows, must be 0"
                        )
    parser.add_argument("--device", type=str, default="cuda", choices='["cpu", "cuda"]',
                        help="select your device to train, if you have a gpu, use 'cuda:0'!")  # 训练设备
    parser.add_argument("--save_path", type=str, default='runs/',
                        help="where to save your data")  # 保存位置
    parser.add_argument("--sample_interval", type=int,
                        default=10, help="how often to sample the img")
    parser.add_argument("--video", type=bool, default=True,
                        help="whether to use video")
    parser.add_argument("--save_video", type=bool, default=True,
                        help="whether to save the video")
    opt = parser.parse_args()
    print(opt)
    return opt


if __name__ == '__main__':
    opt = parse_args()

    predict(opt)
