# DeepTransformerGAN-DTGAN：低照度图像修复

## Languages

*   [English](README_en.md)


本项目实现了一种深度学习方法，用于对灰度图像进行着色。它使用了一种改进的DCGAN（深度卷积生成对抗网络）架构，具有自注意力机制、跨层连接和残差块。该模型在转换为LAB颜色空间的RGB图像上进行训练，使用A和B通道作为颜色信息标签。

## 概述

该项目包括以下关键组件：

*   **`engine.py`**: 包含核心训练和预测逻辑。
*   **`train.py`**: 实现了带有参数解析的训练脚本。
*   **`predict.py`**: 实现了图像着色和实时视频着色的预测脚本。
*   **`models/`**: 包含模型定义：
    *   `base_mode.py`: 定义了生成器（Generator）、判别器（Discriminator）和评论家（Critic）模型。
    *   `Repvit.py`: 定义了RepViTBlock。
    *   `common.py`: 定义了通用层和块。
*   **`datasets/`**: 包含数据集加载和预处理逻辑：
    *   `data_set.py`: 实现了用于加载LOLdataset的`LowLightDataset`类。
*   **`utils/`**: 包含实用函数：
    *   `color_trans.py`: 包含颜色空间转换函数（RGB到LAB和LAB到RGB）。
    *   `loss.py`: 定义了自定义损失函数。
    *   `misic.py`: 包含训练的辅助函数。

## 安装

1.  克隆存储库：

    ```bash
    git clone <repository_url>
    cd INR2RGB
    ```

2.  安装所需的依赖项：

    ```bash
    pip install -r requirements.txt
    ```

    `requirements.txt` 文件包含以下软件包：

    ```
    torch
    opencv-python<=4.9.0.80
    numpy<=1.26.3
    timm<=0.9.16
    torchvision
    tqdm<=4.65.0
    ptflops<= 0.7.2.2
    torcheval<=0.70
    rich~=13.7.1
    cython
    PyYAML~=6.0.1
    ```

## 用法

### 训练

要训练模型，请运行 `train.py` 脚本：

```bash
python train.py --data <数据集路径> --epochs <训练轮数> --batch_size <批次大小>
```

*   `--data`: 数据集目录的路径（例如，`../datasets/LOLdataset`）。
*   `--epochs`: 训练轮数（默认值：500）。
*   `--batch_size`: 批次大小（默认值：16）。
*   `--img_size`: 输入图像的大小（默认值：(256, 256)）。
*   `--optimizer`: 要使用的优化器（默认值：Adam）。
*   `--num_workers`: 数据加载器的工作线程数（默认值：0）。在Windows中必须设置为0。
*   `--seed`: 用于重现性的随机种子（默认值：随机整数）。
*   `--resume`: 用于恢复训练的检查点路径。
*   `--amp`: 是否使用自动混合精度（AMP）进行训练。
*   `--cuDNN`: 是否使用cuDNN进行加速训练。
*   `--loss`: 要使用的损失函数（默认值：bce）。
*   `--lr`: 学习率（默认值：3.5e-4）。
*   `--momentum`: SGD优化器的动量（默认值：0.5）。
*   `--depth`: 生成器的深度（默认值：1）。
*   `--weight`: 生成器的权重（默认值：1）。
*   `--device`: 用于训练的设备（默认值：cuda）。
*   `--save_path`: 用于保存训练结果的目录（默认值：runs/）。
*   `--benchmark`: 是否使用 `torch.benchmark` 来加速训练。
*   `--deterministic`: 是否使用确定性初始化。
*   `--draw_model`: 是否将模型图绘制到TensorBoard。

示例：

```bash
python train.py --data ../datasets/LOLdataset --epochs 200 --batch_size 32
```

### 预测

要使用经过训练的模型对图像进行着色，请运行 `predict.py` 脚本：

```bash
python predict.py --data <图像或目录路径> --model <生成器检查点路径>
```

*   `--data`: 要着色的图像或包含图像的目录的路径。使用 `0` 打开相机进行实时着色。
*   `--model`: 生成器检查点文件的路径（例如，`runs/train(3)/generator/last.pt`）。
*   `--batch_size`: 批次大小（默认值：16）。
*   `--img_size`: 输入图像的大小（默认值：(256, 256)）。
*   `--num_workers`: 数据加载器的工作线程数（默认值：0）。
*   `--device`: 用于预测的设备（默认值：cuda）。
*   `--save_path`: 用于保存着色图像的目录（默认值：runs/）。

示例：

```bash
python predict.py --data test_image.png --model runs/train(3)/generator/last.pt
```

要使用相机进行实时着色：

```bash
python predict.py --data 0 --model runs/train(3)/generator/last.pt
```

## 或者使用我训练好的模型以及处理过的视频进行预测

 基于kitti数据集训练的模型下载链接：[https://cloud.189.cn/web/share?code=2yuIVbFreMjq](https://cloud.189.cn/web/share?code=2yuIVbFreMjq（访问码：rbr4）)
 处理前视频：[https://cloud.189.cn/web/share?code=YJVnyqqYJBru](https://cloud.189.cn/web/share?code=YJVnyqqYJBru（访问码：8lkp）)
 处理后视频：[https://cloud.189.cn/web/share?code=J3q2uyiMRJvi（访问码：lmw6）](https://cloud.189.cn/web/share?code=J3q2uyiMRJvi（访问码：lmw6）)

 ###  你可以使用我处理的视频，或者自己通过darker.py处理自己想要处理的数据集及视频

### 数据集

该项目使用 LOLdataset，其中包含低光照和正常光照图像对。`datasets/data_set.py` 中的 `LowLightDataset` 类处理加载和预处理数据。

数据集目录应具有以下结构：

```
LOLdataset/
    our485/
        high/
            *.png
        low/
            *.png
    eval15/
        high/
            *.png
        low/
            *.png
```

### 模型

着色过程的核心是生成器模型，在 `models/base_mode.py` 中定义。它使用 RepViT 块、SPPELAN、PSA 和其他自定义层来从灰度输入生成着色图像。判别器（或 WGAN 版本中的评论家）用于区分真实图像和生成图像，从而指导生成器产生更逼真的结果。

## 使用 WGAN 进行训练

`engine.py` 文件还包括一个 `train_WGAN` 函数，用于使用 WGAN-GP（具有梯度惩罚的 Wasserstein GAN）方法训练模型。这可能会导致更稳定的训练和更好的结果。要使用它，您需要修改 `train.py` 脚本以调用 `train_WGAN` 而不是 `train`。

## 许可证

[检查 `LICENSE` 文件]

## Examples



Real Image:

![Real Image](examples/real.png)
Real image from the dataset.

Fake Image:

![Fake Image](
![Real Image](examples/fake.png)
Generated image by the model.

### Examples Videos

*   examples/dark_video.mp4: Dark video example.
<<<<<<< HEAD
![原始视频](examples\upc_dark.gif)
*   examples/upc_dark.mp4: UPC dark video example.
![暗处理](examples\dark_video.gif)

*   examples/upc_dark.mp4: generated video example.
![模型生成](examples\fake.gif) 