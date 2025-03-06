# DeepTranserGAN - 低光照图像增强

这是一个基于PyTorch的深度学习项目，用于低光照图像增强。该项目使用生成对抗网络(GAN)和Wasserstein GAN with Gradient Penalty (WGAN-GP)来实现从低光照图像到正常光照图像的转换。

## 项目特点

- 支持多种GAN架构（标准GAN和WGAN-GP）
- 优化的注意力机制和特征提取
- 高效的训练策略（梯度累积、混合精度训练）
- 完善的评估指标（PSNR、SSIM）
- 支持图像和视频处理

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA (推荐用于加速训练)

## 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/DeepTranserGAN.git
cd DeepTranserGAN
```

2. 创建并激活虚拟环境（可选但推荐）：

```bash
# 使用conda
conda create -n deepgan python=3.8
conda activate deepgan

# 或使用venv
python -m venv deepgan
# Windows
deepgan\Scripts\activate
# Linux/Mac
source deepgan/bin/activate
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

## 数据集准备

项目使用LOL数据集（Low-Light paired dataset）或类似的低光照/正常光照配对数据集。

数据集结构应如下：
```
datasets/
└── kitti_LOL/
    ├── eval15/
    │   ├── high/  # 测试集 - 正常光照图像
    │   └── low/   # 测试集 - 低光照图像
    └── our485/
        ├── high/  # 训练集 - 正常光照图像
        └── low/   # 训练集 - 低光照图像
```

您可以从[这里](https://daooshee.github.io/BMVC2018website/)下载LOL数据集，或使用自己的数据集（确保遵循相同的目录结构）。

## 使用方法

### 训练模型

#### 标准GAN训练

```bash
python DeepTranserGAN.py --mode train --data datasets/kitti_LOL --save_path checkpoints --epochs 200 --batch_size 8 --lr 0.0002 --device cuda --optimizer adamw --loss bceblurwithlogitsloss --autocast True --grad_accum_steps 2
```

#### WGAN-GP训练

```bash
python DeepTranserGAN.py --mode train_wgan --data datasets/kitti_LOL --save_path checkpoints --epochs 200 --batch_size 8 --lr 0.0002 --device cuda --optimizer adamw --autocast True --grad_accum_steps 2
```

### 参数说明

- `--mode`: 运行模式 (`train`, `train_wgan`, `predict`)
- `--data`: 数据集路径
- `--save_path`: 模型保存路径
- `--epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--lr`: 学习率
- `--device`: 使用设备 (`cuda` 或 `cpu`)
- `--optimizer`: 优化器类型 (`adam`, `adamw`, `sgd`, `lion`, `rmp`)
- `--loss`: 损失函数 (`bceblurwithlogitsloss`, `mse`, `focalloss`, `bce`)
- `--autocast`: 是否使用混合精度训练 (`True` 或 `False`)
- `--grad_accum_steps`: 梯度累积步数
- `--patience`: 早停耐心值
- `--resume`: 从检查点恢复训练

### 图像增强预测

```bash
python DeepTranserGAN.py --mode predict --data datasets/kitti_LOL/eval15 --model checkpoints/train/generator/best.pt --save_path results --device cuda
```

### 视频增强预测

```bash
python DeepTranserGAN.py --mode predict --video True --data path/to/your/video.mp4 --model checkpoints/train/generator/best.pt --save_path results --save_video True --device cuda
```

## 训练技巧

1. **数据增强**: 默认启用了随机水平翻转、旋转和亮度变化，可以提高模型泛化能力。

2. **学习率调度**: 使用余弦退火学习率调度，帮助模型更好地收敛。

3. **梯度累积**: 对于内存受限的情况，可以增加`grad_accum_steps`来模拟更大的批次大小。

4. **混合精度训练**: 启用`autocast`可以加速训练并减少内存使用。

5. **早停机制**: 当验证性能不再提升时，训练会自动停止，避免过拟合。

## 模型评估

训练过程中，模型会定期在测试集上评估PSNR和SSIM指标。评估结果会记录在日志文件和TensorBoard中。

查看TensorBoard日志：

```bash
tensorboard --logdir checkpoints/train
```

## 项目结构

```
DeepTranserGAN/
├── DeepTranserGAN.py     # 主程序
├── models/               # 模型定义
│   ├── base_mode.py      # 基础网络架构
│   ├── common.py         # 通用模块
│   └── Repvit.py         # RepVit模块
├── datasets/             # 数据集
│   └── data_set.py       # 数据加载
├── utils/                # 工具函数
│   ├── loss.py           # 损失函数
│   └── misic.py          # 杂项工具
└── requirements.txt      # 依赖列表
```

## 常见问题

1. **内存不足**: 
   - 减小批次大小
   - 增加梯度累积步数
   - 启用混合精度训练

2. **训练不稳定**:
   - 尝试不同的学习率
   - 调整判别器预训练次数
   - 使用WGAN-GP代替标准GAN

3. **生成图像质量差**:
   - 增加训练轮数
   - 调整感知损失权重
   - 尝试不同的网络架构

## 引用

如果您在研究中使用了本项目，请引用：

```
@misc{DeepTranserGAN2025,
  author = {黄小海},
  title = {DeepTranserGAN: 低光照图像增强},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/yourusername/DeepTranserGAN}}
}
```

## 许可证

[MIT License](LICENSE)

## 联系方式

如有任何问题，请通过以下方式联系：

- 邮箱: huangxiaohai99@126.com
- GitHub Issues: [https://github.com/yourusername/DeepTranserGAN/issues](https://github.com/yourusername/DeepTranserGAN/issues)
