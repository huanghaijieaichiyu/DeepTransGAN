'''
数据集类，用于加载LOL数据集。

'''


import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image  # 导入 PIL Image


class LowLightDataset(Dataset):
    def __init__(self, image_dir, transform=None, phase="train"):
        """
        Args:
            image_dir (string): 包含LOLdataset的目录。
            transform (callable, optional): 可选的图像转换。
            phase (string, optional):  指定数据集的用途，可以是 "train" 或 "test"。
                                        "train" 加载 our485, "test" 加载 eval15.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.phase = phase  # 指定数据集的用途

        self.data = []  # 存储图像对的列表 (low_img_path, high_img_path)

        if phase == "train":
            subset = "our485"
        elif phase == "test":
            subset = "eval15"
        else:
            raise ValueError("phase must be 'train' or 'test'")

        if subset == "eval15":
            eval15_dir = os.path.join(image_dir, "eval15")
            if os.path.exists(eval15_dir):  # 检查目录是否存在
                eval15_high_dir = os.path.join(eval15_dir, "high")
                eval15_low_dir = os.path.join(eval15_dir, "low")
                eval15_image_names = [f for f in os.listdir(
                    eval15_high_dir) if any(
                    f.endswith(ext) for ext in [".png", ".jpg", ".jpeg"])]  # 兼容多种图片格式
                eval15_image_names.sort()
                for img_name in eval15_image_names:
                    low_img_path = os.path.join(eval15_low_dir, img_name)
                    high_img_path = os.path.join(eval15_high_dir, img_name)
                    self.data.append((low_img_path, high_img_path))

        elif subset == "our485":
            our485_dir = os.path.join(image_dir, "our485")
            if os.path.exists(our485_dir):  # 检查目录是否存在
                our485_high_dir = os.path.join(our485_dir, "high")
                our485_low_dir = os.path.join(our485_dir, "low")
                our485_image_names = [f for f in os.listdir(our485_high_dir) if any(
                    f.endswith(ext) for ext in [".png", ".jpg", ".jpeg"])]  # 兼容多种图片格式
                our485_image_names.sort()
                for img_name in our485_image_names:
                    low_img_path = os.path.join(our485_low_dir, img_name)
                    high_img_path = os.path.join(our485_high_dir, img_name)
                    self.data.append((low_img_path, high_img_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): 索引。

        Returns:
            tuple: (低光照图像, 正常光照图像)。
        """
        low_img_path, high_img_path = self.data[idx]
        if os.path.exists(low_img_path):
            # 使用 PIL 读取图像，确保是 RGB
            low_img = Image.open(low_img_path).convert("RGB")
        else:
            raise FileNotFoundError(f"低光图像文件不存在: {low_img_path}")

        # 使用 PIL 读取图像，确保是 RGB
        high_img = Image.open(high_img_path).convert("RGB")

        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)

        return low_img, high_img


# 示例用法:
if __name__ == '__main__':
    # 1. 设置数据集目录
    data_dir = "../datasets/kitti_LOL"  # 替换成你的数据集路径

    # 2. 定义图像转换
    transform = transforms.Compose([
        transforms.ToPILImage(),  # 先转换为PIL Image, 因为一些transform需要PIL Image作为输入
        transforms.Resize((256, 256)),  # 可选：调整大小
        transforms.ToTensor(),          # 转换为Tensor
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 可选：归一化
    ])

    # 3. 创建数据集实例
    # 创建训练数据集
    train_dataset = LowLightDataset(
        image_dir=data_dir, transform=transform, phase="train")

    # 创建测试/评估数据集
    test_dataset = LowLightDataset(
        image_dir=data_dir, transform=transform, phase="test")

    # 4. 创建 DataLoader
    batch_size = 4
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # 测试时通常不需要shuffle

    # 5. 迭代训练数据
    print("Training data:")
    for i, (low_images, high_images) in enumerate(train_dataloader):
        print(f"Batch {i+1}")
        print("Low images shape:", low_images.shape)
        print("High images shape:", high_images.shape)

        # 在这里进行你的模型训练
        # 例如:
        # outputs = model(low_images)
        # loss = criterion(outputs, high_images)

        if i == 2:  # 只迭代几个批次，方便演示
            break

    # 6. 迭代测试数据
    print("\nTesting data:")
    for i, (low_images, high_images) in enumerate(test_dataloader):
        print(f"Batch {i+1}")
        print("Low images shape:", low_images.shape)
        print("High images shape:", high_images.shape)

        # 在这里进行你的模型测试/评估
        # 例如:
        # outputs = model(low_images)
        # psnr = calculate_psnr(outputs, high_images)

        if i == 2:  # 只迭代几个批次，方便演示
            break
