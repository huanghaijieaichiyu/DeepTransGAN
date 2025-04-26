import os
import sys
from pathlib import Path
from typing import List, Tuple


def get_image_files(directory: str) -> List[Tuple[str, str]]:
    """获取目录下所有图片文件。

    Args:
        directory (str): 目标目录

    Returns:
        List[Tuple[str, str]]: 包含(root, filename)的列表
    """
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')
    files_list = []

    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    files_list.append((root, file))
    except Exception as e:
        print(f"Error scanning directory: {e}")
        sys.exit(1)

    return files_list


def get_image_format(files: List[Tuple[str, str]]) -> str:
    """获取第一个图片文件的格式。

    Args:
        files (List[Tuple[str, str]]): 文件列表

    Returns:
        str: 图片格式
    """
    if not files:
        print("No image files found in the directory!")
        sys.exit(1)

    _, first_file = files[0]
    return Path(first_file).suffix[1:]  # Remove the leading dot


def rename_files(data_dir: str) -> None:
    """批量重命名文件。

    Args:
        data_dir (str): 数据集目录。
    """
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"Directory '{data_dir}' does not exist!")
        return

    # 获取所有图片文件
    print("Scanning for image files...")
    files_to_rename = get_image_files(data_dir)

    if not files_to_rename:
        print("No image files found!")
        return

    # 获取图片格式
    img_format = get_image_format(files_to_rename)

    # 按文件名自然排序
    files_to_rename.sort(key=lambda x: x[1])

    total_files = len(files_to_rename)
    print(f"\nFound {total_files} image files to rename")

    # 执行重命名
    renamed_count = 0
    skipped_count = 0

    for i, (root, file) in enumerate(files_to_rename):
        src = os.path.join(root, file)
        dst = os.path.join(root, f"{i}.{img_format}")

        # 如果源文件和目标文件相同，跳过
        if src == dst:
            continue

        # 如果目标文件已存在则跳过
        if os.path.exists(dst):
            print(f"Skipping {file} - {i}.{img_format} already exists")
            skipped_count += 1
            continue

        try:
            os.rename(src, dst)
            renamed_count += 1
            # 每处理10个文件显示一次进度
            if renamed_count % 10 == 0:
                print(f"Progress: {renamed_count}/{total_files} files renamed")
        except Exception as e:
            print(f"Error renaming {file}: {e}")

    # 打印最终统计
    print(f"\nRename completed!")
    print(f"Successfully renamed: {renamed_count} files")
    print(f"Skipped: {skipped_count} files")


if __name__ == '__main__':
    # 1. 设置数据集目录
    data_dir = "../datasets/NuScenes"  # 替换成你的数据集路径
    # 2. 重命名文件
    rename_files(data_dir)
