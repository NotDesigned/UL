"""
Unified Latents (UL) - Dataset
支持 ImageNet 和通用图像文件夹格式。
"""

import os
from pathlib import Path
from PIL import Image

import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


# ============================================================
# 预处理
# ============================================================

def get_transform(resolution: int = 512, split: str = 'train') -> transforms.Compose:
    """
    标准图像预处理流水线。

    训练集：随机裁剪 + 随机水平翻转（数据增强）
    验证集：中心裁剪（确定性，方便复现评测结果）

    输出范围：[-1, 1]，扩散模型的惯例。
    ImageNet 的 mean/std normalize 不适用于扩散模型，
    因为扩散模型假设数据分布接近标准正态，[-1,1] 线性缩放更符合这个假设。
    """
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(resolution, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),                          # [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5],
                                  [0.5, 0.5, 0.5]),         # → [-1, 1]
        ])
    else:
        # 验证/测试：先缩放短边到略大于 resolution，再中心裁剪
        scale_size = int(resolution * 1.1)
        return transforms.Compose([
            transforms.Resize(scale_size),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                  [0.5, 0.5, 0.5]),
        ])


# ============================================================
# ImageNet Dataset
# ============================================================

class ImageNetDataset(Dataset):
    """
    ImageNet 数据集包装。

    期望的目录结构（torchvision 标准格式）：
        root/
          train/
            n01440764/   ← synset 文件夹
              xxx.JPEG
              ...
            n01443537/
              ...
          val/
            n01440764/
              ...

    如果只有图像文件夹（无类别子目录），用 FlatImageDataset。
    """
    def __init__(self, root: str, split: str = 'train',
                 resolution: int = 512):
        super().__init__()
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"找不到 {split_dir}，请确认 ImageNet 目录结构正确。"
            )
        self.dataset = datasets.ImageFolder(
            root=split_dir,
            transform=get_transform(resolution, split),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # 只返回图像，不返回类别标签（无条件生成）
        image, _ = self.dataset[idx]
        return image


class FlatImageDataset(Dataset):
    """
    通用平铺图像文件夹，目录下直接放图像文件，无需类别子目录。
    适合小规模实验或自定义数据集。

    支持格式：jpg / jpeg / png / webp
    """
    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

    def __init__(self, root: str, split: str = 'train',
                 resolution: int = 512):
        super().__init__()
        self.transform = get_transform(resolution, split)
        self.paths = sorted([
            p for p in Path(root).rglob('*')
            if p.suffix.lower() in self.EXTENSIONS
        ])
        if len(self.paths) == 0:
            raise FileNotFoundError(f"{root} 下找不到任何图像文件。")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)


# ============================================================
# DataLoader 工厂
# ============================================================

def get_dataloader(
    root: str,
    split: str = 'train',
    resolution: int = 512,
    batch_size: int = 16,
    num_workers: int = 8,
    flat: bool = False,
) -> DataLoader:
    """
    创建 DataLoader。

    Args:
        root:        数据集根目录
        split:       'train' 或 'val'
        resolution:  图像分辨率
        batch_size:  批大小
        num_workers: 数据加载线程数
        flat:        True 表示使用 FlatImageDataset（无类别子目录）
    """
    def _worker_init_fn(worker_id: int):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    dataset_cls = FlatImageDataset if flat else ImageNetDataset
    dataset     = dataset_cls(root, split, resolution)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,        # 加速 CPU→GPU 传输
        drop_last=(split == 'train'),   # 训练时丢弃不完整的最后一个 batch
        persistent_workers=(num_workers > 0),   # 避免每个 epoch 重建 worker
        worker_init_fn=_worker_init_fn,
    )