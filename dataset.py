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
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    """
    创建 DataLoader。

    Args:
        root:        数据集根目录
        split:       'train' 或 'val'
        resolution:  图像分辨率
        batch_size:  批大小
        num_workers: 数据加载线程数
        flat:        True 表示使用 FlatImageDataset（无类别子目录）
        distributed: 是否使用 DistributedSampler（DDP 模式）
        rank:        当前进程的 rank
        world_size:  总进程数

    Returns:
        (DataLoader, sampler) — sampler 为 DistributedSampler 或 None
    """
    from torch.utils.data.distributed import DistributedSampler

    def _worker_init_fn(worker_id: int):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    dataset_cls = FlatImageDataset if flat else ImageNetDataset
    dataset     = dataset_cls(root, split, resolution)

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    shuffle = (split == 'train') if sampler is None else False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init_fn,
    )
    return loader, sampler
