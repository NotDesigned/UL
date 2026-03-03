"""
Unified Latents (UL) - Dataset
使用 HuggingFace Datasets Parquet 格式加载图像数据集。
"""

import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset as hf_load_dataset


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
# Dataset
# ============================================================

class HFImageDataset(Dataset):
    """
    通用 HuggingFace Datasets Parquet 格式图像数据集包装。

    适用于任何以 HuggingFace datasets 标准格式存储在本地的图像数据集，
    包括 mini-imagenet、imagenet-1k 等，只需提供本地根目录。

    期望的目录结构：
        root/
          data/
            train-*.parquet
            validation-*.parquet
            test-*.parquet
          README.md              ← 包含 dataset_info 的 YAML 元数据

    数据集字段要求：
        image  —— PIL.Image（HuggingFace Image feature 自动解码）
        label  —— 类别整数（可选，训练时不使用）
    """

    def __init__(self, root: str, split: str = 'train', resolution: int = 512):
        super().__init__()
        self.dataset   = hf_load_dataset(root, split=split, trust_remote_code=False)
        self.transform = get_transform(resolution, split)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.dataset[idx]['image']   # HuggingFace Image feature → PIL.Image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return self.transform(img.convert('RGB'))


# ============================================================
# DataLoader 工厂
# ============================================================

def get_dataloader(
    root: str,
    split: str = 'train',
    resolution: int = 512,
    batch_size: int = 16,
    num_workers: int = 8,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    """
    创建 DataLoader。

    Args:
        root:        数据集根目录（HuggingFace Parquet 格式）
        split:       'train' / 'validation' / 'test'
        resolution:  图像分辨率
        batch_size:  批大小
        num_workers: 数据加载线程数
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

    dataset = HFImageDataset(root, split, resolution)

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train') if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init_fn,
    )
    return loader, sampler
