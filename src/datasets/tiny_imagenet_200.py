import os
import shutil
import logging
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def make_tiny_imagenet200(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder="tiny-imagenet-200",
    training=True,
    drop_last=True
):
    dataset = TinyImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=training
    )
    logger.info(f"Tiny-ImageNet dataset created, {len(dataset)} samples")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        pin_memory=pin_mem,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=num_workers > 0
    )

    logger.info("Tiny-ImageNet dataloader created")
    return dataset, loader, sampler


class TinyImageNet(ImageFolder):
    """Dataset Tiny-ImageNet chuẩn hóa cho Windows, convert ảnh sang .jpg."""

    def __init__(self, root, image_folder="tiny-imagenet-200", transform=None, train=True):
        self.transform = transform
        self.train = train
        split = "train" if train else "val"
        base_path = os.path.join(root, image_folder)

        if split == "train":
            data_path = self._prepare_train(base_path)
        else:
            data_path = self._prepare_val(base_path)

        logger.info(f"Loading Tiny-ImageNet from {data_path}")
        super().__init__(root=data_path, transform=transform)

    def _convert_to_jpg(self, src, dst):
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.save(dst, "JPEG")
        except Exception as e:
            logger.warning(f"Failed to convert {src} to jpg: {e}")

    def _prepare_train(self, base_path):
        src_root = os.path.join(base_path, "train")
        dst_root = os.path.join(base_path, "train_flat")

        if os.path.exists(dst_root):
            shutil.rmtree(dst_root)

        logger.info("Preparing Tiny-ImageNet train split")
        os.makedirs(dst_root, exist_ok=True)

        for idx, class_id in enumerate(os.listdir(src_root)):
            src_class = os.path.join(src_root, class_id)
            if not os.path.isdir(src_class):
                continue
            img_dir = os.path.join(src_class, "images")
            if not os.path.exists(img_dir):
                continue
            dst_class = os.path.join(dst_root, class_id)
            os.makedirs(dst_class, exist_ok=True)

            for img in os.listdir(img_dir):
                if img.lower().endswith((".jpeg", ".jpg", ".png")):
                    src = os.path.join(img_dir, img)
                    dst_name = os.path.splitext(img)[0] + ".jpg"
                    dst = os.path.join(dst_class, dst_name)
                    self._convert_to_jpg(src, dst)

            logger.info(f"[Train] Processed class {idx+1}/{len(os.listdir(src_root))}: {class_id}")

        total = sum(len(os.listdir(os.path.join(dst_root, cls))) for cls in os.listdir(dst_root))
        logger.info(f"Train split ready: {len(os.listdir(dst_root))} classes, {total} images")
        return dst_root

    def _prepare_val(self, base_path):
        src_root = os.path.join(base_path, "val")
        dst_root = os.path.join(base_path, "val_flat")
        anno_file = os.path.join(src_root, "val_annotations.txt")

        if os.path.exists(dst_root):
            shutil.rmtree(dst_root)
        if not os.path.exists(anno_file):
            raise RuntimeError("val_annotations.txt not found")

        logger.info("Preparing Tiny-ImageNet val split")
        os.makedirs(dst_root, exist_ok=True)

        mapping = {}
        with open(anno_file, "r") as f:
            for line in f:
                img, cls = line.split()[:2]
                mapping[img] = cls

        img_dir = os.path.join(src_root, "images")
        for idx, (img, cls) in enumerate(mapping.items()):
            cls_dir = os.path.join(dst_root, cls)
            os.makedirs(cls_dir, exist_ok=True)

            src = os.path.join(img_dir, img)
            dst_name = os.path.splitext(img)[0] + ".jpg"
            dst = os.path.join(cls_dir, dst_name)
            self._convert_to_jpg(src, dst)

            if (idx+1) % 100 == 0 or idx == len(mapping)-1:
                logger.info(f"[Val] Processed {idx+1}/{len(mapping)} images")

        total = sum(len(os.listdir(os.path.join(dst_root, cls))) for cls in os.listdir(dst_root))
        logger.info(f"Val split ready: {len(os.listdir(dst_root))} classes, {total} images")
        return dst_root
