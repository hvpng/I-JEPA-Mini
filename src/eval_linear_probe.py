# src/eval_linear_probe.py
import os
import csv
import json
import yaml
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

from src.helper import init_model


# -----------------------------
# TinyImageNet: build val_flat from val_annotations.txt
# -----------------------------
def _convert_to_jpg(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.save(dst, "JPEG")


def prepare_train_flat(root_path, image_folder):
    base = os.path.join(root_path, image_folder)
    src_root = os.path.join(base, "train")
    dst_root = os.path.join(base, "train_flat")

    # nếu đã có thì dùng luôn (tránh rebuild nhiều lần)
    if os.path.exists(dst_root) and len(os.listdir(dst_root)) > 0:
        return dst_root

    os.makedirs(dst_root, exist_ok=True)
    for class_id in os.listdir(src_root):
        img_dir = os.path.join(src_root, class_id, "images")
        if not os.path.isdir(img_dir):
            continue
        out_dir = os.path.join(dst_root, class_id)
        os.makedirs(out_dir, exist_ok=True)

        for fn in os.listdir(img_dir):
            if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                src = os.path.join(img_dir, fn)
                dst = os.path.join(out_dir, os.path.splitext(fn)[0] + ".jpg")
                if not os.path.exists(dst):
                    _convert_to_jpg(src, dst)

    return dst_root


def prepare_val_flat(root_path, image_folder):
    base = os.path.join(root_path, image_folder)
    src_root = os.path.join(base, "val")
    dst_root = os.path.join(base, "val_flat")
    anno = os.path.join(src_root, "val_annotations.txt")
    img_dir = os.path.join(src_root, "images")

    # nếu đã có thì dùng luôn
    if os.path.exists(dst_root) and len(os.listdir(dst_root)) > 0:
        return dst_root

    if not os.path.exists(anno):
        raise RuntimeError("Missing val_annotations.txt in val split")

    os.makedirs(dst_root, exist_ok=True)

    mapping = {}
    with open(anno, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]

    for img_name, cls in mapping.items():
        src = os.path.join(img_dir, img_name)
        dst = os.path.join(dst_root, cls, os.path.splitext(img_name)[0] + ".jpg")
        if not os.path.exists(dst):
            _convert_to_jpg(src, dst)

    return dst_root


# -----------------------------
# Linear probe core
# -----------------------------
@torch.no_grad()
def extract_image_feature(encoder, images, pool="mean", do_layernorm=True):
    # encoder returns tokens: [B, N, D]
    tokens = encoder(images, masks=None)
    if do_layernorm:
        tokens = F.layer_norm(tokens, (tokens.size(-1),))

    if pool == "mean":
        feat = tokens.mean(dim=1)      # [B, D]
    elif pool == "token0":
        feat = tokens[:, 0]            # [B, D]
    else:
        raise ValueError("pool must be 'mean' or 'token0'")
    return feat


def top1_acc(logits, targets):
    pred = torch.argmax(logits, dim=1)
    return (pred == targets).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="logs/.../params-ijepa.yaml hoặc configs/...yaml")
    ap.add_argument("--ckpt", required=True, help="logs/.../jepa-latest.pth.tar hoặc jepa-ep200.pth.tar")
    ap.add_argument("--use_encoder", default="target_encoder", choices=["target_encoder", "encoder"])
    ap.add_argument("--pool", default="mean", choices=["mean", "token0"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default=None)

    # ---- checkpoint options (added)
    ap.add_argument("--ckpt_freq", type=int, default=10, help="save LP checkpoint every N epochs")
    ap.add_argument("--save_best", action="store_true", help="also save best LP head checkpoint")
    ap.add_argument("--resume", action="store_true", help="resume LP training from lp-*-latest.pth.tar in out_dir")

    args = ap.parse_args()

    # ---- load config (training config đã dump ra params-ijepa.yaml)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["meta"]["model_name"]
    pred_depth = cfg["meta"]["pred_depth"]
    pred_emb_dim = cfg["meta"]["pred_emb_dim"]

    patch_size = int(cfg["mask"]["patch_size"])

    root_path = cfg["data"]["root_path"]
    image_folder = cfg["data"]["image_folder"]
    crop_size = int(cfg["data"]["crop_size"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- prepare datasets (train_flat + val_flat)
    train_dir = prepare_train_flat(root_path, image_folder)
    val_dir = prepare_val_flat(root_path, image_folder)

    # ---- transforms for supervised linear probe (weak aug)
    normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_tf = T.Compose([
        T.RandomResizedCrop(crop_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    val_tf = T.Compose([
        T.Resize(crop_size),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        normalize,
    ])

    train_set = ImageFolder(train_dir, transform=train_tf)
    val_set = ImageFolder(val_dir, transform=val_tf)
    num_classes = len(train_set.classes)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=True, persistent_workers=args.num_workers > 0
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=False, persistent_workers=args.num_workers > 0
    )

    # ---- build encoder (đúng kiến trúc như lúc train) rồi load weights từ ckpt
    encoder, _ = init_model(
        device=device,
        patch_size=patch_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim
    )
    encoder.eval()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if args.use_encoder not in ckpt:
        raise KeyError(f"Checkpoint thiếu key '{args.use_encoder}'. Keys hiện có: {list(ckpt.keys())}")

    encoder.load_state_dict(ckpt[args.use_encoder], strict=True)

    # freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False

    embed_dim = encoder.embed_dim
    head = nn.Linear(embed_dim, num_classes).to(device)

    # optimizer cho linear head
    opt = torch.optim.SGD(head.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # output
    out_dir = args.out_dir or os.path.dirname(args.ckpt)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"linear_probe_{os.path.basename(args.ckpt)}.csv")
    out_json = os.path.join(out_dir, f"linear_probe_{os.path.basename(args.ckpt)}.json")

    # ---- checkpoint paths (added; analogous to train.py)
    ckpt_base = os.path.splitext(os.path.basename(args.ckpt))[0]  # e.g. "jepa-ep100.pth"
    lp_latest_path = os.path.join(out_dir, f"lp-{ckpt_base}-latest.pth.tar")
    lp_save_path = os.path.join(out_dir, f"lp-{ckpt_base}-ep{{epoch}}.pth.tar")
    lp_best_path = os.path.join(out_dir, f"lp-{ckpt_base}-best.pth.tar")

    def save_lp_checkpoint(epoch, tr_loss, tr_acc, va_loss, va_acc, best_dict):
        save_dict = {
            "epoch": int(epoch),
            "head": head.state_dict(),
            "opt": opt.state_dict(),
            "best": dict(best_dict),
            "train_loss": float(tr_loss),
            "train_acc": float(tr_acc),
            "val_loss": float(va_loss),
            "val_acc": float(va_acc),
            "ckpt": args.ckpt,
            "config": args.config,
            "use_encoder": args.use_encoder,
            "pool": args.pool,
            "embed_dim": int(embed_dim),
            "num_classes": int(num_classes),
            "args": vars(args),
        }
        torch.save(save_dict, lp_latest_path)

        # Save periodic epoch checkpoint
        if args.ckpt_freq > 0 and (epoch % args.ckpt_freq == 0):
            torch.save(save_dict, lp_save_path.format(epoch=f"{epoch:03d}"))

        # Save best checkpoint if requested
        if args.save_best and best_dict.get("epoch", -1) == epoch:
            torch.save(save_dict, lp_best_path)

    # CSV header: keep legacy behavior (always overwrite) unless resuming (added)
    best = {"val_acc": -1.0, "epoch": -1}
    start_epoch = 1

    if args.resume and os.path.exists(lp_latest_path):
        lp_ckpt = torch.load(lp_latest_path, map_location="cpu")
        head.load_state_dict(lp_ckpt["head"], strict=True)
        opt.load_state_dict(lp_ckpt["opt"])
        best = lp_ckpt.get("best", best)
        start_epoch = int(lp_ckpt.get("epoch", 0)) + 1

        # If resuming, do not wipe old CSV; append instead (added)
        if not os.path.exists(out_csv):
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        print(f"[LP] Resumed from {lp_latest_path} at epoch={start_epoch-1}, best={best}")
    else:
        # legacy behavior: overwrite CSV each run
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(start_epoch, args.epochs + 1):
        # ---- train head
        head.train()
        tr_loss, tr_acc, n = 0.0, 0.0, 0
        for imgs, y in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                feat = extract_image_feature(encoder, imgs, pool=args.pool, do_layernorm=True)

            logits = head(feat)
            loss = criterion(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_loss += loss.item()
            tr_acc += top1_acc(logits, y)
            n += 1

        tr_loss /= max(n, 1)
        tr_acc /= max(n, 1)

        # ---- val
        head.eval()
        va_loss, va_acc, m = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, y in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                feat = extract_image_feature(encoder, imgs, pool=args.pool, do_layernorm=True)
                logits = head(feat)
                loss = criterion(logits, y)
                va_loss += loss.item()
                va_acc += top1_acc(logits, y)
                m += 1

        va_loss /= max(m, 1)
        va_acc /= max(m, 1)

        if va_acc > best["val_acc"]:
            best = {"val_acc": va_acc, "epoch": epoch}

        print(f"[LP][{epoch:03d}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} best={best['val_acc']:.4f}@{best['epoch']}")

        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, tr_loss, tr_acc, va_loss, va_acc])

        # ---- save checkpoint (added)
        save_lp_checkpoint(epoch, tr_loss, tr_acc, va_loss, va_acc, best)

    summary = {
        "ckpt": args.ckpt,
        "config": args.config,
        "use_encoder": args.use_encoder,
        "pool": args.pool,
        "best_val_acc": best["val_acc"],
        "best_epoch": best["epoch"],
        "num_classes": num_classes,
        "embed_dim": embed_dim,
        "lp_latest_ckpt": lp_latest_path,
        "lp_best_ckpt": lp_best_path if args.save_best else None,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", out_csv)
    print("Saved:", out_json)
    print("LP latest:", lp_latest_path)
    if args.save_best:
        print("LP best:", lp_best_path)


if __name__ == "__main__":
    main()
