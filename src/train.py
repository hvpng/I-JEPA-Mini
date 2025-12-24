# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import copy
import logging
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter
from src.utils.tensors import repeat_interleave_batch
from src.datasets.tiny_imagenet_200 import make_tiny_imagenet200
from src.helper import load_checkpoint, init_model, init_opt
from src.transforms import make_transforms

# ----------------------
log_timings = True
log_freq = 10
checkpoint_freq = 50
# ----------------------

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):
    # -------------------- META --------------------
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']

    # -------------------- DEVICE --------------------
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        logger.info(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        logger.warning('CUDA not available, using CPU. Training will be slower!')

    # -------------------- DATA --------------------
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']

    # -------------------- MASK --------------------
    allow_overlap = args['mask']['allow_overlap']
    patch_size = args['mask']['patch_size']
    num_enc_masks = args['mask']['num_enc_masks']
    min_keep = args['mask']['min_keep']
    enc_mask_scale = args['mask']['enc_mask_scale']
    num_pred_masks = args['mask']['num_pred_masks']
    pred_mask_scale = args['mask']['pred_mask_scale']
    aspect_ratio = args['mask']['aspect_ratio']

    # -------------------- OPTIMIZATION --------------------
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -------------------- LOGGING --------------------
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    os.makedirs(folder, exist_ok=True)
    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)

    # -------------------- INIT MODEL --------------------
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name
    )
    target_encoder = copy.deepcopy(encoder)

    # Freeze target encoder
    for p in target_encoder.parameters():
        p.requires_grad = False

    encoder.to(device)
    predictor.to(device)
    target_encoder.to(device)

    # -------------------- AMP SCALER --------------------
    scaler = torch.amp.GradScaler(enabled=(use_bfloat16 and torch.cuda.is_available()))

    # -------------------- TRANSFORMS & MASK COLLATOR --------------------
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep
    )
    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter
    )

    # -------------------- DATA LOADER --------------------
    _, unsupervised_loader, _ = make_tiny_imagenet200(
        transform=transform,
        batch_size=batch_size,
        collator=mask_collator,
        pin_mem=(pin_mem and torch.cuda.is_available()),
        training=True,
        num_workers=num_workers,
        world_size=1,
        rank=0,
        root_path=root_path,
        image_folder=image_folder,
        drop_last=True
    )
    ipe = len(unsupervised_loader)

    # -------------------- OPTIMIZER & SCHEDULER --------------------
    optimizer, _, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16
    )

    # -------------------- MOMENTUM SCHEDULE --------------------
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    # -------------------- LOGGING PATHS --------------------
    log_file = os.path.join(folder, f'{tag}_r0.csv')
    save_path = os.path.join(folder, f'{tag}-ep{{epoch}}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')

    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    start_epoch = 0
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=(os.path.join(folder, r_file) if r_file is not None else latest_path),
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler
        )
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    # -------------------- SAVE CHECKPOINT --------------------
    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'lr': lr
        }
        torch.save(save_dict, latest_path)
        if (epoch + 1) % checkpoint_freq == 0:
            torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -------------------- TRAINING LOOP --------------------
    for epoch in range(start_epoch, num_epochs):
        logger.info(f'Epoch {epoch + 1}')

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            imgs = udata[0].to(device, non_blocking=True)
            masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
            masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]

            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                with torch.no_grad():
                    h = target_encoder(imgs)
                    h = F.layer_norm(h, (h.size(-1),))
                    B = len(h)
                    h = apply_masks(h, masks_pred)
                    h = repeat_interleave_batch(h, B, repeat=len(masks_enc))

                # forward + loss
                z = encoder(imgs, masks_enc)
                z = predictor(z, masks_enc, masks_pred)
                loss = F.smooth_l1_loss(z, h)

                # --- AMP BACKWARD
                if use_bfloat16 and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # --- MOMENTUM UPDATE
                m = next(momentum_scheduler)
                with torch.no_grad():
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return float(loss), _new_lr, _new_wd, grad_stats

            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # --- LOGGING
            csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
            if itr % log_freq == 0 or np.isnan(loss) or np.isinf(loss):
                logger.info(f'[{epoch + 1}, {itr:5d}] loss: {loss_meter.avg:.3f} '
                            f'masks: {maskA_meter.avg:.1f} {maskB_meter.avg:.1f} '
                            f'[wd: {_new_wd:.2e}] [lr: {_new_lr:.2e}] '
                            f'[mem: {torch.cuda.max_memory_allocated()/1024**2:.2e}] '
                            f'({time_meter.avg:.1f} ms)')

                if grad_stats is not None:
                    logger.info(f'[{epoch + 1}, {itr:5d}] grad_stats: '
                                f'[{grad_stats.first_layer:.2e} {grad_stats.last_layer:.2e}] '
                                f'({grad_stats.min:.2e}, {grad_stats.max:.2e})')

            assert not np.isnan(loss), 'loss is nan'

        logger.info(f'avg. loss {loss_meter.avg:.3f}')
        save_checkpoint(epoch+1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--resume_preempt', action='store_true',
                        help='Resume from preempted checkpoint')
    args_cli = parser.parse_args()

    with open(args_cli.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config, resume_preempt=args_cli.resume_preempt)
