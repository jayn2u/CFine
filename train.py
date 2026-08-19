import os
import shutil
import time
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import random
import numpy as np
from utils.metric import AverageMeter, Loss
from config import data_config, network_config, get_image_unique
from train_config import config
from solver import WarmupMultiStepLR
from test import test
from utils.CRLoss import CRLoss
from utils.efficiency import (
    build_epoch_efficiency_metrics,
    finish_cuda_timer,
    get_peak_vram_metrics,
    start_measurement,
)
from utils.wandb_tracking import (
    finish_train_run,
    log_train_epoch_metrics,
    log_val_metrics,
    start_train_run,
)
from utils.training import (
    autocast_context,
    build_ema_model,
    build_grad_scaler,
    optimizer_step,
    unwrap_model,
    validate_training_options,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def set_seed(args):
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_checkpoint(state, epoch, dst, is_best):
    filename = os.path.join(dst, "best_model") + ".pth.tar"
    torch.save(state, filename)
    if is_best:
        dst_best = os.path.join(dst, "model_best", str(epoch)) + ".pth.tar"
        shutil.copyfile(filename, dst_best)


def train(
    epoch,
    train_loader,
    network,
    optimizer,
    compute_loss,
    cr_loss_fun,
    args,
    scaler,
    ema_model=None,
):
    meters = {
        "loss": AverageMeter(),
        "cmpm_loss": AverageMeter(),
        "cmpc_loss": AverageMeter(),
        "sim_loss": AverageMeter(),
        "image_acc": AverageMeter(),
        "text_acc": AverageMeter(),
    }

    # switch to train mode
    network.train()
    raw_network = unwrap_model(network)
    amp_enabled = getattr(args, "amp", False)
    amp_dtype = getattr(args, "amp_dtype", "fp16")
    global_step = 0

    start_time = time.time()
    for step, (images, captions, labels) in enumerate(train_loader):
        (
            tokens,
            segments,
            input_masks,
            caption_length,
        ) = raw_network.language_model.pre_process(captions)
        tokens = tokens.cuda()
        segments = segments.cuda()
        input_masks = input_masks.cuda()
        images = images.cuda()
        labels = labels.cuda()
        with autocast_context(amp_enabled, amp_dtype):
            img_output, text_output, img_f, text_f, sim_cs, sim_cd = network(
                images, tokens, segments, input_masks
            )
            (
                cmpm_loss,
                cmpc_loss,
                loss,
                image_precision,
                text_precision,
                pos_avg_sim,
                neg_arg_sim,
            ) = compute_loss(
                img_output,
                text_output,
                img_f,
                text_f,
                labels,
                args.lambda_diversity,
            )

            sim = sim_cs + sim_cd
            sim_loss = cr_loss_fun(sim, labels, semi=False)
            loss = loss + sim_loss * 10

        current_lr = []
        for params in optimizer.param_groups:
            current_lr.append(params['lr'])

        global_step += 1
        if global_step % 50 == 0:
            logger.info(
                "Epoch: {}/{}, Step: {}/{}, Lr: {}, cmpm_loss: {:.3f}, cmpc_loss: {:.3f}, sim_loss: {:.3f}, Time/step: {:.4f}".format(
                    epoch,
                    args.num_epoches,
                    step+1,
                    len(train_loader),
                    "-".join([str('%.9f'%itm) for itm in sorted(current_lr)]),
                    cmpm_loss,
                    cmpc_loss,
                    sim_loss,
                    (time.time() - start_time) / 50
                )
            )
            start_time = time.time()

        # compute gradient and do ADAM step
        optimizer.zero_grad()
        step_succeeded = optimizer_step(loss, optimizer, scaler)
        if step_succeeded and ema_model is not None:
            ema_model.update_parameters(raw_network)

        batch_size = images.shape[0]
        meters["loss"].update(loss.item(), batch_size)
        meters["cmpm_loss"].update(cmpm_loss.item(), batch_size)
        meters["cmpc_loss"].update(cmpc_loss.item(), batch_size)
        meters["sim_loss"].update(sim_loss.item(), batch_size)
        meters["image_acc"].update(image_precision, batch_size)
        meters["text_acc"].update(text_precision, batch_size)

    return meters


def main(args, wandb_session=None):

    set_seed(args)
    validate_training_options(args)

    # transform
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            normalize,
        ]
    )
    cap_transform = None
    # data
    train_loader = data_config(
        args.image_dir,
        args.anno_dir,
        args.batch_size,
        "train",
        100,
        train_transform,
        cap_transform=cap_transform,
    )

    test_loader = data_config(
        args.image_dir, args.anno_dir, 64, "test", 100, test_transform
    )
    unique_image = get_image_unique(
        args.image_dir, args.anno_dir, 64, "test", 100, test_transform
    )

    # loss
    compute_loss = Loss(args)
    cr_loss_fun = CRLoss(args)
    nn.DataParallel(compute_loss).cuda()

    # network
    network, optimizer = network_config(
        args, "train", compute_loss.parameters(), args.resume, args.model_path
    )
    amp_enabled = getattr(args, "amp", False)
    amp_dtype = getattr(args, "amp_dtype", "fp16")
    scaler = build_grad_scaler(amp_enabled, amp_dtype)
    ema_model = (
        build_ema_model(network, getattr(args, "ema_decay", 0.999))
        if getattr(args, "ema", False)
        else None
    )
    if args.resume:
        checkpoint = torch.load(args.model_path, map_location="cpu")
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        if ema_model is not None and "ema_model" in checkpoint:
            ema_model.load_state_dict(checkpoint["ema_model"])
    # lr_scheduler
    scheduler = WarmupMultiStepLR(optimizer, (20, 25, 35), 0.1, 0.01, 10, "linear")  # (20, 25, 35)
    ac_t2i_top1_best = 0.0
    best_epoch = 0
    session = wandb_session if wandb_session is not None else start_train_run(args)
    device = torch.device("cuda")
    cumulative_gpu_seconds = 0.0

    try:
        for epoch in range(1, args.num_epoches + 1 - args.start_epoch):
            current_epoch = args.start_epoch + epoch
            network.train()
            train_started_at = start_measurement(device)
            meters = train(
                current_epoch,
                train_loader,
                network,
                optimizer,
                compute_loss,
                cr_loss_fun,
                args,
                scaler,
                ema_model,
            )
            train_seconds = finish_cuda_timer(device, train_started_at)
            train_vram = get_peak_vram_metrics(device)
            cumulative_gpu_seconds += train_seconds
            train_efficiency = build_epoch_efficiency_metrics(
                train_seconds,
                meters["loss"].count,
                cumulative_gpu_seconds,
            )
            log_train_epoch_metrics(
                session,
                current_epoch,
                meters,
                optimizer.param_groups[0]["lr"],
                efficiency_metrics=train_efficiency,
                vram_metrics=train_vram,
            )

            logging.info(
                "Epoch {}/{} Finished, train_loss: {:.3f}, image_precision: {:.3f}, text_precision: {:.3f}".format(
                    current_epoch,
                    args.num_epoches,
                    meters["loss"].avg,
                    meters["image_acc"].avg,
                    meters["text_acc"].avg,
                )
            )
            scheduler.step()

            val_started_at = start_measurement(device)
            eval_network = ema_model.module if ema_model is not None else network
            metrics = test(
                test_loader,
                eval_network,
                args,
                unique_image,
                epoch,
                return_metrics=True,
            )
            val_seconds = finish_cuda_timer(device, val_started_at)
            val_vram = get_peak_vram_metrics(device)
            log_val_metrics(
                session,
                current_epoch,
                metrics,
                efficiency_metrics={"epoch_seconds": val_seconds},
                vram_metrics=val_vram,
            )

            state = {
                "network": network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "W": compute_loss.W,
                "epoch": current_epoch,
                "scaler": scaler.state_dict(),
            }
            if ema_model is not None:
                state["ema_model"] = ema_model.state_dict()

            if metrics["t2i_R1"] > ac_t2i_top1_best:
                best_epoch = current_epoch
                ac_t2i_top1_best = metrics["t2i_R1"]
                save_checkpoint(state, current_epoch, args.checkpoint_dir, False)

            logging.info("Text-to-Image:")
            logging.info(
                " R@1: {:.2f}, R@5: {:.2f}, R@10: {:.2f}".format(
                    metrics["t2i_R1"],
                    metrics["t2i_R5"],
                    metrics["t2i_R10"],
                )
            )
            logging.info("Image-to-Text:")
            logging.info(
                " R@1: {:.2f}, R@5: {:.2f}, R@10: {:.2f}".format(
                    metrics["i2t_R1"],
                    metrics["i2t_R5"],
                    metrics["i2t_R10"],
                )
            )

        logging.info("Train Finished!")
        logging.info(
            "The best epoch:{}, the R@1 is: {:.2f}".format(
                best_epoch,
                ac_t2i_top1_best,
            )
        )
        logging.info(args.checkpoint_dir)
        finish_train_run(
            session,
            ac_t2i_top1_best,
            best_epoch,
            args.checkpoint_dir,
        )
        return ac_t2i_top1_best, best_epoch
    finally:
        session.finish()


if __name__ == "__main__":
    args = config()
    main(args)
