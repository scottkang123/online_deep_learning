import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
from torch.nn import functional as F

from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import DetectionMetric


def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    transform_pipeline: str = "default",
    num_epoch: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 42,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs).to(device)
    model.train()

    train_data = load_data("drive_data/train", transform_pipeline=transform_pipeline, shuffle=True, batch_size=batch_size)
    val_data = load_data("drive_data/val", transform_pipeline="default", shuffle=False, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    global_step = 0

    for epoch in range(num_epoch):
        model.train()
        train_metrics = DetectionMetric()
        train_metrics.reset()

        for batch in train_data:
            img = batch["image"].to(device)
            seg = batch["track"].to(device)
            depth = batch["depth"].to(device)

            optimizer.zero_grad()
            logits, pred_depth = model(img)

            loss_seg = F.cross_entropy(logits, seg)
            loss_depth = F.l1_loss(pred_depth, depth)
            loss = loss_seg * 1.5 + loss_depth

            loss.backward()
            optimizer.step()

            pred_seg = logits.argmax(dim=1)
            train_metrics.add(pred_seg, seg, pred_depth, depth)

            logger.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        train_results = train_metrics.compute()
        logger.add_scalar("train/iou", train_results["iou"], epoch)
        logger.add_scalar("train/abs_depth_error", train_results["abs_depth_error"], epoch)
        logger.add_scalar("train/tp_depth_error", train_results["tp_depth_error"], epoch)

        # --- Validation ---
        model.eval()
        val_metrics = DetectionMetric()
        val_metrics.reset()

        with torch.inference_mode():
            for batch in val_data:
                img = batch["image"].to(device)
                seg = batch["track"].to(device)
                depth = batch["depth"].to(device)

                logits, pred_depth = model(img)
                pred_seg = logits.argmax(dim=1)

                val_metrics.add(pred_seg, seg, pred_depth, depth)

        val_results = val_metrics.compute()
        logger.add_scalar("val/iou", val_results["iou"], epoch)
        logger.add_scalar("val/abs_depth_error", val_results["abs_depth_error"], epoch)
        logger.add_scalar("val/tp_depth_error", val_results["tp_depth_error"], epoch)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{num_epoch}: "
                  f"val_iou={val_results['iou']:.4f}, "
                  f"val_abs_depth_error={val_results['abs_depth_error']:.4f}, "
                  f"val_tp_depth_error={val_results['tp_depth_error']:.4f}")

    save_model(model)
    print(f"Model saved to {save_model(model)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="detector")
    parser.add_argument("--transform_pipeline", type=str, default="default")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    train(**vars(parser.parse_args()))