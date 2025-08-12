"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred/target: (B, n_waypoints, 2)
    mask: (B, n_waypoints) bool
    """
    diff = (pred - target).abs()  # (B, n, 2)
    mask_ = mask.unsqueeze(-1).float()  # (B, n, 1)
    num = mask_.sum().clamp_min(1.0)
    return (diff * mask_).sum() / num

def masked_l1_weighted(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    w_lat: float = 1.0, w_lon: float = 2.0
) -> torch.Tensor:
    """
    Heavier weight on longitudinal (z) to improve speed accuracy.
    """
    mask_ = mask.unsqueeze(-1).float()      # (B, n, 1)
    diff = (pred - target).abs()            # (B, n, 2)
    weights = torch.tensor([w_lat, w_lon], device=pred.device, dtype=pred.dtype)
    diff = diff * weights                   # broadcast per-dim
    num = mask_.sum().clamp_min(1.0)
    return (diff * mask_).sum() / num

@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    """
    Longitudinal error: |z_pred - z_gt|
    Lateral error:      |x_pred - x_gt|
    (Coordinates are in ego BEV where dims are [x, z])
    """
    mask_ = mask.unsqueeze(-1)  # (B, n, 1)
    diff = (pred - target).abs() * mask_.float()
    denom = mask_.sum().clamp_min(1.0)

    lat = diff[..., 0].sum() / denom
    lon = diff[..., 1].sum() / denom
    return {"lat_err": lat.item(), "lon_err": lon.item()}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    print("CUDA/MPS not available, using CPU")
    return torch.device("cpu")

# Cosine with warmup
def build_scheduler(optimizer, num_epoch, warmup_epochs=5):
    main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epoch - warmup_epochs))
    return main, warmup_epochs

def lateral_smoothness(pred, mask, lam=0.05):
    d = pred[:, 1:, 0] - pred[:, :-1, 0]     # diffs of x across waypoints
    m = (mask[:, 1:] & mask[:, :-1]).float()
    num = m.sum().clamp_min(1.0)
    return lam * (d.abs() * m).sum() / num


# def boundary_penalty(pred, track_left, track_right, mask, lam=0.05):
#     """
#     Penalize |x - clamp(x, left_x, right_x)| so predicted x stays between boundaries.
#     pred: (B, n_wp, 2) ; track_*: (B, n_track, 2) ; mask: (B, n_wp)
#     We'll compare to mid anchors by indexing end/start, so we just use global min/max per sample for simplicity.
#     """
#     # global left/right x bounds per sample (loose but effective)
#     left_x  = track_left[..., 0].min(dim=1, keepdim=True).values    # (B,1)
#     right_x = track_right[..., 0].max(dim=1, keepdim=True).values   # (B,1)

#     x = pred[..., 0]                                                # (B, n_wp)
#     x_clamped = x.clamp(min=left_x, max=right_x)
#     m = mask.float()
#     num = m.sum().clamp_min(1.0)
#     return lam * ((x - x_clamped).abs() * m).sum() / num

def z_monotonicity(pred, mask, lam=0.05):
    # penalize negative forward motion: z[t+1] - z[t] should be >= 0
    dz = pred[:, 1:, 1] - pred[:, :-1, 1]     # (B, n_wp-1)
    m  = (mask[:, 1:] & mask[:, :-1]).float()
    num = m.sum().clamp_min(1.0)
    return lam * ((-dz).relu() * m).sum() / num

def boundary_penalty(pred, track_left, track_right, mask, lam=0.05):
    # keep x between left/right corridor (loose, but effective)
    left_x  = track_left[..., 0].min(dim=1, keepdim=True).values
    right_x = track_right[..., 0].max(dim=1, keepdim=True).values
    x = pred[..., 0]
    x_clamped = x.clamp(min=left_x, max=right_x)
    m = mask.float()
    num = m.sum().clamp_min(1.0)
    return lam * ((x - x_clamped).abs() * m).sum() / num

def boundary_penalty_anchor(pred, track_left, track_right, mask, lam=0.05):
    # mid & half-width for each track index
    mid   = 0.5 * (track_left + track_right)    # (B, n, 2)
    halfw = 0.5 * (track_right - track_left)    # (B, n, 2)

    B, n, _ = mid.shape
    n_wp = pred.size(1)

    # same indices you use in the model for anchors
    idx = torch.linspace(0, n-1, steps=n_wp, device=pred.device).round().long()

    mid_wp   = mid.index_select(1, idx)         # (B, n_wp, 2)
    half_wp  = halfw.index_select(1, idx)       # (B, n_wp, 2)

    left_x   = (mid_wp[..., 0] - half_wp[..., 0])
    right_x  = (mid_wp[..., 0] + half_wp[..., 0])

    x        = pred[..., 0]
    x_clamped = x.clamp(min=left_x, max=right_x)

    m   = mask.float()
    num = m.sum().clamp_min(1.0)
    return lam * ((x - x_clamped).abs() * m).sum() / num

def speed_match(pred, target, mask, lam=0.2):
    dz_p = pred[:,1:,1] - pred[:,:-1,1]
    dz_t = target[:,1:,1] - target[:,:-1,1]
    m = (mask[:,1:] & mask[:,:-1]).float()
    num = m.sum().clamp_min(1.0)
    return lam * (((dz_p - dz_t)**2) * m).sum() / num

def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    transform_pipeline: str = "state_only",
    num_epoch: int = 30,
    lr: float = 2e-3,
    batch_size: int = 128,
    weight_decay: float = 1e-4,
    seed: int = 2024,
    # model hyperparams passthrough (works for both planners)
    **model_kwargs,
):
    device = get_device()

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # load model
    model = load_model(model_name, **model_kwargs).to(device)
    model.train()

    if model_name == "cnn_planner":
      transform_pipeline = "default"

    # load data (state_only gives track_left/right + waypoints + mask)
    train_data = load_data(
        "drive_data/train",
        transform_pipeline=transform_pipeline,
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
    )
    val_data = load_data(
        "drive_data/val",
        transform_pipeline=("default" if model_name == "cnn_planner" else "state_only"),
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler, warmup_epochs = build_scheduler(optimizer, num_epoch)
    warmup_factor = 0.01  # start at 1% of lr

    global_step = 0

    for epoch in range(num_epoch):
        # -------- train --------
        model.train()
        for batch in train_data:
            optimizer.zero_grad(set_to_none=True)

            if model_name == "cnn_planner":
                imgs = batch["image"].to(device)                 # (B,3,96,128)
                waypoints      = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                pred = model(image=imgs)

                # losses: no boundary term required (you *can* add it since default has tracks)
                bound  = boundary_penalty_anchor(pred, batch["track_left"].to(device), batch["track_right"].to(device), waypoints_mask, lam=0.05)
                base   = masked_l1_weighted(pred, waypoints, waypoints_mask, w_lat=1.0, w_lon=6.0)
                smooth = lateral_smoothness(pred, waypoints_mask, lam=0.05)
                zmono  = z_monotonicity(pred, waypoints_mask, lam=0.05)
                smatch = speed_match(pred, waypoints, waypoints_mask, lam=0.3)
                loss   = base + smooth + zmono + smatch + bound

            else:
                track_left  = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints      = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                pred = model(track_left=track_left, track_right=track_right)

                is_transformer = hasattr(model, "decoder")
                if is_transformer:
                    base   = masked_l1_weighted(pred, waypoints, waypoints_mask, w_lat=1.0, w_lon=3.5)
                    smooth = lateral_smoothness(pred, waypoints_mask, lam=0.05)
                    bound  = boundary_penalty_anchor(pred, track_left, track_right, waypoints_mask, lam=0.05)
                    zmono  = z_monotonicity(pred, waypoints_mask, lam=0.10)
                    loss   = base + smooth + bound + zmono
                else:
                    loss = masked_l1_weighted(pred, waypoints, waypoints_mask, w_lat=1.0, w_lon=2.0)

            if epoch < warmup_epochs:
              warmup_lr = lr * (warmup_factor + (1 - warmup_factor) * (epoch + 1) / warmup_epochs)
              for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # metrics on-the-fly
            metrics = compute_metrics(pred.detach(), waypoints, waypoints_mask)
            logger.add_scalar("train/loss", loss.item(), global_step)
            logger.add_scalar("train/lat_err", metrics["lat_err"], global_step)
            logger.add_scalar("train/lon_err", metrics["lon_err"], global_step)
            logger.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

        scheduler.step()

        # -------- eval --------
        model.eval()
        eval_losses = []
        eval_lat, eval_lon = [], []
        with torch.inference_mode():
            for batch in val_data:
                if model_name == "cnn_planner":
                    imgs = batch["image"].to(device)
                    waypoints      = batch["waypoints"].to(device)
                    waypoints_mask = batch["waypoints_mask"].to(device)

                    pred = model(image=imgs)
                    bound  = boundary_penalty_anchor(pred, batch["track_left"].to(device), batch["track_right"].to(device), waypoints_mask, lam=0.05)
                    base   = masked_l1_weighted(pred, waypoints, waypoints_mask, w_lat=1.0, w_lon=6.0)
                    smooth = lateral_smoothness(pred, waypoints_mask, lam=0.05)
                    zmono  = z_monotonicity(pred, waypoints_mask, lam=0.05)
                    smatch = speed_match(pred, waypoints, waypoints_mask, lam=0.3)
                    loss   = base + smooth + zmono + smatch + bound

                else:
                    track_left  = batch["track_left"].to(device)
                    track_right = batch["track_right"].to(device)
                    waypoints      = batch["waypoints"].to(device)
                    waypoints_mask = batch["waypoints_mask"].to(device)

                    pred = model(track_left=track_left, track_right=track_right)

                    is_transformer = hasattr(model, "decoder")
                    if is_transformer:
                        base   = masked_l1_weighted(pred, waypoints, waypoints_mask, w_lat=1.0, w_lon=3.5)
                        smooth = lateral_smoothness(pred, waypoints_mask, lam=0.05)
                        bound  = boundary_penalty_anchor(pred, track_left, track_right, waypoints_mask, lam=0.05)
                        zmono  = z_monotonicity(pred, waypoints_mask, lam=0.10)
                        loss   = base + smooth + bound + zmono
                    else:
                        loss = masked_l1_weighted(pred, waypoints, waypoints_mask, w_lat=1.0, w_lon=2.0)

                ms = compute_metrics(pred, waypoints, waypoints_mask)
                eval_losses.append(loss.item())
                eval_lat.append(ms["lat_err"])
                eval_lon.append(ms["lon_err"])

        mean_loss = float(np.mean(eval_losses)) if eval_losses else 0.0
        mean_lat = float(np.mean(eval_lat)) if eval_lat else 0.0
        mean_lon = float(np.mean(eval_lon)) if eval_lon else 0.0

        logger.add_scalar("val/loss", mean_loss, epoch)
        logger.add_scalar("val/lat_err", mean_lat, epoch)
        logger.add_scalar("val/lon_err", mean_lon, epoch)

        print(
            f"Epoch {epoch+1}/{num_epoch} | "
            f"val_loss={mean_loss:.4f} | lat_err={mean_lat:.4f} | lon_err={mean_lon:.4f}"
        )

    # save model for grader
    path = save_model(model)
    torch.save(model.state_dict(), Path(log_dir) / f"{model_name}.th")
    print(f"Saved: {path} and {Path(log_dir) / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="cnn_planner", choices=["mlp_planner", "transformer_planner", "cnn_planner"])    
    parser.add_argument("--transform_pipeline", type=str, default="state_only")
    parser.add_argument("--num_epoch", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2024)

    # MLP hyperparams
    parser.add_argument("--hidden_dims", type=int, nargs="*", default=[256, 256, 256])
    parser.add_argument("--dropout", type=float, default=0.0)

    # Transformer hyperparams
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--tf_dropout", type=float, default=0.0)

    args = parser.parse_args()

    # Route model-specific kwargs cleanly
    model_kwargs = {}
    if args.model_name == "mlp_planner":
        model_kwargs = {
            "hidden_dims": tuple(args.hidden_dims),
            "dropout": args.dropout,
        }
    elif args.model_name == "transformer_planner":
        model_kwargs = {
            "d_model": args.d_model,
            "n_waypoints": 3,
            "n_track": 10,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.tf_dropout,
        }

    train(
        exp_dir=args.exp_dir,
        model_name=args.model_name,
        transform_pipeline=args.transform_pipeline,
        num_epoch=args.num_epoch,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        seed=args.seed,
        **model_kwargs,
    )
