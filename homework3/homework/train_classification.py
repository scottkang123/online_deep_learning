import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from homework.models import ClassificationLoss, load_model, save_model
from homework.datasets.classification_dataset import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    transform_pipeline: str = "aug",
    num_epoch: int = 20,
    lr: float = 2e-4,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA/MPS not available, using CPU")
        device = torch.device("cpu")

    # reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # logging directory
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # load model
    model = load_model(model_name, **kwargs).to(device)
    model.train()

    # load data
    train_data = load_data(
        "classification_data/train",
        transform_pipeline=transform_pipeline,
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
    )

    val_data = load_data(
        "classification_data/val",
        transform_pipeline="default",
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
    )

    # loss and optimizer
    loss_func = ClassificationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    for epoch in range(num_epoch):
        metrics["train_acc"].clear()
        metrics["val_acc"].clear()

        model.train()
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            logits = model(img)
            loss = loss_func(logits, label)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            acc = (preds == label).float().mean().item()
            metrics["train_acc"].append(acc)

            logger.add_scalar("train/loss", loss.item(), global_step)
            logger.add_scalar("train/acc", acc, global_step)
            global_step += 1

        # evaluation
        with torch.inference_mode():
            model.eval()
            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                logits = model(img)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == label).float().mean().item()
                metrics["val_acc"].append(acc)

        epoch_train_acc = torch.tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.tensor(metrics["val_acc"]).mean()

        logger.add_scalar("epoch/train_acc", epoch_train_acc, epoch)
        logger.add_scalar("epoch/val_acc", epoch_val_acc, epoch)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epoch}: "
                f"train_acc={epoch_train_acc:.4f}, val_acc={epoch_val_acc:.4f}"
            )

    # save model for grader
    save_model(model)

    # also save checkpoint in log dir
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="classifier")
    parser.add_argument("--transform_pipeline", type=str, default="aug")
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)

    train(**vars(parser.parse_args()))
