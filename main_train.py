import argparse
import os
import random
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from utils.config import load_config
from data.dataset_loader import load_dataset, summarize_hetero_data
from models.han import HANNodeClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train HAN for MACF on a specified config.")
    parser.add_argument("--config", type=str, default="configs/acm.yaml", help="Path to yaml config")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    correct = (pred == y).sum().item()
    total = y.numel()
    return 0.0 if total == 0 else correct / total


@torch.no_grad()
def evaluate(model, data, target_node_type, mask_name: str):
    model.eval()
    logits = model(data)
    y = data[target_node_type].y.to(logits.device)
    mask = getattr(data[target_node_type], mask_name).to(logits.device)
    logits_masked = logits[mask]
    y_masked = y[mask]
    loss = F.cross_entropy(logits_masked, y_masked).item()
    acc = accuracy(logits_masked, y_masked)
    return loss, acc


def main():
    args = parse_args()
    print("=" * 80)
    print("MACF Step-2: HAN Training")
    print("=" * 80)
    cfg = load_config(args.config)

    set_seed(cfg["seed"])
    device_str = cfg.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA is not available, fallback to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"[INFO] using device: {device}")

    loaded = load_dataset(cfg)
    summarize_hetero_data(loaded)

    data = loaded.data
    target_node_type = loaded.target_node_type
    num_classes = loaded.num_classes

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    model = HANNodeClassifier(
        data=data,
        target_node_type=target_node_type,
        hidden_dim=model_cfg["hidden_dim"],
        out_dim=num_classes,
        heads=model_cfg.get("heads", 8),
        dropout=model_cfg.get("dropout", 0.5),
    ).to(device)

    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    y = data[target_node_type].y
    train_mask = data[target_node_type].train_mask

    epochs = train_cfg["epochs"]
    best_val_acc = -1.0
    best_train_acc = -1.0
    best_test_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits[train_mask], y[train_mask])
        val_loss, val_acc = evaluate(model, data, target_node_type, "val_mask")
        test_loss, test_acc = evaluate(model, data, target_node_type, "test_mask")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_acc = train_acc
            best_test_acc = test_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "target_node_type": target_node_type,
                "num_classes": num_classes,
                "metadata": loaded.data.metadata(),
            }

        print(
            f"Epoch {epoch:03d} | train_loss={loss.item():.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/han_{loaded.dataset_name}.pt"
    if best_state is not None:
        torch.save(best_state, ckpt_path)
        print("\n" + "=" * 80)
        print("[OK] Training finished.")
        print(f"[OK] Best tarin_acc = {best_train_acc:.4f}, Best val_acc = {best_val_acc:.4f}, Best test_acc = {best_test_acc: .4f}")
        print(f"[OK] Checkpoint saved to: {ckpt_path}")
        print("=" * 80)
    else:
        print("[WARN] No best state was saved.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR] main_train.py failed.")
        print(f"[ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
