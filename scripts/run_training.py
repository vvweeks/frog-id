"""
scripts/run_training.py - Entry point for model training + plotting +
results logging.
Run explicitly:
  python -m scripts.run_training --model resnet
  python -m scripts.run_training --model birdnet
"""
import argparse
import csv
import json
import os
import subprocess
from datetime import datetime

import matplotlib.pyplot as plt

from config import RESULTS_DIR


def get_git_commit_hash():
    """Short hash of the current commit, so every saved run is traceable
    back to the exact code that produced it. Returns 'nogit' if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


def make_run_id(model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{timestamp}_{get_git_commit_hash()}"


def save_results(run_id, model_name, train_losses, val_losses, val_accuracies, final_test_acc,
                  hyperparams=None, checkpoint_path=None):
    """Saves a full-metrics JSON, and appends a summary row to a master
    CSV log for cross-run comparison. Both are keyed by run_id, which
    also ties back to the saved checkpoint and the git commit it came from."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Full per-run metrics (every epoch) ---
    metrics_path = os.path.join(RESULTS_DIR, f"{run_id}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            "run_id": run_id,
            "model_name": model_name,
            "checkpoint_path": checkpoint_path,
            "hyperparams": hyperparams or {},
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "best_val_accuracy": max(val_accuracies),
            "final_test_accuracy": final_test_acc,
            "epochs_run": len(train_losses),
        }, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    # --- Master CSV log: one row per run, easy to compare across runs ---
    log_path = os.path.join(RESULTS_DIR, "results_log.csv")
    log_exists = os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["run_id", "model_name", "epochs_run",
                              "best_val_accuracy", "final_test_accuracy", "checkpoint_path"])
        writer.writerow([run_id, model_name, len(train_losses),
                          f"{max(val_accuracies):.2f}", f"{final_test_acc:.2f}", checkpoint_path])
    print(f"Appended summary row to: {log_path}")

    return metrics_path


def plot_results(train_losses, val_losses, val_accuracies, final_test_acc, title_prefix, run_id):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(train_losses, label="Training Loss")
    axes[0].plot(val_losses, label="Validation Loss", linestyle="--")
    axes[0].set_title(f"{title_prefix}: Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(val_accuracies, color='green', label="Validation Acc")
    axes[1].axhline(y=final_test_acc, color='red', linestyle=':',
                     label=f"Final Test Acc ({final_test_acc:.1f}%)")
    axes[1].set_title(f"{title_prefix}: Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy %")
    axes[1].legend()
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, f"{run_id}_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")

    plt.show()

    print(f"\n=== Summary ({title_prefix}) ===")
    print(f"Best Validation Accuracy: {max(val_accuracies):.2f}%")
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet", "birdnet"], required=True)
    args = parser.parse_args()

    if args.model == "resnet":
        from config import EPOCHS, LEARNING_RATE, BACKBONE_LR_MULT, WEIGHT_DECAY
        from train_resnet import train_model

        run_id = make_run_id("resnet18")
        train_losses, val_losses, val_accuracies, final_test_acc, checkpoint_path = train_model(run_id=run_id)
        hyperparams = {"epochs": EPOCHS, "lr": LEARNING_RATE,
                       "backbone_lr_mult": BACKBONE_LR_MULT, "weight_decay": WEIGHT_DECAY}
        save_results(run_id, "resnet18", train_losses, val_losses, val_accuracies,
                     final_test_acc, hyperparams, checkpoint_path)
        plot_results(train_losses, val_losses, val_accuracies, final_test_acc, "ResNet18", run_id)
    else:
        from train_birdnet import train_birdnet_classifier

        run_id = make_run_id("birdnet_head")
        train_losses, val_losses, val_accuracies, final_test_acc, checkpoint_path = train_birdnet_classifier(run_id=run_id)
        save_results(run_id, "birdnet_head", train_losses, val_losses, val_accuracies,
                     final_test_acc, checkpoint_path=checkpoint_path)
        plot_results(train_losses, val_losses, val_accuracies, final_test_acc, "BirdNET-Embedding", run_id)
