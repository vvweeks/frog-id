"""
scripts/run_training.py - Entry point for model training + plotting +
results logging.
Run explicitly:
  python -m scripts.run_training --model resnet --nickname "v6-dropout50" --note "Bumped dropout to 0.5"
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

EARLY_STOP_PATIENCE = 5


def get_git_commit_hash():
    """Short hash of the current commit, so every saved run is traceable
    back to the exact code that produced it. Returns 'nogit' if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


def get_git_commit_message():
    """Subject line of the current commit - used as the default 'what's
    different about this run' note when --note isn't given, since a
    commit message should already describe the one tweak it contains."""
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--pretty=%s"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


def make_run_id(model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{timestamp}_{get_git_commit_hash()}"


def save_results(run_id, model_name, train_losses, val_losses, val_accuracies, final_test_acc,
                  hyperparams=None, checkpoint_path=None, nickname=None, note=None,
                  early_stopped=False):
    """Saves a full-metrics JSON (includes test accuracy, for the record),
    and appends a row to a master CSV log for cross-run comparison. The
    CSV deliberately omits test accuracy - it's what you'd end up
    optimizing against by eye if it were sitting next to validation
    accuracy on every experiment, which defeats the point of a held-out
    test set."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    nickname = nickname or run_id
    note = note or ""

    # --- Full per-run metrics (every epoch) ---
    metrics_path = os.path.join(RESULTS_DIR, f"{run_id}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            "run_id": run_id,
            "nickname": nickname,
            "model_name": model_name,
            "note": note,
            "checkpoint_path": checkpoint_path,
            "hyperparams": hyperparams or {},
            "early_stopped": early_stopped,
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
            writer.writerow(["run_id", "nickname", "model_name", "note", "best_val_accuracy",
                              "epochs_run", "early_stopped", "hyperparams", "checkpoint_path"])
        writer.writerow([run_id, nickname, model_name, note, f"{max(val_accuracies):.2f}",
                          len(train_losses), early_stopped, json.dumps(hyperparams or {}),
                          checkpoint_path])
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
    parser.add_argument("--nickname", default=None,
                        help="Short human-friendly label for this run (defaults to run_id).")
    parser.add_argument("--note", default=None,
                        help="What's different about this run (defaults to the latest git commit message).")
    parser.add_argument("--head", choices=["mlp", "linear"], default="mlp",
                        help="BirdNET classifier head: mlp (default) or a linear probe.")
    args = parser.parse_args()
    note = args.note or get_git_commit_message()

    if args.model == "resnet":
        from config import EPOCHS, LEARNING_RATE, BACKBONE_LR_MULT, WEIGHT_DECAY
        from train_resnet import train_model

        run_id = make_run_id("resnet18")
        train_losses, val_losses, val_accuracies, final_test_acc, checkpoint_path, early_stopped = (
            train_model(run_id=run_id, early_stop_patience=EARLY_STOP_PATIENCE)
        )
        hyperparams = {"epochs": EPOCHS, "lr": LEARNING_RATE, "backbone_lr_mult": BACKBONE_LR_MULT,
                       "weight_decay": WEIGHT_DECAY, "early_stop_patience": EARLY_STOP_PATIENCE}
        save_results(run_id, "resnet18", train_losses, val_losses, val_accuracies, final_test_acc,
                     hyperparams, checkpoint_path, args.nickname, note, early_stopped)
        plot_results(train_losses, val_losses, val_accuracies, final_test_acc, "ResNet18", run_id)
    else:
        from train_birdnet import train_birdnet_classifier

        birdnet_epochs, birdnet_lr = 50, 1e-3
        run_id = make_run_id("birdnet_head")
        train_losses, val_losses, val_accuracies, final_test_acc, checkpoint_path, early_stopped = (
            train_birdnet_classifier(epochs=birdnet_epochs, lr=birdnet_lr, run_id=run_id,
                                      early_stop_patience=EARLY_STOP_PATIENCE, head_type=args.head)
        )
        hyperparams = {"epochs": birdnet_epochs, "lr": birdnet_lr,
                       "early_stop_patience": EARLY_STOP_PATIENCE, "head_type": args.head}
        save_results(run_id, "birdnet_head", train_losses, val_losses, val_accuracies, final_test_acc,
                     hyperparams, checkpoint_path, args.nickname, note, early_stopped)
        plot_results(train_losses, val_losses, val_accuracies, final_test_acc, "BirdNET-Embedding", run_id)
