"""
scripts/run_training.py - Entry point for model training + plotting.
Run explicitly:
  python -m scripts.run_training --model resnet
  python -m scripts.run_training --model birdnet
"""
import argparse
import matplotlib.pyplot as plt


def plot_results(train_losses, val_losses, val_accuracies, final_test_acc, title_prefix):
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
    plt.show()

    print(f"\n=== Summary ({title_prefix}) ===")
    print(f"Best Validation Accuracy: {max(val_accuracies):.2f}%")
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet", "birdnet"], required=True)
    args = parser.parse_args()

    if args.model == "resnet":
        from train_resnet import train_model
        train_losses, val_losses, val_accuracies, final_test_acc = train_model()
        plot_results(train_losses, val_losses, val_accuracies, final_test_acc, "ResNet18")
    else:
        from train_birdnet import train_birdnet_classifier
        train_losses, val_losses, val_accuracies, final_test_acc = train_birdnet_classifier()
        plot_results(train_losses, val_losses, val_accuracies, final_test_acc, "BirdNET-Embedding")
