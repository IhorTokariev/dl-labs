"""Рисунки без графічного інтерфейсу; підписи українською."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from common import CLASSES, MEAN, STD, image_path

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def batch_figure(images, targets, path):
    restored = (images[:16].cpu().permute(0, 2, 3, 1).numpy() * np.array(STD) + np.array(MEAN)).clip(0, 1)
    fig, axes = plt.subplots(4, 4, figsize=(9, 9), layout="constrained")
    for i, ax in enumerate(axes.flat):
        ax.imshow(restored[i])
        ax.set_title(CLASSES[int(targets[i])])
        ax.axis("off")
    fig.suptitle("Навчальний батч після аугментацій (обернене нормування)")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def curves_figure(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    colors = {"scratch": "#9b354c", "frozen": "#286c9d", "unfreeze": "#268257"}
    styles = {1: "-", 2: "--", 3: ":"}
    for run in runs:
        x = [r["epoch"] for r in run["history"]]
        for ax, key in zip(axes, ["train_loss", "val_macro_f1"]):
            ax.plot(x, [r[key] for r in run["history"]], color=colors[run["regime"]],
                    linestyle=styles[run["seed"]], marker="o", markersize=3,
                    label=f'{run["regime"]}, seed={run["seed"]}')
    for ax in axes:
        ax.set_xlabel("Епоха")
        ax.set_xticks(range(1, len(runs[0]["history"])+1))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Середня втрата train")
    axes[1].set_ylabel("Macro-F1 validation")
    axes[1].legend(fontsize=8)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def confusion_figure(cm, path):
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    plot = ax.imshow(cm, cmap="Blues")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "#17212b")
    ax.set_xticks(range(6), CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(6), CLASSES)
    ax.set_xlabel("Передбачений клас")
    ax.set_ylabel("Справжній клас")
    ax.set_title("Матриця плутанини test (CPU)")
    fig.colorbar(plot, ax=ax, shrink=0.8, label="Кількість зображень")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def errors_figure(errors, root, path):
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.2), layout="constrained")
    for ax, error in zip(axes, errors):
        row = {"file": error["file"], "label": error["true_class"], "part": "test"}
        with Image.open(image_path(root, row)) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f'{error["file"]}\n{error["true_class"]} → {error["predicted_class"]}', fontsize=10)
        ax.set_xlabel(f'p={error["probability"]:.4f}; таких помилок: {error["pair_count"]}')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("П’ять характерних помилок за частотою пар класів")
    fig.savefig(path, dpi=180)
    plt.close(fig)
