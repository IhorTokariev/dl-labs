"""Спільний протокол ЛР3: дані, моделі, журнали та вибір за validation."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

BASE = Path(__file__).resolve().parent
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
PARTS = ["train", "validation", "test"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMG_SIZE = 160
RUNS = [("scratch", 1), ("frozen", 1), ("frozen", 2), ("frozen", 3),
        ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3)]
MODULES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
EVAL_MODULES = {"scratch": [], "frozen": MODULES[:-1], "unfreeze": MODULES[:-2]}
TRAINABLE = {"scratch": 11179590, "frozen": 3078, "unfreeze": 8396806}
CONTROL = {
    "buildings": [(200, 1996761), (218, 4837323), (219, 4837559)],
    "forest": [(200, 2032625), (237, 5276962), (237, 5291326)],
    "glacier": [(200, 2183923), (276, 6135652), (277, 6157250)],
    "mountain": [(200, 1950345), (262, 5789535), (263, 5849765)],
    "sea": [(200, 1987613), (255, 5655068), (255, 5668011)],
    "street": [(200, 1841817), (250, 5541845), (251, 5547533)],
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def source_hash():
    h = hashlib.sha256()
    for name in ("common.py", "checks.py", "plots.py", "train.py"):
        h.update(name.encode())
        h.update((BASE / name).read_bytes())
    return h.hexdigest()


def build_split(data_root, split_path, train_per_class=200):
    """Два незалежні генератори, кожен послідовно обслуговує шість класів."""
    root = Path(data_root)
    records = []
    rng = np.random.default_rng(0)
    for label in CLASSES:
        names = sorted(p.name for p in (root / "seg_train" / "seg_train" / label).glob("*.jpg"))
        assert len(names) >= train_per_class, (label, len(names))
        idx = np.arange(len(names))
        rng.shuffle(idx)
        records.extend({"file": names[i], "label": label, "part": "train"}
                       for i in idx[:train_per_class])
    rng = np.random.default_rng(0)
    for label in CLASSES:
        names = sorted(p.name for p in (root / "seg_test" / "seg_test" / label).glob("*.jpg"))
        assert len(names) >= 2
        idx = np.arange(len(names))
        rng.shuffle(idx)
        cut = len(names) // 2
        for part, indices in (("validation", idx[:cut]), ("test", idx[cut:])):
            records.extend({"file": names[i], "label": label, "part": part} for i in indices)
    # Існуючий поділ не перезаписуємо мовчки при --resume або зміні набору.
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["file", "label", "part"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    content = stream.getvalue()
    split_path = Path(split_path)
    if split_path.exists():
        assert split_path.read_text() == content, "Існуючий split.csv не відповідає процедурі"
    else:
        split_path.write_text(content)
    return load_split(split_path)


def load_split(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["file", "label", "part"]
        rows = list(reader)
    for row in rows:
        assert row["label"] in CLASSES and row["part"] in PARTS
        assert Path(row["file"]).name == row["file"] and row["file"].endswith(".jpg")
        int(Path(row["file"]).stem)
    return rows


def check_split(rows, strict=True):
    names = [r["file"] for r in rows]
    assert len(names) == len(set(names)), "Повтор або перетин файлів у split.csv"
    result = []
    print("Контроль поділу: клас | train | validation | test | збіг", flush=True)
    for label in CLASSES:
        values = []
        for i, part in enumerate(PARTS):
            selected = [r for r in rows if r["label"] == label and r["part"] == part]
            count, total = len(selected), sum(int(Path(r["file"]).stem) for r in selected)
            match = (count, total) == CONTROL[label][i] if strict else None
            values.append({"part": part, "count": count, "sum": total, "match": match})
            if strict:
                assert match, (label, part, count, total, CONTROL[label][i])
        result.append({"class": label, "parts": values})
        print(label, " | ", " | ".join(f'{v["count"]} / {v["sum"]}' for v in values),
              " | ", "так" if strict else "smoke: звірку вимкнено", flush=True)
    totals = [{"part": p, "count": sum(x["parts"][i]["count"] for x in result),
               "sum": sum(x["parts"][i]["sum"] for x in result)} for i, p in enumerate(PARTS)]
    if strict:
        assert [(x["count"], x["sum"]) for x in totals] == [
            (1200, 11993084), (1498, 33236385), (1502, 33351444)]
    print("Разом:", totals, "; перетинів немає", flush=True)
    return {"table": result, "totals": totals, "no_overlap": True, "strict": strict}


def transform(train):
    ops = [transforms.RandomResizedCrop(160, scale=(0.7, 1.0)),
           transforms.RandomHorizontalFlip()] if train else [transforms.Resize((160, 160))]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def image_path(root, row):
    directory = "seg_train" if row["part"] == "train" else "seg_test"
    return Path(root) / directory / directory / row["label"] / row["file"]


class SplitDataset(Dataset):
    def __init__(self, root, split_path, part):
        self.root = Path(root)
        self.rows = [r for r in load_split(split_path) if r["part"] == part]
        self.transform = transform(part == "train")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(image_path(self.root, row)) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, CLASSES.index(row["label"])


def loader(root, split_path, part):
    return DataLoader(SplitDataset(root, split_path, part), batch_size=32 if part == "train" else 64,
                      shuffle=part == "train", drop_last=False, num_workers=0)


def choose_device(requested):
    available = {"mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available(), "cpu": True}
    name = next(x for x in ("mps", "cuda", "cpu") if available[x]) if requested == "auto" else requested
    if not available[name]:
        raise RuntimeError(f"Пристрій {name} недоступний: {available}")
    return torch.device(name)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def make_model(regime, seed, device):
    """Порядок 1–5 з умови; повторного засівання перед навчанням немає."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    weights = None if regime == "scratch" else ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Linear(512, 6)
    for name, param in model.named_parameters():
        param.requires_grad = regime == "scratch" or name.startswith("fc.") or (
            regime == "unfreeze" and name.startswith("layer4."))
    model = model.to(device=device, dtype=torch.float32)
    if regime == "scratch":
        groups = [{"params": list(model.parameters()), "lr": 3e-3}]
    else:
        groups = [{"params": list(model.fc.parameters()), "lr": 3e-3}]
        if regime == "unfreeze":
            groups.append({"params": list(model.layer4.parameters()), "lr": 3e-4})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    assert all(p.requires_grad for g in optimizer.param_groups for p in g["params"])
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            assert module.momentum == 0.1 and module.eps == 1e-5
    return model, optimizer


def set_train_mode(model, regime):
    model.train()
    for name in EVAL_MODULES[regime]:
        getattr(model, name).eval()


def optimizer_description(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [{"numel": sum(p.numel() for p in g["params"]),
             "parameter_names": [names[id(p)] for p in g["params"]],
             **{k: v for k, v in g.items() if k != "params"}}
            for g in optimizer.param_groups]


def parameter_report(model, optimizer, regime):
    total = sum(p.numel() for p in model.parameters())
    trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total == 11179590 and trained == TRAINABLE[regime]
    groups = optimizer_description(model, optimizer)
    actual = [(g["numel"], g["lr"]) for g in groups]
    expected = {"scratch": [(11179590, 3e-3)], "frozen": [(3078, 3e-3)],
                "unfreeze": [(3078, 3e-3), (8393728, 3e-4)]}[regime]
    assert actual == expected, (actual, expected)
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {
        id(p) for p in model.parameters() if p.requires_grad}
    return {"regime": regime, "total_numel": total, "trainable_numel": trained,
            "optimizer_groups": groups, "match": True}


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    logits, labels = [], []
    for images, targets in data_loader:
        logits.append(model(images.to(device=device, dtype=torch.float32)).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def git_metadata():
    def query(args):
        return subprocess.run(["git", *args], cwd=BASE, capture_output=True, text=True)
    # Це єдині два дозволені git-запити в усьому проєкті.
    head = query(["rev-parse", "HEAD"])
    status = query(["status", "--porcelain"])
    return {"commit": head.stdout.strip() if head.returncode == 0 else None,
            "uncommitted_changes": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "commit_unavailable": head.returncode != 0, "repository_unavailable": status.returncode != 0}


def environment(device):
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        name = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    else:
        name = platform.processor() or platform.machine()
    return {"python": platform.python_version(), "packages": {
        p: importlib.metadata.version(p) for p in ["torch", "torchvision", "numpy", "scikit-learn", "matplotlib", "pillow"]},
        "device": device.type, "device_name": name, "platform": platform.platform(),
        "dtype": "float32", "PYTORCH_ENABLE_MPS_FALLBACK": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK"),
        "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads()}


def settings(model, optimizer, regime, epochs=8):
    return {"classes": CLASSES, "model": "torchvision.models.resnet18",
        "model_constructor_defaults": {"progress": True, "num_classes_before_fc_replacement": 1000,
            "zero_init_residual": False, "groups": 1, "width_per_group": 64,
            "replace_stride_with_dilation": None, "norm_layer": "BatchNorm2d"},
        "weights": None if regime == "scratch" else "ResNet18_Weights.IMAGENET1K_V1",
        "fc": {"in_features": 512, "out_features": 6, "bias": True, "initialization": "torch.nn.Linear default"},
        "dtype": "float32", "regime": regime, "eval_modules_during_train": EVAL_MODULES[regime],
        "batchnorm": {"momentum": 0.1, "eps": 1e-5, "affine": True, "track_running_stats": True},
        "transforms": {"train": [
            {"RandomResizedCrop": {"size": [160, 160], "scale": [0.7, 1.0],
                "ratio": [0.75, 4/3], "interpolation": "BILINEAR", "antialias": True}},
            {"RandomHorizontalFlip": {"p": 0.5}}, "ToTensor", {"Normalize": {"mean": MEAN, "std": STD, "inplace": False}}],
            "validation_test": [{"Resize": {"size": [160, 160], "interpolation": "BILINEAR", "max_size": None, "antialias": True}},
                "ToTensor", {"Normalize": {"mean": MEAN, "std": STD, "inplace": False}}]},
        "optimizer": {"class": "torch.optim.AdamW", "groups": optimizer_description(model, optimizer),
                      "defaults": optimizer.defaults},
        "loss": {"class": "CrossEntropyLoss", "weight": None, "size_average": None, "ignore_index": -100,
                 "reduce": None, "reduction": "mean", "label_smoothing": 0.0},
        "loader": {"train": {"batch_size": 32, "shuffle": True}, "validation_test": {"batch_size": 64, "shuffle": False},
            "shared": {"num_workers": 0, "drop_last": False, "pin_memory": False, "timeout": 0,
                "worker_init_fn": None, "generator": None, "custom_sampler": None, "collate_fn": "default_collate",
                "prefetch_factor": None, "persistent_workers": False, "multiprocessing_context": None,
                "pin_memory_device": "", "in_order": True}},
        "epochs": epochs, "steps_per_epoch": 38, "optimizer_steps": 304, "last_batch_size": 16,
        "zero_grad": {"set_to_none": inspect.signature(optimizer.zero_grad).parameters["set_to_none"].default},
        "macro_f1": {"average": "macro", "labels": [0,1,2,3,4,5], "zero_division": 0},
        "scheduler": None, "gradient_clipping": None, "mixed_precision": False,
        "early_stopping": False, "checkpoint_selection": "final epoch only", "seed_repeated_before_training": False,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic}


def load_runs(output=BASE, expected=RUNS):
    split_hash = sha256(Path(output) / "split.csv")
    result = []
    for regime, seed in expected:
        directory = Path(output) / "runs" / f"{regime}_seed{seed}"
        run = read_json(directory / "run.json")
        assert run["regime"] == regime and run["seed"] == seed
        assert run["split_sha256"] == split_hash, "Хеш split.csv у журналі не збігається"
        assert run["status"] == "complete" and len(run["history"]) == run["settings"]["epochs"]
        result.append(run)
    return result


def select_run(runs):
    means = {regime: float(np.mean([r["history"][-1]["val_macro_f1"] for r in runs if r["regime"] == regime]))
             for regime in ("scratch", "frozen", "unfreeze")}
    # При точній нічиїй фіксований порядок режимів, потім менше зерно.
    regime = max(means, key=means.get)
    selected = max((r for r in runs if r["regime"] == regime), key=lambda r: r["history"][-1]["val_macro_f1"])
    return selected, means


def effects(runs):
    values = {regime: [r["history"][-1]["val_macro_f1"] for r in runs if r["regime"] == regime]
              for regime in ("scratch", "frozen", "unfreeze")}
    fs = values["scratch"][0]
    frozen, unfreeze = float(np.mean(values["frozen"])), float(np.mean(values["unfreeze"]))
    rf, ru = float(np.ptp(values["frozen"])), float(np.ptp(values["unfreeze"]))
    threshold = max(rf, ru)
    result = {"F_scratch_1": fs, "F_mean_frozen": frozen, "R_frozen": rf,
              "F_mean_unfreeze": unfreeze, "R_unfreeze": ru, "R": threshold,
              "delta_pre": frozen-fs, "delta_unf": unfreeze-frozen, "sum": unfreeze-fs}
    result["verdicts"] = {key: ("не відрізняється від розкиду, результату немає" if abs(result[key]) <= threshold
                            else "ефект вартий перевірки, але не доведений") for key in ("delta_pre", "delta_unf")}
    return result


def configure_weight_cache():
    # Тимчасовий кеш не додає зайвого великого файлу до lab3/.
    torch.hub.set_dir(str(Path(tempfile.gettempdir()) / "lab3_torch_cache"))
