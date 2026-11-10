"""Навчання ЛР3. Test тут не завантажується й не оцінюється."""
from __future__ import annotations
import argparse
import gc
import math
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
import torch

from common import (BASE, CLASSES, RUNS, build_split, check_split, choose_device, configure_weight_cache,
    environment, evaluate, git_metadata, load_runs, loader, make_model, parameter_report, read_json,
    select_run, set_train_mode, settings, sha256, source_hash, synchronize, write_json)
from checks import run_checks
from plots import curves_figure


def train_run(regime, seed, data_root, output, device, epochs=8, smoke=False):
    model, optimizer = make_model(regime, seed, device)
    counts = parameter_report(model, optimizer, regime)
    configuration = settings(model, optimizer, regime, epochs)
    train_loader = loader(data_root, Path(output)/"split.csv", "train")
    val_loader = loader(data_root, Path(output)/"split.csv", "validation")
    if not smoke:
        assert len(train_loader.dataset) == 1200 and len(train_loader) == 38
        assert len(val_loader.dataset) == 1498
    else:
        configuration.update(steps_per_epoch=len(train_loader), optimizer_steps=len(train_loader)*epochs,
                             last_batch_size=len(train_loader.dataset) % 32 or 32)
    criterion = torch.nn.CrossEntropyLoss()
    history = []
    synchronize(device)
    start = time.perf_counter()
    total_steps = 0
    for epoch in range(1, epochs+1):
        epoch_start = time.perf_counter()
        set_train_mode(model, regime)
        weighted_loss, seen, steps = 0.0, 0, 0
        for images, targets in train_loader:
            images, targets = images.to(device=device, dtype=torch.float32), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.detach().cpu()) * len(targets)
            seen += len(targets)
            steps += 1
            last_batch = len(targets)
        assert seen == len(train_loader.dataset)
        assert steps == configuration["steps_per_epoch"] and last_batch == configuration["last_batch_size"]
        total_steps += steps
        logits, labels = evaluate(model, val_loader, device)
        val_f1 = float(f1_score(labels, logits.argmax(1), average="macro", labels=range(6), zero_division=0))
        synchronize(device)
        record = {"epoch": epoch, "train_loss": weighted_loss/seen, "val_macro_f1": val_f1,
                  "steps": steps, "last_batch_size": last_batch, "epoch_seconds": time.perf_counter()-epoch_start}
        assert math.isfinite(record["train_loss"]) and np.isfinite(logits).all()
        history.append(record)
        print(f'{regime} seed={seed} | епоха {epoch}/{epochs} | loss={record["train_loss"]:.9f} | '
              f'val macro-F1={val_f1:.9f} | {record["epoch_seconds"]:.2f} с', flush=True)
    synchronize(device)
    seconds = time.perf_counter()-start
    assert total_steps == configuration["optimizer_steps"]
    directory = Path(output) / "runs" / f"{regime}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory/"val_logits.npy", logits)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state, directory/"weights.pt")
    record = {"status": "complete", "regime": regime, "seed": seed, "settings": configuration,
              "parameter_counts": counts, "split_sha256": sha256(Path(output)/"split.csv"),
              "source_sha256": source_hash(), "git": git_metadata(), "environment": environment(device),
              "history": history, "training_seconds": seconds,
              "timing_definition": "Цикл 8 епох з train і validation; без побудови моделі й запису артефактів",
              "optimizer_steps": total_steps, "smoke": smoke,
              "weights_sha256": sha256(directory/"weights.pt"),
              "val_logits_sha256": sha256(directory/"val_logits.npy")}
    # run.json з’являється останнім і є ознакою завершеного запуску.
    write_json(directory/"run.json", record)
    del model, optimizer, state
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return record


def resume_valid(directory, split_path):
    required = [directory/name for name in ("run.json", "val_logits.npy", "weights.pt")]
    if not all(p.exists() for p in required):
        return False
    record = read_json(directory/"run.json")
    assert record["status"] == "complete" and len(record["history"]) == 8
    assert record["split_sha256"] == sha256(split_path), "Змінився поділ; не можна змішувати запуски"
    assert record["source_sha256"] == source_hash(), "Змінився код; потрібні всі сім запусків з одного коду"
    assert record["weights_sha256"] == sha256(directory/"weights.pt")
    assert record["val_logits_sha256"] == sha256(directory/"val_logits.npy")
    logits = np.load(directory/"val_logits.npy", allow_pickle=False)
    assert logits.shape == (1498, 6) and np.isfinite(logits).all()
    return True


def save_selection(runs, output):
    selected, means = select_run(runs)
    directory = Path(output)/"runs"/f'{selected["regime"]}_seed{selected["seed"]}'
    state = torch.load(directory/"weights.pt", map_location="cpu", weights_only=True)
    torch.save({"state_dict": state, "regime": selected["regime"], "seed": selected["seed"],
                "classes": CLASSES, "img_size": 160}, Path(output)/"lab3_model.pt")
    print("Середні validation:", means, flush=True)
    print(f'Вибрано лише за validation: {selected["regime"]}_seed{selected["seed"]}', flush=True)


def smoke(device):
    # Усі синтетичні зображення й артефакти зникають після перевірки.
    with tempfile.TemporaryDirectory(prefix="lab3_smoke_") as tmp:
        output = Path(tmp)
        root = output/"intel_image"
        rng = np.random.default_rng(73)
        number = 0
        for directory, per_class in (("seg_train", 32), ("seg_test", 8)):
            for index, label in enumerate(CLASSES):
                folder = root/directory/directory/label
                folder.mkdir(parents=True)
                for j in range(per_class):
                    height, width = [(113,150), (150,150), (123,137)][j%3]
                    pixels = rng.integers(0, 60, (height,width,3), dtype=np.uint8)
                    pixels[:,:,index%3] = np.clip(pixels[:,:,index%3].astype(int)+100+index*12,0,255)
                    Image.fromarray(pixels).save(folder/f"{number}.jpg")
                    number += 1
        rows = build_split(root, output/"split.csv", train_per_class=32)
        report = check_split(rows, strict=False)
        run_checks(root, output, device, report)
        runs = [train_run(regime, 1, root, output, device, epochs=1, smoke=True)
                for regime in ("scratch", "frozen", "unfreeze")]
        curves_figure(runs, output/"figures"/"curves.png")
        save_selection(runs, output)
        assert (output/"lab3_model.pt").exists()
        print("SMOKE OK: три режими, JPEG різних розмірів, перевірки BN і запис артефактів. "
              "Ці числа не є результатами лабораторної.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=BASE/"intel_image")
    parser.add_argument("--device", choices=["auto","mps","cuda","cpu"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=BASE,
                        help="Тека артефактів; відносний шлях рахується від теки скрипта")
    args = parser.parse_args(argv)
    configure_weight_cache()
    device = choose_device(args.device)
    print("Середовище:", environment(device), flush=True)
    if args.smoke:
        smoke(device)
        return
    output = args.output_dir.expanduser()
    output = (output if output.is_absolute() else BASE/output).resolve()
    data_root = args.data_root.expanduser().resolve()
    if output == data_root or data_root in output.parents:
        raise RuntimeError("Тека результатів не може бути всередині датасету")
    # Перевіряємо конфлікт ДО запису поділу й повторення перевірок.
    if not args.resume and any(
            (output/"runs"/f"{regime}_seed{seed}"/name).exists()
            for regime, seed in RUNS for name in ("run.json", "val_logits.npy", "weights.pt")):
        raise RuntimeError(
            f"У {output} вже є результати. Для нового експерименту вкажіть нову "
            "--output-dir; для продовження цього експерименту — --resume")
    output.mkdir(parents=True, exist_ok=True)
    rows = build_split(data_root, output/"split.csv")
    split_report = check_split(rows)
    checks_path = output/"checks"/"checks_report.json"
    if args.resume and checks_path.exists() and read_json(checks_path).get("status") == "OK":
        checks = read_json(checks_path)
        assert checks["split_sha256"] == sha256(output/"split.csv")
        assert checks["source_sha256"] == source_hash()
        print("--resume: перевірки зі статусом OK вже пройдено.", flush=True)
    else:
        run_checks(data_root, output, device, split_report)
    for regime, seed in RUNS:
        directory = output/"runs"/f"{regime}_seed{seed}"
        if args.resume and resume_valid(directory, output/"split.csv"):
            print(f"--resume: пропущено завершений {regime}_seed{seed}", flush=True)
            continue
        train_run(regime, seed, data_root, output, device)
    runs = load_runs(output)
    frozen = [r["history"][-1]["val_macro_f1"] for r in runs if r["regime"] == "frozen"]
    scratch = runs[0]["history"][-1]["val_macro_f1"]
    assert min(frozen) >= 0.80, "Порушено орієнтир frozen; перевірте BN і нормування до аналізу test"
    assert scratch > 0.25 and scratch < float(np.mean(frozen)), "Порушено орієнтир scratch; перевірте реалізацію"
    curves_figure(runs, output/"figures"/"curves.png")
    save_selection(runs, output)
    print("TRAIN OK: рівно сім запусків, кожен 8 епох і 304 кроки. Test не обчислювався.", flush=True)


if __name__ == "__main__":
    main()
