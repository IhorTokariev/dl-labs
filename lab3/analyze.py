"""Аналіз журналів та єдина тестова процедура на CPU."""
import argparse
import math
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
from torchvision.models import resnet18
from common import (BASE, CLASSES, RUNS, check_split, effects, evaluate, image_path, load_runs,
                    load_split, loader, select_run, sha256, write_json)
from plots import confusion_figure, curves_figure, errors_figure

# Допуски перевірки MPS → CPU фіксовані й не впливають на вибір або навчання.
VALIDATION_LOGITS_RTOL = 1e-4
VALIDATION_LOGITS_ATOL = 1e-4


def check_checkpoint_metadata(checkpoint, selected):
    """Перевіряємо правило вибору до завантаження test."""
    if set(checkpoint) != {"state_dict", "regime", "seed", "classes", "img_size"}:
        raise RuntimeError("Некоректні ключі lab3_model.pt")
    if (checkpoint["regime"], checkpoint["seed"]) != (selected["regime"], selected["seed"]):
        raise RuntimeError("lab3_model.pt не відповідає правилу вибору за журналами")
    if checkpoint["classes"] != CLASSES or checkpoint["img_size"] != 160:
        raise RuntimeError("Класи або розмір зображення lab3_model.pt не відповідають протоколу")


def check_validation_logits(actual, saved):
    """Звіряємо поведінку моделі без локальних ваг окремого запуску."""
    if actual.shape != (1498, 6) or saved.shape != (1498, 6):
        raise RuntimeError("Некоректна форма логітів validation: очікується 1498×6")
    if not np.isfinite(actual).all() or not np.isfinite(saved).all():
        raise RuntimeError("Логіти validation містять NaN або нескінченність")
    difference = float(np.max(np.abs(actual - saved)))
    if not np.allclose(actual, saved, rtol=VALIDATION_LOGITS_RTOL, atol=VALIDATION_LOGITS_ATOL):
        raise RuntimeError(
            f"Логіти validation lab3_model.pt не відповідають вибраному запуску: "
            f"max |Δ|={difference:.9g}; test не обчислювався")
    matching = int(np.count_nonzero(actual.argmax(1) == saved.argmax(1)))
    if matching != len(saved):
        raise RuntimeError(
            f"Передбачення validation lab3_model.pt не відповідають вибраному запуску: "
            f"збіг {matching}/{len(saved)}; test не обчислювався")
    return {"method": "cpu_validation_logits", "rtol": VALIDATION_LOGITS_RTOL,
            "atol": VALIDATION_LOGITS_ATOL, "max_abs_difference": difference,
            "matching_predictions": matching, "validation_rows": len(saved),
            "requires_local_run_weights": False}


def pick_errors(rows, targets, predicted, probabilities, cm):
    pairs = sorted([(i,j) for i in range(6) for j in range(6) if i != j and cm[i,j] > 0],
                   key=lambda pair: (-int(cm[pair]), pair[0], pair[1]))
    candidates = {pair: sorted([i for i, row in enumerate(rows)
                                if targets[i] == pair[0] and predicted[i] == pair[1]],
                               key=lambda i: rows[i]["file"]) for pair in pairs}
    chosen = []
    rank = 0
    while len(chosen) < 5:
        added = False
        for pair in pairs:
            if len(chosen) == 5:
                break
            if rank < len(candidates[pair]):
                i = candidates[pair][rank]
                chosen.append({"file": rows[i]["file"], "true_class": CLASSES[pair[0]],
                    "predicted_class": CLASSES[pair[1]], "probability": float(probabilities[i,pair[1]]),
                    "pair_count": int(cm[pair]), "test_row_index": i})
                added = True
        if not added:
            raise RuntimeError("На test менше п’яти помилок; неможливо обрати п’ять різних файлів")
        rank += 1
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=BASE/"intel_image")
    parser.add_argument("--output-dir", type=Path, default=BASE,
                        help="Тека артефактів; та сама, що передана команді навчання")
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser()
    output = (output if output.is_absolute() else BASE/output).resolve()
    data_root = args.data_root.expanduser().resolve()
    if output == data_root or data_root in output.parents:
        raise RuntimeError("Тека результатів не може бути всередині датасету")
    rows = load_split(output/"split.csv")
    split_report = check_split(rows)
    runs = load_runs(output)
    assert len(runs) == 7
    validation = [r for r in rows if r["part"] == "validation"]
    val_targets = np.array([CLASSES.index(r["label"]) for r in validation])
    print("\nТаблиця семи запусків (остання, восьма епоха):")
    print("| # | Режим | Зерно | Train loss | Val macro-F1 | Час, с |")
    print("|---|---|---:|---:|---:|---:|")
    run_table = []
    for i, run in enumerate(runs, 1):
        directory = output/"runs"/f'{run["regime"]}_seed{run["seed"]}'
        assert sha256(directory/"val_logits.npy") == run["val_logits_sha256"]
        val_logits = np.load(directory/"val_logits.npy", allow_pickle=False)
        assert val_logits.shape == (1498,6) and np.isfinite(val_logits).all()
        recomputed = float(f1_score(val_targets, val_logits.argmax(1), average="macro", labels=range(6), zero_division=0))
        assert abs(recomputed-run["history"][-1]["val_macro_f1"]) < 1e-12
        entry = {"number": i, "regime": run["regime"], "seed": run["seed"],
                 "train_loss": run["history"][-1]["train_loss"], "val_macro_f1": recomputed,
                 "training_seconds": run["training_seconds"]}
        run_table.append(entry)
        print(f'| {i} | {run["regime"]} | {run["seed"]} | {entry["train_loss"]:.9f} | '
              f'{recomputed:.9f} | {run["training_seconds"]:.3f} |')
    decomposition = effects(runs)
    print("\nРозклад ефектів:")
    for key, value in decomposition.items():
        if key != "verdicts":
            print(f"{key} = {value:.12f}")
    for key, verdict in decomposition["verdicts"].items():
        print(f"{key}: {verdict}")
    selected, means = select_run(runs)
    checkpoint = torch.load(output/"lab3_model.pt", map_location="cpu", weights_only=True)
    check_checkpoint_metadata(checkpoint, selected)
    # Лише опубліковані артефакти: основна модель та логіти validation запуску.
    device = torch.device("cpu")
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 6)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device=device, dtype=torch.float32)
    selected_directory = output/"runs"/f'{selected["regime"]}_seed{selected["seed"]}'
    saved_validation = np.load(selected_directory/"val_logits.npy", allow_pickle=False)
    actual_validation, actual_targets = evaluate(model, loader(data_root, output/"split.csv", "validation"), device)
    if not np.array_equal(actual_targets, val_targets):
        raise RuntimeError("Порядок міток validation не відповідає split.csv")
    verification = check_validation_logits(actual_validation, saved_validation)
    print(f'\nВибраний запуск: {selected["regime"]}_seed{selected["seed"]}; правило вибору перевірено.')
    print(f'Логіти validation моделі на CPU: max |Δ|={verification["max_abs_difference"]:.9g}; '
          f'atol={VALIDATION_LOGITS_ATOL:g}, rtol={VALIDATION_LOGITS_RTOL:g}; '
          f'збіг передбачень {verification["matching_predictions"]}/{verification["validation_rows"]}.')
    if abs(decomposition["delta_unf"]) <= decomposition["R"]:
        print("Вибір між frozen і unfreeze визначив шум.")
    # Лише після перевірки вибору завантажуємо test. Усі обчислення CPU float32.
    test_loader = loader(data_root, output/"split.csv", "test")
    assert len(test_loader.dataset) == 1502
    logits, targets = evaluate(model, test_loader, device)
    assert logits.shape == (1502,6) and np.isfinite(logits).all()
    predicted = logits.argmax(1)
    test_f1 = float(f1_score(targets, predicted, average="macro", labels=range(6), zero_division=0))
    accuracy = float(accuracy_score(targets, predicted))
    standard_error = math.sqrt(accuracy*(1-accuracy)/len(targets))
    cm = confusion_matrix(targets, predicted, labels=range(6))
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    error_rows = pick_errors(test_loader.dataset.rows, targets, predicted, probabilities, cm)
    chosen_val = selected["history"][-1]["val_macro_f1"]
    print(f"Validation macro-F1 вибраного запуску = {chosen_val:.12f}")
    print(f"Test macro-F1 = {test_f1:.12f}")
    print(f"Test accuracy = {accuracy:.12f}")
    print(f"Правильних відповідей = {int((targets==predicted).sum())} / {len(targets)}")
    print(f"Стандартна похибка accuracy sqrt(p(1-p)/n) = {standard_error:.12f}")
    print(f"Test F1 − val F1 = {test_f1-chosen_val:+.12f}")
    print("\nМатриця плутанини: рядки справжні, стовпці передбачені;", ", ".join(CLASSES))
    print("| Справжній / передбачений | " + " | ".join(CLASSES) + " |")
    print("|---|" + "---:|"*6)
    for name, values in zip(CLASSES,cm):
        print("| " + name + " | " + " | ".join(map(str,values)) + " |")
    print("\nП’ять характерних помилок:")
    print("| Файл | Справжній клас | Передбачений клас | Імовірність | Кількість пари |")
    print("|---|---|---|---:|---:|")
    for error in error_rows:
        print(f'| {error["file"]} | {error["true_class"]} | {error["predicted_class"]} | '
              f'{error["probability"]:.12f} | {error["pair_count"]} |')
    (output/"checks").mkdir(parents=True, exist_ok=True)
    (output/"figures").mkdir(parents=True, exist_ok=True)
    np.save(output/"checks"/"test_logits_cpu.npy", logits)
    result = {"split_sha256": sha256(output/"split.csv"), "model_sha256": sha256(output/"lab3_model.pt"),
        "runs": run_table, "effects": decomposition, "regime_means": means,
        "selection_verification": verification,
        "selected": {"regime": selected["regime"], "seed": selected["seed"], "val_macro_f1": chosen_val},
        "test": {"device": "cpu", "dtype": "float32", "n": len(targets), "macro_f1": test_f1,
            "accuracy": accuracy, "correct": int((targets==predicted).sum()), "accuracy_standard_error": standard_error,
            "test_minus_val_macro_f1": test_f1-chosen_val, "confusion_matrix": cm.tolist(),
            "logits_sha256": sha256(output/"checks"/"test_logits_cpu.npy")}, "errors": error_rows}
    write_json(output/"checks"/"analysis_report.json", result)
    curves_figure(runs, output/"figures"/"curves.png")
    confusion_figure(cm, output/"figures"/"confusion_matrix.png")
    errors_figure(error_rows, data_root, output/"figures"/"top5_errors.png")
    print("\nANALYZE OK: хеші, вибір, логіти validation й тестова оцінка на CPU перевірені.")


if __name__ == "__main__":
    main()
