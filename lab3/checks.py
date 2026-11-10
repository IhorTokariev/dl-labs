"""Перевірки до експерименту; моделі звідси не входять до семи запусків."""
import itertools
import math
from pathlib import Path
import torch
from common import (MODULES, EVAL_MODULES, make_model, parameter_report, loader,
                    set_train_mode, write_json, sha256, source_hash)
from plots import batch_figure

BN_PREDICTION = (
    "Прогноз до запуску: параметри замороженого кістяка не зміняться, але його "
    "ковзні статистики BN змінюватимуться: це буфери, на які requires_grad не впливає. "
    "На кривій втрати train цю помилку, ймовірно, не буде помітно."
)


def state_copy(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def five_steps(model, optimizer, data_loader, device):
    losses = []
    for images, targets in itertools.islice(data_loader, 5):
        optimizer.zero_grad()
        loss = torch.nn.CrossEntropyLoss()(model(images.to(device)), targets.to(device))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    assert len(losses) == 5
    return losses


def changes(model, before, regime, bad=False):
    after = state_copy(model)
    parameter_names = set(dict(model.named_parameters()))
    rows = []
    for name in MODULES:
        param_keys = [k for k in parameter_names if k.startswith(name + ".")]
        buffer_keys = [k for k in before if k.startswith(name + ".") and
                       k.endswith(("running_mean", "running_var", "num_batches_tracked"))]
        params_changed = any(not torch.equal(before[k], after[k]) for k in param_keys)
        bn_changed = any(not torch.equal(before[k], after[k]) for k in buffer_keys) if buffer_keys else None
        expected_params = name not in EVAL_MODULES[regime]
        expected_bn = (True if bad else expected_params) if buffer_keys else None
        match = params_changed == expected_params and bn_changed == expected_bn
        row = {"regime": regime, "module": name, "parameters_changed": params_changed,
               "bn_statistics_changed": bn_changed, "match": match}
        print("Зміни:", row, flush=True)
        assert match, row
        # Окремо гарантуємо незмінність кожного тензора замороженого модуля.
        if not bad and name in EVAL_MODULES[regime]:
            assert all(torch.equal(before[k], after[k]) for k in before if k.startswith(name+"."))
        rows.append(row)
    return rows, after


def run_checks(data_root, output, device, split_report):
    output = Path(output)
    directory = output / "checks"
    directory.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    split_path = output / "split.csv"
    report = {"status": "RUNNING", "split_sha256": sha256(split_path),
              "source_sha256": source_hash(), "device": str(device), "split": split_report}
    initial = []
    for regime in ("scratch", "frozen"):
        model, optimizer = make_model(regime, 1, device)
        images, targets = next(iter(loader(data_root, split_path, "train")))
        if regime == "scratch":
            batch_figure(images, targets, output / "figures" / "batch_check.png")
        set_train_mode(model, regime)
        with torch.no_grad():
            loss = float(torch.nn.CrossEntropyLoss()(model(images.to(device)), targets.to(device)).cpu())
        entry = {"regime": regime, "seed": 1, "loss": loss, "ln6": math.log(6), "deviation": loss-math.log(6)}
        print("Початкова втрата:", entry, flush=True)
        initial.append(entry)
        del model, optimizer
    report["initial_losses"] = initial
    report["parameters"] = []
    for regime in ("scratch", "frozen", "unfreeze"):
        model, optimizer = make_model(regime, 1, device)
        row = parameter_report(model, optimizer, regime)
        print("Параметри:", regime, row["total_numel"], row["trainable_numel"],
              [(g["numel"], g["lr"]) for g in row["optimizer_groups"]], flush=True)
        report["parameters"].append(row)
        del model, optimizer
    report["changes"] = []
    for regime in ("frozen", "unfreeze"):
        model, optimizer = make_model(regime, 1, device)
        before = state_copy(model)
        set_train_mode(model, regime)
        five_steps(model, optimizer, loader(data_root, split_path, "train"), device)
        rows, _ = changes(model, before, regime)
        report["changes"].extend(rows)
        del model, optimizer, before
    # Фіксуємо прогноз на диску ДО створення моделі та виконання помилкового досліду.
    (directory / "intentional_bn_prediction.txt").write_text(BN_PREDICTION+"\n")
    print(BN_PREDICTION, flush=True)
    model, optimizer = make_model("frozen", 1, device)
    before = state_copy(model)
    model.train()
    losses = five_steps(model, optimizer, loader(data_root, split_path, "train"), device)
    rows, after = changes(model, before, "frozen", bad=True)
    max_changes = {name: float((after["layer1.0.bn1."+name] - before["layer1.0.bn1."+name]).abs().max())
                   for name in ("running_mean", "running_var")}
    assert all(value > 0 for value in max_changes.values())
    report["intentional_bn_error"] = {"prediction_before_experiment": BN_PREDICTION,
        "changes": rows, "five_train_losses": losses, "layer1.0.bn1_max_abs_change": max_changes,
        "prediction_supported": True,
        "curve_visibility": "П’ять кроків підтверджують зміну буферів; помітність на повній кривій не вимірювалася."}
    print("Навмисна помилка BN, max |Δ|:", max_changes, flush=True)
    report["status"] = "OK"
    write_json(directory / "checks_report.json", report)
    write_checks_markdown(report, directory / "checks_report.md")
    print("Усі перевірки п.1–2: OK", flush=True)
    return report


def yes(value):
    return "немає BN" if value is None else ("так" if value else "ні")


def changes_table(rows):
    text = "| Режим | Модуль | Параметри змінилися | Статистики BN змінилися | Збіг |\n|---|---|---|---|---|\n"
    return text + "".join(f'| {r["regime"]} | {r["module"]} | {yes(r["parameters_changed"])} | '
                          f'{yes(r["bn_statistics_changed"])} | {yes(r["match"])} |\n' for r in rows)


def write_checks_markdown(report, path):
    lines = ["# Перевірки п.1–2\n", f'Статус: **{report["status"]}**. Пристрій: {report["device"]}.\n',
             "## Поділ\n", "| Клас | train: кількість / сума | validation | test | Збіг |",
             "|---|---:|---:|---:|---|"]
    for row in report["split"]["table"]:
        values = " | ".join(f'{v["count"]} / {v["sum"]}' for v in row["parts"])
        lines.append(f'| {row["class"]} | {values} | {"так" if report["split"]["strict"] else "smoke"} |')
    lines += ["\nПеретинів і повторів файлів немає.\n", "## Початкова втрата\n",
              "| Режим | L₀ | ln 6 | L₀ − ln 6 |", "|---|---:|---:|---:|"]
    for r in report["initial_losses"]:
        lines.append(f'| {r["regime"]} | {r["loss"]:.9f} | {r["ln6"]:.9f} | {r["deviation"]:+.9f} |')
    lines += ["\n## Параметри й групи оптимізатора\n", "| Режим | Усього | Навчаються | Групи: numel / lr | Збіг |",
              "|---|---:|---:|---|---|"]
    for r in report["parameters"]:
        groups = "; ".join(f'{g["numel"]} / {g["lr"]}' for g in r["optimizer_groups"])
        lines.append(f'| {r["regime"]} | {r["total_numel"]} | {r["trainable_numel"]} | {groups} | так |')
    lines += ["\nЗаморожених параметрів в оптимізаторі немає.\n", "## Зміни після п’яти кроків\n",
              changes_table(report["changes"]), "\n## Навмисна помилка model.train()\n",
              report["intentional_bn_error"]["prediction_before_experiment"] + "\n",
              changes_table(report["intentional_bn_error"]["changes"])]
    for key, value in report["intentional_bn_error"]["layer1.0.bn1_max_abs_change"].items():
        lines.append(f"max |Δ {key}| у layer1.0.bn1 = {value:.9f}.\n")
    lines += ["Прогноз зміни параметрів і буферів підтверджено. " + report["intentional_bn_error"]["curve_visibility"],
              "\nrequires_grad=False керує обчисленням градієнтів параметрів; running_mean, running_var та "
              "num_batches_tracked є буферами й оновлюються у forward за training=True. У unfreeze "
              "layer4 навчається, тому оновлення його BN-статистик відповідає протоколу."]
    Path(path).write_text("\n".join(lines)+"\n")
