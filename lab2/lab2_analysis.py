"""ЛР2: незалежні перевірки, поріг на validation і оцінка зафіксованого test.

Використання: uv run python lab2_analysis.py --saved results
CSV та повторне навчання для аналізу не потрібні.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

# Кеш шрифтів і конфігурація також залишаються всередині робочої теки.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "work" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / "work" / "cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from lab2_starter import load_predictions

COUNTERS = ["TP", "FP", "FN", "TN", "C"]
RULES = ["стандартний поріг 0.5", "мінімум вартості t*", "усі негативні (+inf)"]


def confusion(y: np.ndarray, s: np.ndarray, t: float) -> tuple[int, int, int, int]:
    """Прогноз 1 рівно за s >= t; порядок результатів: TP, FP, FN, TN."""
    positive = s >= t
    truth = y == 1
    return (
        int(np.count_nonzero(positive & truth)),
        int(np.count_nonzero(positive & ~truth)),
        int(np.count_nonzero(~positive & truth)),
        int(np.count_nonzero(~positive & ~truth)),
    )


def metrics(tp: int, fp: int, fn: int) -> tuple[int, float | None, float | None]:
    """Нульовий знаменник дає None, а не умовний нуль."""
    return (10 * fn + fp,
            tp / (tp + fp) if tp + fp else None,
            tp / (tp + fn) if tp + fn else None)


def fast_search(s: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, float]:
    """Один прохід після сортування, без циклу за кандидатами."""
    order = np.argsort(-s, kind="stable")
    scores = s[order]
    labels = y[order]
    # Остання позиція кожної групи: всі рівні оцінки включаємо разом.
    ends = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True])
    tp = np.r_[0, np.cumsum(labels == 1, dtype=np.int64)[ends]]
    fp = np.r_[0, np.cumsum(labels == 0, dtype=np.int64)[ends]]
    p = int(np.count_nonzero(y == 1))
    n = int(np.count_nonzero(y == 0))
    fn, tn = p - tp, n - fp
    table = pd.DataFrame({"threshold": np.r_[np.inf, scores[ends].astype(np.float64)],
                          "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                          "C": 10 * fn + fp})
    # Кандидати вже спадають; перший мінімум має найбільший поріг.
    best = int(np.argmin(table["C"].to_numpy()))
    return table, float(table.iloc[best]["threshold"])


def brute_force(s: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, float]:
    """Незалежна пряма підстановка, дозволена лише для малих A/B/C.

    Не викликає fast_search, confusion чи metrics і не використовує їхні таблиці.
    """
    candidates = np.r_[np.unique(s), np.inf]
    rows = []
    best_cost = None
    best_threshold = -math.inf
    for candidate in candidates:
        prediction = s >= candidate
        tp = int(np.sum((y == 1) & prediction))
        fp = int(np.sum((y == 0) & prediction))
        fn = int(np.sum((y == 1) & ~prediction))
        tn = int(np.sum((y == 0) & ~prediction))
        cost = 10 * fn + fp
        threshold = float(candidate)
        rows.append({"threshold": threshold, "TP": tp, "FP": fp,
                     "FN": fn, "TN": tn, "C": cost})
        if best_cost is None or cost < best_cost or (cost == best_cost and threshold > best_threshold):
            best_cost, best_threshold = cost, threshold
    return pd.DataFrame(rows).sort_values("threshold", ascending=False).reset_index(drop=True), best_threshold


def display(value: object) -> str:
    """Округлення лише в Markdown; CSV і JSON зберігають повну точність."""
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return "не визначена"
    if isinstance(value, (float, np.floating)):
        if np.isposinf(value):
            return "+inf"
        return f"{value:.9g}"
    return str(value)


def markdown(table: pd.DataFrame) -> str:
    """Формування таблиці без додаткової залежності tabulate."""
    columns = list(table.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(display(v) for v in row) + " |"
                 for row in table.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sanity_checks(out: Path) -> pd.DataFrame:
    """Будь-яка незбіжність припиняє виконання до завантаження прогнозів."""
    cases = {
        "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
        "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
        # PDF, стор. 2: одинадцять 0.5, перша позитивна, десять негативних.
        "C": ([0.8] + [0.5] * 11 + [0.2], [1, 1] + [0] * 10 + [0]),
    }
    descriptions = {
        "A": "Рівні оцінки 0.7 з різними мітками включаються одночасно.",
        "B": "Позитивів немає: recall не визначений; +inf дає мінімальну вартість.",
        "C": "Два кандидати мають однаковий мінімум; перемагає більший поріг.",
    }
    report = ["# Незалежні перевірки A/B/C", "",
              "Масиви оцінок float32. Пороги в таблицях округлено лише для відображення.",
              "Набір C візуально звірено з таблицею та приміткою на сторінці 2 PDF: 13 елементів.", ""]
    summary = []
    for name, (scores, labels) in cases.items():
        s = np.asarray(scores, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        fast, fast_t = fast_search(s, y)
        slow, slow_t = brute_force(s, y)
        fast.to_csv(out / f"sanity_{name}_fast.csv", index=False)
        slow.to_csv(out / f"sanity_{name}_brute.csv", index=False)
        failures = []
        if not np.array_equal(fast["threshold"].to_numpy(), slow["threshold"].to_numpy()):
            failures.append("списки кандидатів не збігаються точно")
        if not np.array_equal(fast[COUNTERS].to_numpy(), slow[COUNTERS].to_numpy()):
            failures.append("TP/FP/FN/TN/C швидкої та повільної версій не збігаються точно")
        if fast_t != slow_t:
            failures.append("обрані пороги не збігаються точно")
        # Додатково порівнюємо власний прямий підрахунок з незалежним перебором.
        for row in slow.itertuples(index=False):
            direct = confusion(y, s, row.threshold)
            if direct != (row.TP, row.FP, row.FN, row.TN):
                failures.append(f"confusion помиляється при t={row.threshold!r}")
        if name == "B":
            all_negative = confusion(y, s, math.inf)
            cost, precision, recall = metrics(*all_negative[:3])
            if any(metrics(int(r.TP), int(r.FP), int(r.FN))[2] is not None for r in slow.itertuples()):
                failures.append("recall має бути None для всіх порогів без позитивів")
            if not (math.isinf(fast_t) and cost == 0 and precision is None and recall is None):
                failures.append("некоректний граничний поріг або метрики з нульовим знаменником")
        if name == "C":
            minima = slow.loc[slow["C"] == slow["C"].min()]
            if len(minima) != 2 or fast_t != float(minima["threshold"].max()):
                failures.append("не дотримано правила найбільшого порога серед двох мінімумів")
        ok = not failures
        chosen = slow.loc[slow["threshold"] == slow_t].iloc[0]
        summary.append({"Набір": name, "n": len(y), "t*": fast_t,
                        "C": int(chosen["C"]), "Збіг": "ТОЧНИЙ" if ok else "ПОМИЛКА"})
        report += [f"## Набір {name}", "", descriptions[name], "",
                   "### Швидкий алгоритм", "", markdown(fast), "",
                   "### Незалежний перебір", "", markdown(slow), "",
                   f"Поріг: {display(fast_t)}. Порівняння: " + ("ТОЧНИЙ ЗБІГ." if ok else "; ".join(failures)), ""]
        if failures:
            (out / "sanity_checks.md").write_text("\n".join(report), encoding="utf-8")
            raise SystemExit(f"Перевірка {name} НЕ ПРОЙШЛА: {'; '.join(failures)}. Оцінку test заборонено.")
        print(f"Перевірка {name}: ТОЧНИЙ ЗБІГ, t*={fast_t!r}, C={int(chosen['C'])}")
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out / "sanity_summary.csv", index=False)
    report += ["## Підсумок", "", markdown(summary_df), "",
               "Усі три перевірки пройшли. Recall набору B не визначений через відсутність позитивів;",
               "precision при +inf не визначена через відсутність позитивних прогнозів.", ""]
    (out / "sanity_checks.md").write_text("\n".join(report), encoding="utf-8")
    return summary_df


def check_saved(data: dict[str, np.ndarray], out: Path) -> pd.DataFrame:
    """Перевірка збереженого поділу; жодного читання вихідного CSV."""
    rows = []
    ids = []
    for name in ("train", "validation", "test"):
        y = data[f"{name}_y"]
        row_ids = data[f"{name}_row_ids"]
        amount = data[f"{name}_amount"]
        if y.ndim != 1 or not np.isin(y, [0, 1]).all() or len(y) != len(row_ids) or len(y) != len(amount):
            raise SystemExit(f"Некоректні мітки, індекси чи Amount у {name}.")
        if not np.isfinite(amount).all():
            raise SystemExit(f"Некоректні Amount у {name}.")
        if name != "train":
            s = data[f"{name}_scores"]
            if s.shape != y.shape or s.dtype != np.float32 or not np.isfinite(s).all() or ((s < 0) | (s > 1)).any():
                raise SystemExit(f"Некоректні оцінки {name}: потрібний скінченний масив float32 у [0,1].")
        ids.append(row_ids)
        rows.append({"split": name, "n": len(y), "positives": int(np.count_nonzero(y == 1)),
                     "positive_fraction": float(np.mean(y == 1))})
    summary = pd.DataFrame(rows)
    if summary["n"].tolist() != [170884, 56961, 56962] or int(summary["positives"].sum()) != 492:
        raise SystemExit("Поділ не відповідає умові: 170884/56961/56962 рядків і 492 позитиви разом.")
    if not np.array_equal(np.sort(np.concatenate(ids)), np.arange(int(summary["n"].sum()))):
        raise SystemExit("Індекси поділу мають пропуски, повторення або перетин вибірок.")
    persisted = pd.read_csv(out / "split_summary.csv", float_precision="round_trip")
    if not np.array_equal(summary[["n", "positives"]].to_numpy(), persisted[["n", "positives"]].to_numpy()):
        raise SystemExit("split_summary.csv не збігається зі збереженими масивами.")
    return summary


def rule_table(y: np.ndarray, s: np.ndarray, threshold: float, with_metrics: bool) -> pd.DataFrame:
    rows = []
    for name, t in zip(RULES, [0.5, threshold, math.inf], strict=True):
        tp, fp, fn, tn = confusion(y, s, t)
        cost, precision, recall = metrics(tp, fp, fn)
        row = {"rule": name, "threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost}
        if with_metrics:
            row.update(precision=precision, recall=recall)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_validation(y: np.ndarray, s: np.ndarray, out: Path) -> float:
    auc = float(roc_auc_score(y, s))
    precision, recall, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    prevalence = float(np.mean(y == 1))
    fig, ax = plt.subplots(figsize=(7.2, 5), layout="constrained")
    ax.step(recall, precision, where="post", color="#0072B2", linewidth=1.8, label="Модель")
    ax.axhline(prevalence, color="#D55E00", linestyle="--", label=f"Частка позитивів: {prevalence:.6g}")
    ax.set(xlabel="Повнота (recall)", ylabel="Точність (precision)", title="PR-крива на validation",
           xlim=(-0.01, 1.01), ylim=(-0.015, 1.02))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.savefig(out / "pr_curve.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 5), layout="constrained")
    ax.plot(fpr, tpr, color="#0072B2", linewidth=1.8, label=f"ROC-AUC = {auc:.9g}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Випадкове ранжування")
    ax.set(xlabel="Частка хибних тривог (FPR)", ylabel="Повнота (TPR)", title="ROC-крива на validation",
           xlim=(-0.01, 1.01), ylim=(-0.01, 1.02))
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.savefig(out / "roc_curve.png", dpi=180)
    plt.close(fig)
    return auc


def error_examples(data: dict[str, np.ndarray], threshold: float, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    y, s = data["test_y"], data["test_scores"]
    row_ids, amount = data["test_row_ids"], data["test_amount"]
    predicted = s >= threshold
    rows = []
    for kind, mask in [("FP", predicted & (y == 0)), ("FN", ~predicted & (y == 1))]:
        positions = np.flatnonzero(mask)
        first = positions[np.argsort(row_ids[positions], kind="stable")[:5]]
        for i in first:
            rows.append({"csv_row": int(row_ids[i]), "type": kind, "label": int(y[i]),
                         "score": float(s[i]), "Amount": float(amount[i]),
                         "distance": float(np.abs(s[i] - np.float32(threshold)))})
    examples = pd.DataFrame(rows, columns=["csv_row", "type", "label", "score", "Amount", "distance"])
    examples.to_csv(out / "error_examples.csv", index=False)
    summaries = []
    for kind in ("FP", "FN"):
        chosen = examples[examples["type"] == kind]
        summaries.append({"type": kind, "n": len(chosen),
                          "Amount_mean": float(chosen["Amount"].mean()) if len(chosen) else None,
                          "Amount_median": float(chosen["Amount"].median()) if len(chosen) else None,
                          "distance_mean": float(chosen["distance"].mean()) if len(chosen) else None,
                          "distance_median": float(chosen["distance"].median()) if len(chosen) else None})
    stats = pd.DataFrame(summaries)
    stats.to_csv(out / "error_summary.csv", index=False)
    return examples, stats


def save_decision_changes(out: Path, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    """Зберегти зміну FN, FP та вартості відносно стандартного порога 0.5."""
    rows = []
    for name, table in (("validation", validation), ("test", test)):
        standard, chosen = table.iloc[0], table.iloc[1]
        rows.append({"Вибірка": name,
                     "FN уникнуто (0.5 → t*)": int(standard["FN"] - chosen["FN"]),
                     "Додаткові FP (0.5 → t*)": int(chosen["FP"] - standard["FP"]),
                     "Зменшення C (0.5 → t*)": int(standard["C"] - chosen["C"])})
    pd.DataFrame(rows).to_csv(out / "decision_changes.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved", type=Path, required=True, help="Тека з predictions.npz; CSV не потрібен")
    args = parser.parse_args()
    out = args.saved
    if not out.is_dir() or not (out / "predictions.npz").is_file():
        raise SystemExit("У --saved немає predictions.npz. Аналіз потребує збереженого запуску, без навчання.")

    # Бар'єр: невдала перевірка припиняє програму до завантаження даних моделі.
    sanity_checks(out)
    data = load_predictions(out)
    check_saved(data, out)
    y_va, s_va = data["validation_y"], data["validation_scores"]
    candidates, threshold = fast_search(s_va, y_va)
    candidates.to_csv(out / "threshold_candidates.csv", index=False)
    write_json(out / "threshold.json", {"threshold": "inf" if math.isinf(threshold) else threshold,
                                        "selected_on": "validation", "rule": "s >= t",
                                        "cost": "10*FN + FP", "tie_break": "largest_threshold"})
    validation = rule_table(y_va, s_va, threshold, with_metrics=False)
    validation.to_csv(out / "validation_rules.csv", index=False)
    auc = plot_validation(y_va, s_va, out)

    # Єдине джерело робочого порога для test - файл, а не локальний результат пошуку.
    frozen = json.loads((out / "threshold.json").read_text(encoding="utf-8"))
    test_threshold = math.inf if frozen["threshold"] == "inf" else float(frozen["threshold"])
    test = rule_table(data["test_y"], data["test_scores"], test_threshold, with_metrics=True)
    test.to_csv(out / "test_rules.csv", index=False)
    examples, stats = error_examples(data, test_threshold, out)
    write_json(out / "analysis_summary.json", {
        "threshold": frozen["threshold"], "validation_roc_auc": auc,
        "validation_cost": int(validation.iloc[1]["C"]), "test_cost": int(test.iloc[1]["C"]),
        "validation_candidate_count": len(candidates), "sanity_checks_passed": True,
        "prediction_sha256": hashlib.sha256((out / "predictions.npz").read_bytes()).hexdigest(),
    })
    save_decision_changes(out, validation, test)
    print(f"\nValidation: t*={threshold!r}, ROC-AUC={auc!r}")
    print(markdown(validation))
    print("\nTest (поріг із threshold.json):")
    print(markdown(test))
    print("\nПриклади помилок:")
    print(markdown(examples))
    print("\nСтатистики лише наведених FP/FN:")
    print(markdown(stats))
    print("Precision не визначена за TP+FP=0; recall не визначений за TP+FN=0.")
    print("Результати сформовано зі збережених прогнозів без CSV та навчання.")


if __name__ == "__main__":
    main()
