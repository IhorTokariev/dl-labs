"""Перевірки двох команд викладача з окремою текою результатів.

У тесті маршрутизації навчання замінене тестовою функцією: вона копіює
еталонні артефакти лише в тимчасову теку. Нові навчальні результати не створюються.
Команда аналізу виконується повністю на CPU.
"""
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
import train
from common import RUNS

BASE = Path(__file__).resolve().parents[1]


class TeacherCommandsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lab3_teacher_commands_")
        self.addCleanup(self.temp.cleanup)
        self.checkout = (Path(self.temp.name)/"checkout").resolve()
        self.checkout.mkdir()
        for name in ("analyze.py", "common.py", "plots.py", "split.csv"):
            shutil.copy2(BASE/name, self.checkout/name)
        for source in sorted((BASE/"runs").glob("*/run.json")):
            target = self.checkout/"runs"/source.parent.name
            target.mkdir(parents=True)
            shutil.copy2(source, target/"run.json")
            shutil.copy2(source.parent/"val_logits.npy", target/"val_logits.npy")
        (self.checkout/"checks").mkdir()
        shutil.copy2(BASE/"checks/checks_report.json", self.checkout/"checks/checks_report.json")
        self.before = self.published_hashes()

    def published_hashes(self):
        return {str(p.relative_to(self.checkout)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.checkout.rglob("*") if p.is_file() and "reproduced" not in p.parts}

    def training_patches(self):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(train, "BASE", self.checkout))
        stack.enter_context(mock.patch.object(train, "configure_weight_cache"))
        stack.enter_context(mock.patch.object(train, "choose_device", return_value=torch.device("cpu")))
        stack.enter_context(mock.patch.object(train, "environment", return_value={"device": "cpu"}))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def test_existing_results_stop_before_any_checks_or_writes(self):
        with self.training_patches(), mock.patch.object(train,"build_split") as split, \
                mock.patch.object(train,"run_checks") as checks, mock.patch.object(train,"train_run") as run:
            with self.assertRaisesRegex(RuntimeError, "--output-dir"):
                train.main(["--data-root", str(BASE/"intel_image")])
            split.assert_not_called()
            checks.assert_not_called()
            run.assert_not_called()
        self.assertEqual(self.published_hashes(), self.before)

    def test_result_directory_inside_dataset_is_rejected(self):
        for entrypoint in (train, analyze):
            with self.subTest(command=entrypoint.__name__), self.training_patches():
                with self.assertRaisesRegex(RuntimeError,"всередині датасету"):
                    entrypoint.main(["--data-root", str(BASE/"intel_image"),
                                     "--output-dir", str(BASE/"intel_image"/"results")])
        self.assertEqual(self.published_hashes(), self.before)

    def test_resume_reuses_checks_and_skips_only_complete_runs_in_selected_directory(self):
        output = self.checkout/"reproduced"
        output.mkdir()
        shutil.copy2(BASE/"split.csv",output/"split.csv")
        (output/"checks").mkdir()
        (output/"figures").mkdir()
        source_hash = train.source_hash()
        checks_fixture = {"status":"OK","split_sha256":train.sha256(output/"split.csv"),
                          "source_sha256":source_hash}
        (output/"checks/checks_report.json").write_text(json.dumps(checks_fixture))
        incomplete = RUNS[-1]
        for regime,seed in RUNS:
            name = f"{regime}_seed{seed}"
            target = output/"runs"/name
            target.mkdir(parents=True)
            shutil.copy2(BASE/"runs"/name/"val_logits.npy",target/"val_logits.npy")
            # Контрольні файли тесту, не ваги експериментальної моделі.
            (target/"weights.pt").write_bytes(f"test fixture {name}".encode())
            run = json.loads((BASE/"runs"/name/"run.json").read_text())
            run.update(source_sha256=source_hash,weights_sha256=train.sha256(target/"weights.pt"))
            (target/"run.json").write_text(json.dumps(run))
        (output/"runs"/f"{incomplete[0]}_seed{incomplete[1]}"/"weights.pt").unlink()
        with self.training_patches(), mock.patch.object(train,"run_checks") as checks, \
                mock.patch.object(train,"train_run") as run, mock.patch.object(train,"save_selection"):
            train.main(["--data-root",str(BASE/"intel_image"),"--output-dir","reproduced","--resume"])
        checks.assert_not_called()
        run.assert_called_once()
        self.assertEqual(run.call_args.args[:2],incomplete)
        self.assertEqual(run.call_args.args[3],output)
        self.assertEqual(self.published_hashes(),self.before)

    @unittest.skipUnless((BASE/"intel_image/seg_test/seg_test").is_dir(),
                         "Для повної перевірки аналізу потрібен локальний набір")
    def test_fresh_training_routes_all_seven_runs_and_analysis_to_same_directory(self):
        output = self.checkout/"reproduced"
        self.assertFalse(output.exists())

        def reference_training_artifacts(regime, seed, data_root, destination, device):
            # Тестова заміна дорогого навчання, не експериментальний запуск.
            target = destination/"runs"/f"{regime}_seed{seed}"
            target.mkdir(parents=True,exist_ok=True)
            source = BASE/"runs"/target.name
            for name in ("run.json", "val_logits.npy"):
                shutil.copy2(source/name, target/name)

        def reference_checks(data_root, destination, device, split_report):
            (destination/"checks").mkdir()
            (destination/"figures").mkdir()
            shutil.copy2(BASE/"checks/checks_report.json", destination/"checks/checks_report.json")

        def reference_model(runs, destination):
            shutil.copy2(BASE/"lab3_model.pt", destination/"lab3_model.pt")

        with self.training_patches(), \
                mock.patch.object(train,"run_checks",side_effect=reference_checks) as checks, \
                mock.patch.object(train,"train_run",side_effect=reference_training_artifacts) as run, \
                mock.patch.object(train,"save_selection",side_effect=reference_model):
            train.main(["--data-root",str(BASE/"intel_image"),"--output-dir","reproduced"])
        checks.assert_called_once()
        self.assertEqual(checks.call_args.args[1], output)
        self.assertEqual([(call.args[0],call.args[1]) for call in run.call_args_list], RUNS)
        self.assertTrue(all(call.args[3] == output for call in run.call_args_list))
        self.assertEqual(len(list((output/"runs").glob("*/run.json"))),7)
        self.assertEqual(list(output.rglob("weights.pt")),[])

        # Пошкоджуємо лише копію старого поділу: аналіз повинен читати reproduced/.
        (self.checkout/"split.csv").write_text("Навмисно некоректна тестова копія\n")
        result = subprocess.run([sys.executable,"-B",str(self.checkout/"analyze.py"),
                                 "--data-root",str(BASE/"intel_image"),"--output-dir","reproduced"],
                                cwd=Path(self.temp.name),capture_output=True,text=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn("ANALYZE OK",result.stdout)
        actual = json.loads((output/"checks/analysis_report.json").read_text())
        baseline = json.loads((BASE/"checks/analysis_report.json").read_text())
        for key in ("runs","effects","selected","test","errors","model_sha256","split_sha256"):
            self.assertEqual(actual[key],baseline[key],key)
        self.assertFalse((self.checkout/"checks/analysis_report.json").exists())
        # Повертаємо нашу тестову зміну й перевіряємо збереження старих артефактів.
        shutil.copy2(BASE/"split.csv",self.checkout/"split.csv")
        self.assertEqual(self.published_hashes(),self.before)


if __name__ == "__main__":
    unittest.main()
