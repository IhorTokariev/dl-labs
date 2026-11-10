"""Регресійні перевірки захисту від підміни вибраної моделі."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import check_checkpoint_metadata, check_validation_logits
from common import CLASSES

BASE = Path(__file__).resolve().parents[1]


class AnalysisVerificationTests(unittest.TestCase):
    def setUp(self):
        self.saved = np.tile(np.array([3, 1, 0, -1, -2, -3], dtype=np.float32), (1498, 1))

    def test_small_device_roundoff_is_accepted(self):
        result = check_validation_logits(self.saved + np.float32(1e-5), self.saved)
        self.assertEqual(result["matching_predictions"], 1498)
        self.assertFalse(result["requires_local_run_weights"])

    def test_different_model_logits_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "не відповідають вибраному запуску"):
            check_validation_logits(self.saved + np.float32(0.1), self.saved)

    def test_changed_prediction_is_rejected_even_with_close_logits(self):
        saved = self.saved.copy()
        saved[0, :2] = [1, 1.00001]
        actual = saved.copy()
        actual[0, 0] = 1.00002
        with self.assertRaisesRegex(RuntimeError, "Передбачення validation"):
            check_validation_logits(actual, saved)

    def test_nonfinite_logits_are_rejected(self):
        actual = self.saved.copy()
        actual[0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            check_validation_logits(actual, self.saved)

    def test_incorrect_shape_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "1498×6"):
            check_validation_logits(self.saved[:-1], self.saved)

    def test_wrong_seed_is_rejected(self):
        checkpoint = {"state_dict": {}, "regime": "unfreeze", "seed": 2,
                      "classes": CLASSES, "img_size": 160}
        with self.assertRaisesRegex(RuntimeError, "правилу вибору"):
            check_checkpoint_metadata(checkpoint, {"regime": "unfreeze", "seed": 3})

    @unittest.skipUnless((BASE/"intel_image"/"seg_test"/"seg_test").is_dir(),
                         "Для інтеграційної перевірки потрібна локальна копія набору")
    def test_distributable_copy_runs_without_individual_weights(self):
        """Імітуємо файли після клонування; датасет читається з початкового шляху."""
        baseline = json.loads((BASE/"checks"/"analysis_report.json").read_text())
        with tempfile.TemporaryDirectory(prefix="lab3_portable_checkout_") as tmp:
            checkout = Path(tmp)
            for name in ("analyze.py", "common.py", "plots.py", "split.csv", "lab3_model.pt"):
                shutil.copy2(BASE/name, checkout/name)
            (checkout/"checks").mkdir()
            (checkout/"figures").mkdir()
            for directory in sorted((BASE/"runs").iterdir()):
                if not directory.is_dir():
                    continue
                target = checkout/"runs"/directory.name
                target.mkdir(parents=True)
                for name in ("run.json", "val_logits.npy"):
                    shutil.copy2(directory/name, target/name)
            self.assertEqual(list(checkout.rglob("weights.pt")), [])
            result = subprocess.run(
                [sys.executable, "-B", str(checkout/"analyze.py"),
                 "--data-root", str(BASE/"intel_image")],
                cwd=checkout, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn("ANALYZE OK", result.stdout)
            actual = json.loads((checkout/"checks"/"analysis_report.json").read_text())
            for key in ("runs", "effects", "selected", "test", "errors", "model_sha256", "split_sha256"):
                self.assertEqual(actual[key], baseline[key], key)
            self.assertFalse(actual["selection_verification"]["requires_local_run_weights"])


if __name__ == "__main__":
    unittest.main()
