"""Command-line smoke tests on small deterministic inputs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from direct_contribution_engine import direct_contributions, THRESHOLDS

ROOT = Path(__file__).resolve().parents[1]


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        slots = np.concatenate([np.arange(32) + i * 10000 for i in range(4)])
        key = np.arange(len(slots)) // 6 % 3
        self.frame = pd.DataFrame(dict(slot=slots, builder_pubkey=key.astype(str),
                                      extra_data_tag=(key // 2).astype(str),
                                      historical_service=key, historical_network=key // 2))
        self.input = self.base / "input.parquet"
        self.frame.to_parquet(self.input, index=False)

    def run_command(self, path, *args):
        result = subprocess.run([sys.executable, str(ROOT / path), *map(str, args)],
                                cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_calibration_entry(self):
        out = self.base / "calibration"
        self.run_command("scripts/calibrate_persistence_null.py", "--payloads", self.input,
                         "--out-dir", out, "--null-replications", 2, "--power-replications", 0,
                         "--profile-windows", 4, "--permutations", 19, "--workers", 2)
        rows = pd.read_csv(out / "outer_tests_sorted.csv")
        self.assertEqual(len(rows), 2 * 2 * 14)
        self.assertTrue(rows.family_adjusted_p.between(.05, 1).all())

    def test_full_sample_entry(self):
        out = self.base / "real"
        self.run_command("scripts/adjust_persistence_threshold_family.py", "--payloads", self.input,
                         "--out-dir", out, "--permutations", 19, "--workers", 2)
        self.assertEqual(len(pd.read_csv(out / "observed_inference.csv")), 14)

    def test_window_entries(self):
        for script, name in [("scripts/compute_xatu_common_support.py", "windows"),
                             ("analysis/run_identity_windows.py", "external")]:
            out = self.base / name
            self.run_command(script, "--payloads", self.input, "--out-dir", out,
                             "--windows", 50, "--permutations", 19, "--workers", 2)
            rows = pd.read_csv(out / "all_inference.csv")
            self.assertEqual(len(rows), 56)
            np.testing.assert_allclose(rows.across_settings_p,
                                       np.minimum(1, rows.family_adjusted_p * 4))

    def test_contribution_entry_matches_engine(self):
        out = self.base / "contributions"
        self.run_command("scripts/analyze_persistence_stability.py", "--payloads", self.input,
                         "--columns", "historical_service", "--windows", 50, "--out-dir", out)
        rows = pd.read_csv(out / "contributions.csv")
        periods = np.repeat(np.arange(4), 32)
        for offset in (0, 25):
            observed, expected, _, _ = direct_contributions(self.frame.slot.to_numpy(),
                self.frame.historical_service.to_numpy(), periods, 50, offset)
            for d, definition in enumerate(("window_bounded", "gap_only")):
                group = rows[(rows.offset_slots == offset) & (rows.definition == definition)]
                for label in range(3):
                    for k, threshold in enumerate(THRESHOLDS):
                        total = group[(group.period == 0) & (group.identity == label) &
                                      (group.threshold == threshold)].iloc[0]
                        self.assertEqual(total.observed, observed[d, :, label, k].sum())
                        self.assertAlmostEqual(total.exact_expectation, expected[d, :, label, k].sum())
        self.assertEqual(json.loads((out / "completion.json").read_text())["rows"], 128)

    def test_help_entries(self):
        for script in ("analysis/run_controls.py", "analysis/analyze_identity.py",
                       "analysis/analyze_partition_decomposition.py", "analysis/summarize_history.py",
                       "scripts/compute_xatu_builder_persistence.py",
                       "scripts/compute_xatu_extra_data_sensitivity.py"):
            self.run_command(script, "--help")


if __name__ == "__main__":
    unittest.main()
