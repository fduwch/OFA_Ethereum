"""Unit checks for paired nonstationary Markov-copy controls."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_controls as base
import run_joint_controls as joint


class JointControlTests(unittest.TestCase):
    def setUp(self):
        self.slots = np.r_[np.arange(10, 110), np.arange(10000, 10100),
                           np.arange(20000, 20100), np.arange(30000, 30100)]
        rng = np.random.default_rng(5)
        service = rng.integers(0, 8, len(self.slots), dtype=np.int32)
        self.labels = np.column_stack((service, service//2)).astype(np.int32)
        self.early, self.late = base.make_profiles(self.slots)

    def generate(self, model, rho, rep=2):
        return joint.generate_at_rho(self.labels, self.slots, self.early, self.late, model, rep, rho)

    def test_zero_matches_original_generator_exactly(self):
        for model in ("smooth_iid", "step_iid"):
            for rep in (0, 2, 99):
                np.testing.assert_array_equal(self.generate(model, 0, rep),
                    base.generate(self.labels, self.slots, self.early, self.late, model, rep))

    def test_independent_scalar_copy(self):
        for model in ("smooth_iid", "step_iid"):
            fresh = base.generate(self.labels, self.slots, self.early, self.late, model, 2)
            for rho in (0, .005, .02, .05, .5, 1):
                rng = np.random.default_rng([base.SEED, 2, base.MODELS.index(model), 2])
                uniforms = rng.random(len(fresh))
                expected = fresh.copy()
                for i in range(1, len(fresh)):
                    if self.slots[i] == self.slots[i-1]+1 and uniforms[i] < rho:
                        expected[i] = expected[i-1]
                np.testing.assert_array_equal(self.generate(model, rho), expected)

    def test_gap_reset_at_complete_copy(self):
        mask = np.r_[True, np.diff(self.slots) != 1]
        for model in ("smooth_iid", "step_iid"):
            fresh = self.generate(model, 0)
            result = self.generate(model, 1)
            np.testing.assert_array_equal(result[mask], fresh[mask])
            np.testing.assert_array_equal(result[~mask], result[np.flatnonzero(~mask)-1])

    def test_copy_masks_nested(self):
        for model in ("smooth_iid", "step_iid"):
            previous = np.zeros(len(self.slots), dtype=bool)
            for rho in (0, .005, .02, .05, 1):
                current = joint.copy_mask(self.slots, model, 2, rho)
                self.assertFalse(np.any(previous & ~current))
                previous = current

    def test_identity_pairs_preserved(self):
        for model in joint.MODELS:
            result = joint.generate(self.labels, self.slots, self.early, self.late, model, 2)
            np.testing.assert_array_equal(result[:, 1], result[:, 0]//2)
            self.assertTrue(result.flags.c_contiguous)

    def test_task_order_does_not_change_samples(self):
        before = {m: joint.generate(self.labels, self.slots, self.early, self.late, m, 3)
                  for m in joint.MODELS}
        for model in reversed(joint.MODELS):
            np.testing.assert_array_equal(before[model], joint.generate(
                self.labels, self.slots, self.early, self.late, model, 3))

    def test_no_original_module_mutation(self):
        self.assertEqual(base.MODELS, ("stationary_iid", "smooth_iid", "step_iid",
                                     "sticky_005", "sticky_020", "sticky_050"))
        self.assertEqual(base.STAGES["formal"], (200, 1999))

    def test_invalid_generator_specs(self):
        for rho in (-.01, 1.01):
            with self.assertRaises(ValueError):
                self.generate("smooth_iid", rho)
        with self.assertRaises(ValueError):
            joint.copy_mask(np.array([2, 1]), "smooth_iid", 0, .05)
        with self.assertRaises(ValueError):
            joint.copy_mask(self.slots, "stationary_iid", 0, .05)
        with self.assertRaises(ValueError):
            joint.specification("smooth_r999")

    def test_paired_events_counts(self):
        rows = []
        for model in ("smooth_r005", "smooth_iid"):
            for rep in range(4):
                reject = rep in ([0, 1] if model.endswith("005") else [0, 2])
                rows.append(dict(model=model, replication=rep, width_slots=50, offset_slots=0,
                    definition="gap_only", threshold=8, across_settings_p=.01 if reject else 1.,
                    observed=12., permutation_mean=10.))
        frame = pd.DataFrame(rows)
        with patch.object(joint, "MODELS", ("smooth_r005",)):
            result = joint.paired_event_counts(frame[frame.model.eq("smooth_r005")],
                                               frame[frame.model.eq("smooth_iid")])
        positive = result[result.direction.eq("positive")]
        self.assertEqual(len(positive), 2)
        for row in positive.itertuples():
            self.assertEqual((row.both_reject, row.joint_only, row.baseline_only, row.neither_reject),
                             (1, 1, 1, 1))
            self.assertEqual(row.paired_rejection_difference, 0.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
