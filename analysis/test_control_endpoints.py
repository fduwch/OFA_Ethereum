from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from summarize_control_endpoints import endpoint_summaries, load_control_run, rate, validate_grid, verify_original_global


def sample():
    config = dict(models=["stationary_iid","sticky_020"], outer_samples_per_model=2,
                  widths=[50], definitions=["window_bounded","gap_only"],
                  units=["historical_service","historical_network"], thresholds=[2,8,10], multiplicity_settings=32)
    frame = pd.DataFrame([dict(model=m,replication=r,width_slots=w,offset_slots=o,definition=d,unit=u,
                               threshold=k,observed=10,permutation_mean=10.,family_adjusted_p=1.,across_settings_p=1.)
                          for m in config["models"] for r in range(2) for w in config["widths"]
                          for o in (0,w//2) for d in config["definitions"] for u in config["units"] for k in config["thresholds"]])
    return frame, config


def reject(frame, mask, observed=12, p=.05):
    frame.loc[mask,"observed"] = observed
    frame.loc[mask,"across_settings_p"] = p
    frame.loc[mask,"family_adjusted_p"] = p/32


class EndpointChecks(unittest.TestCase):
    def test_complete_grid(self):
        frame, config = sample()
        validate_grid(frame, config)

    def test_duplicate_grid(self):
        frame, config = sample()
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            validate_grid(pd.concat([frame,frame.iloc[:1]]),config)

    def test_missing_coordinate(self):
        frame, config = sample()
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            validate_grid(frame.iloc[1:],config)

    def test_wrong_origin(self):
        frame, config = sample()
        frame.loc[0,"offset_slots"] = 1
        with self.assertRaisesRegex(ValueError,"Unexpected origin"):
            validate_grid(frame,config)

    def test_invalid_p(self):
        for value in (np.nan, np.inf, -.1, 1.01):
            frame, config = sample()
            frame.loc[0,"across_settings_p"] = value
            with self.assertRaises(ValueError):
                validate_grid(frame,config)

    def test_multiplicity_cannot_change(self):
        frame, config = sample()
        frame.loc[0,"family_adjusted_p"] = .001
        with self.assertRaises(AssertionError):
            validate_grid(frame,config)

    def test_wilson_pointwise_interval(self):
        result = rate([True,True,False,False])
        self.assertEqual(result["outer_samples"],4)
        self.assertAlmostEqual(result["wilson_95_low"],.15003898915214947)
        self.assertAlmostEqual(result["wilson_95_high"],.8499610108478506)

    def test_family_union_not_additive_directions(self):
        frame, config = sample()
        same = frame.model.eq("stationary_iid") & frame.replication.eq(0) & frame.offset_slots.eq(0) & frame.definition.eq("gap_only")
        reject(frame,same & frame.threshold.eq(8),12)
        reject(frame,same & frame.threshold.eq(10),8)
        validate_grid(frame,config)
        settings, totals = endpoint_summaries(frame)
        selected = settings[settings.model.eq("stationary_iid") & settings.offset_slots.eq(0)
                            & settings.definition.eq("gap_only") & settings.endpoint.eq("long_runs_8_10")]
        self.assertTrue(selected.rejections.eq(1).all())
        self.assertTrue(selected.outer_samples.eq(2).all())
        self.assertTrue(selected.rejection_rate.eq(.5).all())
        self.assertTrue(totals[totals.model.eq("stationary_iid")].outer_samples.eq(2).all())

    def test_short_rejection_not_long_power(self):
        frame, _ = sample()
        reject(frame,frame.model.eq("stationary_iid") & frame.replication.eq(0) & frame.threshold.eq(2))
        _, totals = endpoint_summaries(frame)
        selected = totals[totals.model.eq("stationary_iid") & totals.direction.eq("positive")].set_index("endpoint")
        self.assertEqual(selected.loc["all_coordinates","rejections"],1)
        self.assertEqual(selected.loc["long_runs_8_10","rejections"],0)

    def test_global_union_not_sum_across_scales_or_origins(self):
        frame, _ = sample()
        reject(frame,frame.model.eq("stationary_iid") & frame.replication.eq(0))
        _, totals = endpoint_summaries(frame)
        selected = totals[totals.model.eq("stationary_iid") & totals.direction.eq("two_sided")]
        self.assertTrue(selected.rejections.eq(1).all())
        self.assertTrue(selected.outer_samples.eq(2).all())

    def test_rejected_zero_direction_fails(self):
        frame, _ = sample()
        reject(frame,frame.index == 0,10)
        with self.assertRaisesRegex(ValueError,"zero-direction"):
            endpoint_summaries(frame)

    def test_original_global_reproduction(self):
        frame, _ = sample()
        reject(frame,frame.model.eq("stationary_iid") & frame.replication.eq(0))
        _, totals = endpoint_summaries(frame)
        original = pd.DataFrame([dict(model="stationary_iid",**rate([True,False])),
                                 dict(model="sticky_020",**rate([False,False]))])
        self.assertEqual(verify_original_global(totals,original),10)
        original.loc[0,"rejections"] = 2
        with self.assertRaises(AssertionError):
            verify_original_global(totals,original)

    def test_formal_gate_requires_completed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_control_run(Path(tmp))


if __name__ == "__main__":
    unittest.main()
