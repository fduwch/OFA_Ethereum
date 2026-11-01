"""Independent finite-sample checks for the partition decomposition."""
import itertools
import unittest
import numpy as np
from analyze_partition_decomposition import contingency, decompose


class PartitionTests(unittest.TestCase):
    def test_identity(self):
        s, *_ = decompose(np.diag([1, 3, 5]))
        self.assertEqual((s["merge_hhi"], s["split_hhi"]), (0, 0))

    def test_merge(self):
        s, *_ = decompose(np.array([[2], [3]]))
        self.assertEqual(s["merge_ordered_pairs"], 12)
        self.assertEqual(s["split_ordered_pairs"], 0)
        self.assertAlmostEqual(s["delta_hhi"], .48)

    def test_split(self):
        s, *_ = decompose(np.array([[2, 3]]))
        self.assertEqual(s["split_ordered_pairs"], 12)
        self.assertAlmostEqual(s["delta_hhi"], -.48)

    def test_crossing_cancel(self):
        s, *_ = decompose(np.array([[1, 1], [1, 1]]))
        self.assertEqual((s["merge_ordered_pairs"], s["split_ordered_pairs"]), (4, 4))
        self.assertEqual(s["delta_hhi"], 0)

    def test_payload_weight_not_identity_count(self):
        s, *_ = decompose(np.array([[9], [1]]))
        self.assertAlmostEqual(s["source_hhi"], .82)
        self.assertAlmostEqual(s["delta_hhi"], .18)

    def test_all_binary_sequences_against_direct_ordered_pairs(self):
        for a in itertools.product(range(2), repeat=4):
            for b in itertools.product(range(2), repeat=4):
                s, *_ = decompose(contingency(np.array(a), np.array(b), 2, 2))
                merge = sum(a[i] != a[j] and b[i] == b[j] for i in range(4) for j in range(4))
                split = sum(a[i] == a[j] and b[i] != b[j] for i in range(4) for j in range(4))
                self.assertEqual(s["merge_ordered_pairs"], merge)
                self.assertEqual(s["split_ordered_pairs"], split)

    def test_transpose_reverses_sign(self):
        tab = np.array([[1, 4, 5], [0, 3, 2]])
        a, *_ = decompose(tab)
        b, *_ = decompose(tab.T)
        self.assertEqual(a["merge_ordered_pairs"], b["split_ordered_pairs"])
        self.assertEqual(a["delta_hhi"], -b["delta_hhi"])

    def test_duplicate_sample_invariant(self):
        tab = np.array([[1, 4, 5], [0, 3, 2]])
        a, *_ = decompose(tab)
        b, *_ = decompose(tab * 10)
        for metric in ("source_hhi", "target_hhi", "merge_hhi", "split_hhi", "delta_hhi"):
            self.assertEqual(a[metric], b[metric])

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            decompose(np.zeros((2, 2), dtype=int))

    def test_invalid_counts_rejected(self):
        for tab in (np.array([1, 2]), np.array([[1, -1]]), np.array([[1., 2.]])):
            with self.assertRaises(ValueError):
                decompose(tab)

    def test_unaligned_labels_rejected(self):
        with self.assertRaises(ValueError):
            contingency(np.array([0, 1]), np.array([0]), 2, 2)

    def test_invalid_code_rejected(self):
        for a in (np.array([-1]), np.array([2]), np.array([0.5])):
            with self.assertRaises(ValueError):
                contingency(a, np.array([0]), 2, 2)


if __name__ == "__main__":
    unittest.main()
