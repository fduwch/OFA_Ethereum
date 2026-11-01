"""Deterministic, enumeration, and implementation-agreement tests."""

import itertools
import statistics
import unittest

import numpy as np

import persistence_test as core


def scalar_counts(labels, breaks):
    result = np.zeros((2, 7), dtype=int)
    for unit in range(2):
        lengths = []
        current = 0
        previous = None
        for i, value in enumerate(labels[:, unit]):
            if i == 0 or breaks[i] or value != previous:
                if current:
                    lengths.append(current)
                current = 1
            else:
                current += 1
            previous = value
        if current:
            lengths.append(current)
        result[unit] = [sum(length >= k for length in lengths) for k in core.THRESHOLDS]
    return result


class TestPersistence(unittest.TestCase):
    def fixture(self):
        labels = np.array([[0, 3], [0, 3], [1, 3], [2, 4]] * 8, dtype=np.int32)
        breaks = np.zeros(len(labels), dtype=bool)
        breaks[[0, 7, 16, 21]] = True
        return labels, breaks, np.array([0, 16]), np.array([16, 32])

    def test_hand_count_and_gap(self):
        labels = np.array([1,1,1,1,2,2,2,1,1])
        breaks = np.array([1,0,0,1,0,0,0,0,0], dtype=bool)
        result = core.count_runs(np.column_stack([labels,labels]), breaks)
        np.testing.assert_array_equal(result[0], [3,2,0,0,0,0,0])

    def test_empty_and_long_runs(self):
        result = core.count_runs(np.zeros((0,2), dtype=np.int32), np.zeros(0, dtype=bool))
        self.assertEqual(result.sum(), 0)
        breaks = np.zeros(200, dtype=bool)
        breaks[0] = True
        np.testing.assert_array_equal(core.count_runs(np.zeros((200,2), dtype=np.int32), breaks), np.ones((2,7)))

    def test_fast_counts_match_scalar_oracle(self):
        rng = np.random.default_rng(910)
        for size in (1,2,9,23,101,1000):
            for _ in range(20):
                labels = rng.integers(0, 4, size=(size,2), dtype=np.int32)
                breaks = rng.random(size) < .1
                breaks[0] = True
                order = rng.permutation(size)
                np.testing.assert_array_equal(core.ordered_counts(labels, order, breaks), scalar_counts(labels[order], breaks))

    def test_shuffle_windows_and_pairs(self):
        labels, breaks, starts, ends = self.fixture()
        rng = np.random.default_rng(82)
        for _ in range(100):
            order = core.permutation_order(rng, len(labels), starts, ends)
            for start,end in zip(starts,ends):
                self.assertEqual(sorted(order[start:end]), list(range(start,end)))
                self.assertEqual(sorted(map(tuple, labels[order[start:end]])), sorted(map(tuple, labels[start:end])))

    def test_permutation_distribution_matches_scalar_reference(self):
        labels, breaks, starts, ends = self.fixture()
        actual = core.permutation_counts(labels, breaks, starts, ends, 99, np.random.default_rng(777))
        rng = np.random.default_rng(777)
        expected = np.array([scalar_counts(labels[core.permutation_order(rng, len(labels), starts, ends)], breaks)
                             for _ in range(99)])
        np.testing.assert_array_equal(actual, expected)

    def test_complete_test_equals_same_count_matrix(self):
        labels, breaks, starts, ends = self.fixture()
        result = core.complete_test(labels, breaks, starts, ends, 99, np.random.default_rng(827))
        sims = core.permutation_counts(labels, breaks, starts, ends, 99, np.random.default_rng(827))
        expected = core.infer_from_counts(scalar_counts(labels, breaks), sims)
        for name in core.P_COLUMNS:
            np.testing.assert_array_equal(result[name], expected[name])

    def test_constant_statistics_are_not_rejected(self):
        observed = np.ones((2,7), dtype=int)
        result = core.infer_from_counts(observed, np.ones((99,2,7), dtype=int))
        for name in core.P_COLUMNS:
            np.testing.assert_array_equal(result[name], np.ones((2,7)))
        self.assertFalse(result['family_reject'])
        self.assertTrue(result['zero_pooled_sd'].all())

    def test_zero_reference_variance_observed_differs(self):
        observed = np.zeros((2,7), dtype=int)
        observed[0,6] = 1
        result = core.infer_from_counts(observed, np.zeros((99,2,7), dtype=int))
        self.assertEqual(result['family_adjusted_p'][0,6], .01)
        self.assertEqual(result['raw_upper_p'][0,6], .01)
        self.assertEqual(result['raw_two_sided_p'][0,6], .02)
        self.assertTrue(result['zero_permutation_sd'][0,6])
        self.assertFalse(result['zero_pooled_sd'][0,6])

    def test_inclusive_ties_and_minimum_p(self):
        observed = np.ones((2,7), dtype=int)*3
        simulations = np.broadcast_to(np.array([0,1,3,3])[:,None,None], (4,2,7))
        result = core.infer_from_counts(observed, simulations)
        self.assertEqual(result['raw_upper_p'][0,0], 3/5)
        self.assertTrue(np.all(result['family_adjusted_p'] >= 1/5))

    def test_full_family_dominates_unit_family(self):
        rng = np.random.default_rng(541)
        for _ in range(40):
            result = core.infer_from_counts(rng.poisson(3,size=(2,7)), rng.poisson(3,size=(99,2,7)))
            self.assertTrue(np.all(result['family_adjusted_p'] >= result['unit_adjusted_p']))
            for field in core.P_COLUMNS:
                self.assertTrue(np.all(np.isfinite(result[field])))
                self.assertTrue(np.all(result[field] >= .01))
                self.assertTrue(np.all(result[field] <= 1))

    def test_reference_row_order_invariance(self):
        rng = np.random.default_rng(214)
        simulations = rng.poisson(4,size=(99,2,7))
        observed = rng.poisson(4,size=(2,7))
        a = core.infer_from_counts(observed, simulations)
        b = core.infer_from_counts(observed, simulations[rng.permutation(99)])
        for field in core.P_COLUMNS:
            np.testing.assert_array_equal(a[field], b[field])

    def test_exact_enumeration_family_error(self):
        vectors = []
        breaks = np.array([True]+[False]*7)
        for positions in itertools.combinations(range(8), 4):
            labels = np.zeros(8, dtype=np.int32)
            labels[list(positions)] = 1
            vectors.append(scalar_counts(np.column_stack([labels,labels]), breaks))
        vectors = np.array(vectors)
        for alpha in (.05, .1, .2):
            errors = 0
            for index in range(len(vectors)):
                result = core.infer_from_counts(vectors[index], np.delete(vectors,index,axis=0), alpha)
                errors += result['family_reject']
            self.assertLessEqual(errors/len(vectors), alpha)

    def test_sticky_resets_at_gaps(self):
        labels, breaks, starts, ends = self.fixture()
        sampled = core.synthetic_labels(labels, breaks, starts, ends, np.random.default_rng(64), 'iid_sticky', 1.)
        self.assertTrue(np.all(sampled[1:][~breaks[1:]] == sampled[:-1][~breaks[1:]]))
        for start,end in zip(starts,ends):
            self.assertTrue(set(map(tuple,sampled[start:end])).issubset(set(map(tuple,labels[start:end]))))

    def test_iid_sampling_preserves_empirical_pair_support(self):
        labels, breaks, starts, ends = self.fixture()
        for seed in range(20):
            sampled = core.synthetic_labels(labels, breaks, starts, ends, np.random.default_rng(seed), 'iid_sticky', 0.)
            for start,end in zip(starts,ends):
                self.assertTrue(set(map(tuple,sampled[start:end])).issubset(set(map(tuple,labels[start:end]))))

    def test_invalid_counts_raise(self):
        for bad in (np.full((2,7),np.nan), np.full((2,7),-1)):
            with self.assertRaises(ValueError):
                core.infer_from_counts(bad,np.ones((99,2,7)))

    def test_scalar_maximum_rank_oracle(self):
        rng = np.random.default_rng(665)
        pooled = rng.poisson(5, size=(40,2,7))
        standardized = np.zeros_like(pooled, dtype=float)
        for unit in range(2):
            for k in range(7):
                values = [float(x) for x in pooled[:,unit,k]]
                center = statistics.mean(values)
                sd = statistics.stdev(values)
                standardized[:,unit,k] = [(x-center)/sd if sd else 0. for x in values]
        maxima = [max(abs(x) for x in row.ravel()) for row in standardized]
        expected = np.zeros((2,7))
        for unit in range(2):
            for k in range(7):
                expected[unit,k] = sum(value >= abs(standardized[0,unit,k]) for value in maxima)/len(maxima)
        actual = core.infer_from_counts(pooled[0],pooled[1:])
        np.testing.assert_array_equal(actual['family_adjusted_p'],expected)

    def test_exact_enumeration_distinct_identity_units(self):
        breaks = np.array([True]+[False]*7)
        vectors = []
        for zeros in itertools.combinations(range(8),4):
            remainder = sorted(set(range(8))-set(zeros))
            for ones in itertools.combinations(remainder,2):
                key = np.full(8,2,dtype=np.int32)
                key[list(zeros)] = 0
                key[list(ones)] = 1
                proxy = (key == 2).astype(np.int32)
                vectors.append(scalar_counts(np.column_stack([key,proxy]),breaks))
        vectors = np.array(vectors)
        self.assertEqual(len(vectors),420)
        for alpha in (.05,.1,.2):
            errors = sum(core.infer_from_counts(vectors[i],np.delete(vectors,i,axis=0),alpha)['family_reject']
                         for i in range(len(vectors)))
            self.assertLessEqual(errors/len(vectors),alpha)

    def test_identity_recoding_invariance(self):
        labels,breaks,starts,ends = self.fixture()
        recoded = labels*np.array([101,79])+np.array([99,2])
        a = core.complete_test(labels,breaks,starts,ends,99,np.random.default_rng(43))
        b = core.complete_test(recoded,breaks,starts,ends,99,np.random.default_rng(43))
        for field in core.P_COLUMNS:
            np.testing.assert_array_equal(a[field],b[field])


if __name__ == '__main__':
    unittest.main(verbosity=2)
