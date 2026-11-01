"""Enumeration and implementation checks for short-window analysis."""

import itertools
import unittest

import numpy as np

import persistence_test as core
import window_engine as engine
import run_windows as runner


class WindowTests(unittest.TestCase):
    def test_layout_offsets_gaps_and_edges(self):
        slots=np.array([48,49,50,51,53,74,75,76,99,100,101])
        bounded,gaps,starts,ends=engine.layout(slots,50,0)
        np.testing.assert_array_equal(starts,[0,2,9])
        self.assertTrue(bounded[2] and not gaps[2])
        self.assertTrue(gaps[4])
        b,g,s,e=engine.layout(slots,50,25)
        np.testing.assert_array_equal(s,[0,6])
        np.testing.assert_array_equal(g,gaps)

    def test_joint_permutation_preserves_each_window(self):
        labels=np.column_stack([np.arange(40)%5,np.arange(40)%3]).astype(np.int32)
        slots=np.arange(40)+90
        bounded,gaps,starts,ends=engine.layout(slots,10,5)
        rng=np.random.default_rng(185)
        for _ in range(100):
            order=engine.shuffled_order(rng,len(labels),starts,ends)
            for start,end in zip(starts,ends):
                self.assertEqual(sorted(order[start:end]),list(range(start,end)))
                self.assertEqual(sorted(map(tuple,labels[order[start:end]])),sorted(map(tuple,labels[start:end])))

    def test_compiled_batch_matches_shared_count_function(self):
        rng=np.random.default_rng(254)
        labels=rng.integers(0,5,size=(73,2),dtype=np.int32)
        slots=np.arange(73)+101
        slots[37:]+=4
        bounded,gaps,starts,ends=engine.layout(slots,20,10)
        actual=engine.draw_batch(labels,bounded,gaps,starts,ends,99,np.random.default_rng(123))
        rng=np.random.default_rng(123)
        for i in range(99):
            order=engine.shuffled_order(rng,len(labels),starts,ends)
            np.testing.assert_array_equal(actual[i,0],core.count_runs(labels[order],bounded))
            np.testing.assert_array_equal(actual[i,1],core.count_runs(labels[order],gaps))

    def test_gap_only_statistic_invariant(self):
        labels=np.column_stack([np.arange(203)//12,np.arange(203)//19]).astype(np.int32)
        slots=np.arange(203)+99
        slots[90:]+=5
        reference=None
        for width in (50,100,250,10000):
            for offset in (0,width//2):
                bounded,gaps,starts,ends=engine.layout(slots,width,offset)
                counts=core.count_runs(labels,gaps)
                if reference is None:
                    reference=counts
                np.testing.assert_array_equal(counts,reference)

    def test_exact_expectation_one_window_enumeration(self):
        for gaps in ([True]+[False]*5,[True,False,False,True,False,False]):
            gaps=np.array(gaps)
            labels=np.array([[0,0]]*3+[[1,0]]*2+[[2,1]],dtype=np.int32)
            starts,ends=np.array([0]),np.array([6])
            exact=engine.exact_window_expectation(labels,gaps,starts,ends)
            all_counts=np.array([core.count_runs(labels[list(order)],gaps)
                                 for order in itertools.permutations(range(6))])
            np.testing.assert_allclose(exact,all_counts.mean(axis=0),atol=1e-13)

    def test_exact_expectation_multiple_windows(self):
        labels=np.array([[0,0],[0,0],[1,1],[0,0],[1,1],[1,1]],dtype=np.int32)
        starts,ends=np.array([0,3]),np.array([3,6])
        bounded=np.array([1,0,0,1,0,0],dtype=bool)
        all_counts=[]
        for a in itertools.permutations(range(3)):
            for b in itertools.permutations(range(3,6)):
                all_counts.append(core.count_runs(labels[list(a+b)],bounded))
        exact=engine.exact_window_expectation(labels,bounded,starts,ends)
        np.testing.assert_allclose(exact,np.mean(all_counts,axis=0),atol=1e-13)

    def test_exact_expectation_threshold_equals_size(self):
        labels=np.zeros((10,2),dtype=np.int32)
        bounded=np.r_[True,np.zeros(9,dtype=bool)]
        exact=engine.exact_window_expectation(labels,bounded,np.array([0]),np.array([10]))
        np.testing.assert_array_equal(exact,np.ones((2,7)))

    def test_exact_expectation_singletons(self):
        labels=np.zeros((20,2),dtype=np.int32)
        exact=engine.exact_window_expectation(labels,np.ones(20,dtype=bool),np.arange(20),np.arange(1,21))
        np.testing.assert_array_equal(exact,np.zeros((2,7)))

    def test_original_layout_equivalence(self):
        import pandas as pd
        frame=pd.DataFrame(dict(slot=[9998,9999,10000,10001,10003],
                                builder_pubkey=['a','a','a','b','b'],proxy=['x','x','x','x','x']))
        labels,breaks,starts,ends=core.arrays_from_frame(frame,10000)
        b,g,s,e=engine.layout(frame.slot.to_numpy(),10000,0)
        np.testing.assert_array_equal(b,breaks)
        np.testing.assert_array_equal(s,starts)
        np.testing.assert_array_equal(e,ends)

    def test_fixed_batch_seeds_and_coverage(self):
        for count in (2,99,10000):
            a=runner.tasks_for(count,20261023,50,25)
            b=runner.tasks_for(count,20261023,50,25)
            self.assertEqual(sum(n for seed,n in a),count)
            for first,second in zip(a,b):
                np.testing.assert_array_equal(first[0],second[0])
                self.assertEqual(first[1],second[1])


if __name__=='__main__':
    unittest.main(verbosity=2)
