"""Finite enumeration and independent position-wise probability reference."""
from collections import Counter
from fractions import Fraction
import itertools
import unittest

import numpy as np

from direct_contribution_engine import direct_contributions,falling_probabilities


def observed_reference(slots,labels,periods,width,offset,thresholds):
    result=np.zeros((2,int(periods.max())+1,int(labels.max())+1,len(thresholds)),dtype=np.int64)
    for definition in range(2):
        cuts=np.r_[True,(np.diff(slots)!=1)|(np.diff(labels)!=0)]
        if definition==0:
            cuts |= np.r_[True,np.diff((slots-offset)//width)!=0]
        starts=np.flatnonzero(cuts)
        lengths=np.diff(np.r_[starts,len(slots)])
        for start,length in zip(starts,lengths):
            for j,k in enumerate(thresholds):
                if length>=k:
                    result[definition,periods[start],labels[start],j]+=1
    return result


def probability_reference(slots,labels,periods,width,offset,thresholds):
    ids=(slots-offset)//width
    distributions={wid:Counter(labels[ids==wid]) for wid in set(ids)}
    sizes=Counter(ids)
    result=np.zeros((2,int(periods.max())+1,int(labels.max())+1,len(thresholds)))
    def probability(indices,label):
        value=Fraction(1)
        for window,needed in Counter(ids[indices]).items():
            count=distributions[window][label]
            if needed>count:
                return Fraction(0)
            for j in range(needed):
                value*=Fraction(int(count-j),int(sizes[window]-j))
        return value
    for definition in range(2):
        for i in range(len(slots)):
            for j,k in enumerate(thresholds):
                stop=i+int(k)
                if stop>len(slots) or slots[stop-1]-slots[i]!=k-1:
                    continue
                if definition==0 and ids[i]!=ids[stop-1]:
                    continue
                predecessor=i>0 and slots[i-1]==slots[i]-1
                if definition==0:
                    predecessor=predecessor and ids[i-1]==ids[i]
                for label in range(result.shape[2]):
                    value=probability(np.arange(i,stop),label)
                    if predecessor:
                        value-=probability(np.arange(i-1,stop),label)
                    result[definition,periods[i],label,j]+=float(value)
    return result


def enumerated(slots,labels,periods,width,offset,thresholds):
    ids=(slots-offset)//width
    groups=[np.flatnonzero(ids==wid) for wid in sorted(set(ids))]
    choices=[sorted(set(itertools.permutations(labels[g]))) for g in groups]
    values=[]
    for combinations in itertools.product(*choices):
        shuffled=labels.copy()
        for group,assignment in zip(groups,combinations):
            shuffled[group]=assignment
        values.append(observed_reference(slots,shuffled,periods,width,offset,thresholds))
    return np.mean(values,axis=0)


class DirectTests(unittest.TestCase):
    def check_enumeration(self,slots,labels,periods,width,offset,thresholds=(2,3,4)):
        slots,labels,periods=map(lambda x:np.array(x,dtype=np.int64),(slots,labels,periods))
        thresholds=np.array(thresholds,dtype=np.int64)
        actual,expected,bridge,diagnostic=direct_contributions(slots,labels,periods,width,offset,thresholds)
        np.testing.assert_array_equal(actual,observed_reference(slots,labels,periods,width,offset,thresholds))
        np.testing.assert_allclose(expected,enumerated(slots,labels,periods,width,offset,thresholds),atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(expected,probability_reference(slots,labels,periods,width,offset,thresholds),atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(expected[1]-expected[0],bridge,atol=1e-12)

    def test_two_windows_crossing_period(self):
        self.check_enumeration(range(8),[0,0,0,1,0,0,1,1],[0,0,0,1,1,1,1,1],4,0)

    def test_gaps_and_three_windows(self):
        self.check_enumeration([0,1,3,4,5,7,8,9],[0,0,1,1,0,0,1,0],[0,0,0,0,1,1,1,1],4,0)

    def test_shifted_origin(self):
        self.check_enumeration(range(8),[0,0,0,1,0,0,1,1],[0,0,0,1,1,1,1,1],4,2)

    def test_singleton_windows_join_only_in_gap_definition(self):
        self.check_enumeration([2,3],[0,0],[0,1],3,0,(2,3))

    def test_one_identity_across_many_windows(self):
        slots=np.arange(20)
        periods=np.searchsorted([3,9,14],slots,side="right")
        self.check_enumeration(slots,np.zeros(20,dtype=int),periods,5,0,(2,3,4,5))

    def test_true_gaps_never_join(self):
        self.check_enumeration([0,1,4,5],[0,0,0,0],[0,0,1,1],4,0,(2,3,4))

    def test_period_boundary_does_not_cut_run(self):
        slots=np.arange(8)
        actual,expected,_,_=direct_contributions(slots,np.zeros(8,dtype=int),np.array([0,0,0,1,1,1,1,1]),8,0,np.array([2,4,8]))
        np.testing.assert_array_equal(actual[:,0,0],[ [1,1,1],[1,1,1] ])
        np.testing.assert_array_equal(actual[:,1,0],0)
        np.testing.assert_allclose(actual,expected)

    def test_boundary_changes_identity(self):
        self.check_enumeration(range(8),[0,0,0,0,1,1,1,1],[0,0,0,0,1,1,1,1],4,0)

    def test_absent_period_keeps_zero_contribution(self):
        self.check_enumeration(range(8),[0,0,0,1,0,0,1,1],[0,0,0,2,2,2,2,2],4,0)

    def test_randomized_fraction_reference(self):
        for seed in range(100):
            rng=np.random.default_rng(seed)
            slots=np.sort(rng.choice(np.arange(30),size=12,replace=False))
            labels=rng.integers(0,3,size=len(slots))
            periods=np.searchsorted([10,20],slots,side="right")
            width=int(rng.integers(4,9))
            offset=int(rng.integers(0,width))
            thresholds=np.array([2,3,4])
            observed,expected,_,_=direct_contributions(slots,labels,periods,width,offset,thresholds)
            np.testing.assert_array_equal(observed,observed_reference(slots,labels,periods,width,offset,thresholds))
            np.testing.assert_allclose(expected,probability_reference(slots,labels,periods,width,offset,thresholds),rtol=1e-11,atol=1e-12)

    def test_falling_probabilities_small_counts(self):
        actual=falling_probabilities(np.array([0,1,3]),4,5)
        np.testing.assert_allclose(actual[2],[1,.75,.5,.25,0,0])
        np.testing.assert_allclose(actual[1],[1,.25,0,0,0,0])

    def test_totals_match_frozen_implementation(self):
        import persistence_test as core
        import window_engine as engine
        rng=np.random.default_rng(222)
        slots=np.sort(rng.choice(np.arange(1200),size=1000,replace=False))
        labels=rng.integers(0,5,size=len(slots))
        periods=np.searchsorted([333,700],slots,side="right")
        pair=np.column_stack([labels,labels]).astype(np.int32)
        for width in (50,100,500):
            for offset in (0,width//2):
                actual,expected,_,_=direct_contributions(slots,labels,periods,width,offset)
                bounded,gaps,starts,ends=engine.layout(slots,width,offset)
                np.testing.assert_array_equal(actual[0].sum(axis=(0,1)),core.count_runs(pair,bounded)[0])
                np.testing.assert_array_equal(actual[1].sum(axis=(0,1)),core.count_runs(pair,gaps)[0])
                np.testing.assert_allclose(expected[0].sum(axis=(0,1)),engine.exact_window_expectation(pair,bounded,starts,ends)[0],rtol=1e-11,atol=1e-12)

    def test_gap_observations_identical_at_all_scales(self):
        slots=np.arange(200)
        labels=np.repeat(np.arange(10)%3,20)
        periods=np.searchsorted([70,130],slots,side="right")
        previous=None
        for width in (10,25,50,100):
            for offset in (0,width//2):
                actual,_,_,_=direct_contributions(slots,labels,periods,width,offset)
                if previous is not None:
                    np.testing.assert_array_equal(actual[1],previous)
                previous=actual[1]

    def test_invalid_support_rejected(self):
        with self.assertRaises(ValueError):
            direct_contributions(np.array([1,1]),np.array([0,0]),np.array([0,0]),50,0)

    def test_invalid_periods_rejected(self):
        with self.assertRaises(ValueError):
            direct_contributions(np.array([1,2]),np.array([0,0]),np.array([1,0]),50,0)

    def test_invalid_width_rejected(self):
        with self.assertRaises(ValueError):
            direct_contributions(np.array([1,2]),np.array([0,0]),np.array([0,0]),5,0)


if __name__=="__main__":
    unittest.main()
