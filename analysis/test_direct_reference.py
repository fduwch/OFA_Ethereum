import unittest
import numpy as np
from direct_probability_reference import reference_expectation
from test_direct_contributions import probability_reference


class ReferenceTests(unittest.TestCase):
    def test_randomized_reference_against_fraction_enumeration(self):
        for seed in range(100):
            rng=np.random.default_rng(seed+100)
            slots=np.sort(rng.choice(np.arange(30),size=15,replace=False))
            labels=rng.integers(0,3,size=len(slots))
            periods=np.searchsorted([10,20],slots,side="right")
            width=int(rng.integers(4,9))
            offset=int(rng.integers(width))
            k=np.array([2,3,4])
            expected=probability_reference(slots,labels,periods,width,offset,k)
            actual=reference_expectation(slots,labels,periods,width,offset,k)
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)

    def test_prefix_in_multiple_full_windows(self):
        slots=np.arange(21)
        labels=np.zeros(21,dtype=int)
        periods=np.searchsorted([3,13],slots,side="right")
        k=np.array([2,3,4])
        actual=reference_expectation(slots,labels,periods,4,1,k)
        expected=probability_reference(slots,labels,periods,4,1,k)
        np.testing.assert_allclose(actual,expected)


if __name__=="__main__":
    unittest.main()
