"""Generator, random-stream, and nested-inference tests."""
import multiprocessing as mp
import unittest

import numpy as np
import pandas as pd

import run_controls as c


def numpy_counts(labels, breaks):
    output = np.zeros((2,7), dtype=np.int64)
    for u in range(2):
        changes = breaks | np.r_[True, labels[1:,u] != labels[:-1,u]]
        lengths = np.diff(np.r_[np.flatnonzero(changes), len(labels)])
        for k, threshold in enumerate(c.core.THRESHOLDS):
            output[u,k] = np.sum(lengths >= threshold)
    return output


def generated_hash(rep):
    slots = np.arange(40000, dtype=np.int64)
    labels = np.column_stack((slots % 3, (slots % 3)//2)).astype(np.int32)
    early, late = c.make_profiles(slots)
    return c.array_hash(c.generate(labels,slots,early,late,"smooth_iid",rep))


class ControlsTests(unittest.TestCase):
    def setUp(self):
        self.slots = np.arange(40000,dtype=np.int64)
        self.labels = np.column_stack((self.slots%3,(self.slots%3)//2)).astype(np.int32)
        self.early,self.late = c.make_profiles(self.slots)

    def test_four_stage_profiles(self):
        np.testing.assert_array_equal(self.early,np.arange(10000))
        np.testing.assert_array_equal(self.late,np.arange(30000,40000))

    def test_profiles_use_absolute_slot_windows(self):
        slots=np.array([10001,10003,30002,40001,80001],dtype=np.int64)
        early,late=c.make_profiles(slots)
        np.testing.assert_array_equal(early,[0,1])
        np.testing.assert_array_equal(late,[4])

    def test_invalid_profile_support(self):
        for slots in (np.array([],dtype=int),np.array([1,1]),np.arange(10),np.array([3,2,1])):
            with self.assertRaises(ValueError):
                c.make_profiles(slots)

    def test_model_support_and_joint_identity(self):
        for model in c.MODELS:
            generated=c.generate(self.labels,self.slots,self.early,self.late,model,0)
            self.assertEqual(generated.shape,self.labels.shape)
            self.assertEqual(generated.dtype,np.int32)
            self.assertTrue(np.all(generated[:,1] == generated[:,0]//2))

    def test_reproducible_outer_samples(self):
        for model in c.MODELS:
            a=c.generate(self.labels,self.slots,self.early,self.late,model,3)
            b=c.generate(self.labels,self.slots,self.early,self.late,model,3)
            np.testing.assert_array_equal(a,b)
            self.assertNotEqual(c.array_hash(a),c.array_hash(c.generate(self.labels,self.slots,self.early,self.late,model,4)))

    def test_copy_anchors(self):
        labels=np.arange(12).reshape(6,2)
        result=c.copy_at_anchors(labels,np.array([1,0,0,1,0,1],bool))
        np.testing.assert_array_equal(result,labels[[0,0,0,3,3,5]])

    def test_sticky_resets_at_every_gap(self):
        slots=np.arange(2000,dtype=np.int64)*2
        labels=np.column_stack((np.arange(2000),np.arange(2000)%3)).astype(np.int32)
        old=c.RHOS["sticky_050"]
        try:
            c.RHOS["sticky_050"]=1.
            actual=c.generate(labels,slots,np.arange(10),np.arange(10),"sticky_050",4)
            rng=np.random.default_rng([c.SEED,0,c.MODELS.index("sticky_050"),4])
            expected=labels[rng.integers(0,len(labels),size=len(labels))]
            np.testing.assert_array_equal(actual,expected)
        finally:
            c.RHOS["sticky_050"]=old

    def test_sticky_known_transition_probability(self):
        slots=np.arange(300000,dtype=np.int64)
        labels=np.column_stack((slots%3,(slots%3)//2)).astype(np.int32)
        early,late=c.make_profiles(slots)
        for model,rho in c.RHOS.items():
            sample=c.generate(labels,slots,early,late,model,0)
            same=np.mean(sample[1:,0]==sample[:-1,0])
            self.assertLess(abs(same-(rho+(1-rho)/3)),.005)
            self.assertTrue(np.all(np.abs(np.bincount(sample[:,0])/len(sample)-1/3)<.005))

    def test_stationary_marginal_and_independence(self):
        sample=c.generate(self.labels,self.slots,self.early,self.late,"stationary_iid",0)
        self.assertTrue(np.all(np.abs(np.bincount(sample[:,0])/len(sample)-1/3)<.01))
        self.assertLess(abs(np.mean(sample[1:,0]==sample[:-1,0])-1/3),.01)

    def test_step_profile_selection(self):
        labels=np.zeros((40000,2),dtype=np.int32)
        labels[30000:]=1
        sample=c.generate(labels,self.slots,self.early,self.late,"step_iid",0)
        expected=(self.slots+137)//2500%2
        np.testing.assert_array_equal(sample[:,0],expected)

    def test_smooth_profile_probability(self):
        slots=np.arange(400000,dtype=np.int64)
        labels=np.zeros((len(slots),2),dtype=np.int32)
        labels[-100000:]=1
        early,late=c.make_profiles(slots)
        sample=c.generate(labels,slots,early,late,"smooth_iid",0)
        p=(1+np.sin(2*np.pi*slots/10000))/2
        for lo,hi in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
            take=(p>=lo)&(p<hi)
            self.assertLess(abs(sample[take,0].mean()-p[take].mean()),.01)

    def test_inner_seeds_and_draw_prefix(self):
        labels=self.labels[:60]
        slots=np.delete(np.arange(64),[7,8,45,46])
        bounded,gaps,starts,ends=c.engine.layout(slots,50,25)
        first=c.engine.draw_batch(labels,bounded,gaps,starts,ends,19,c.inner_rng("stationary_iid",0,50,25))
        longer=c.engine.draw_batch(labels,bounded,gaps,starts,ends,31,c.inner_rng("stationary_iid",0,50,25))
        np.testing.assert_array_equal(first,longer[:19])
        self.assertNotEqual(str(c.inner_rng("stationary_iid",0,50,25).bit_generator.state),
                            str(c.inner_rng("stationary_iid",0,50,0).bit_generator.state))

    def test_joint_counts_against_numpy_on_identical_orders(self):
        labels=np.column_stack(([0]*10+[1]*6+[0]*12+[2]*12,[0]*16+[1]*24)).astype(np.int32)
        slots=np.delete(np.arange(44),[8,9,31,32])
        bounded,gaps,starts,ends=c.engine.layout(slots,50,25)
        rng1=c.inner_rng("step_iid",3,50,25)
        rng2=c.inner_rng("step_iid",3,50,25)
        actual=c.engine.draw_batch(labels,bounded,gaps,starts,ends,29,rng1)
        reference=[]
        for _ in range(29):
            order=c.engine.shuffled_order(rng2,len(labels),starts,ends)
            reference.append(np.stack([numpy_counts(labels[order],bounded),numpy_counts(labels[order],gaps)]))
        np.testing.assert_array_equal(actual,reference)

    def test_original_inference_reused_exactly(self):
        rng=np.random.default_rng(2)
        draws=rng.poisson(10,size=(199,2,2,7))
        observed=rng.poisson(12,size=(2,2,7))
        records=c.inference_rows(observed,draws,"stationary_iid",0,500,0)
        self.assertEqual(len(records),28)
        for d in range(2):
            expected=c.core.infer_from_counts(observed[d],draws[:,d])
            for u in range(2):
                for k in range(7):
                    row=records[d*14+u*7+k]
                    self.assertEqual(row["family_adjusted_p"],expected["family_adjusted_p"][u,k])
                    self.assertEqual(row["across_settings_p"],min(1.,32*expected["family_adjusted_p"][u,k]))

    def test_worker_schedule_invariance(self):
        serial=[generated_hash(x) for x in [0,1,2]]
        with mp.get_context("fork").Pool(2) as pool:
            parallel=pool.map(generated_hash,[0,1,2])
        self.assertEqual(serial,parallel)

    def test_rate_uses_actual_per_dataset_pvalues(self):
        values=np.array([.01,.9,.8,.02])
        result=c.rate_record(values)
        self.assertEqual(result["rejections"],2)
        self.assertEqual(result["rejection_rate"],.5)

    def test_invalid_generation_request(self):
        with self.assertRaises(ValueError):
            c.generate(self.labels,self.slots,self.early,self.late,"unknown",0)


if __name__ == "__main__":
    unittest.main()
