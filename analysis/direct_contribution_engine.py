"""Exact identity/period contributions, including gap-only cross-window runs."""
import numpy as np
from numba import njit

THRESHOLDS=np.array([2,3,4,5,6,8,10],dtype=np.int64)


@njit(cache=True)
def observed_by_start(slots,labels,periods,width,offset,n_periods,n_labels,thresholds):
    result=np.zeros((2,n_periods,n_labels,len(thresholds)),dtype=np.int64)
    for definition in range(2):
        start=0
        for pos in range(1,len(slots)+1):
            stop=pos==len(slots)
            if not stop:
                stop=(slots[pos]!=slots[pos-1]+1 or labels[pos]!=labels[pos-1])
                if definition==0:
                    stop=stop or (slots[pos]-offset)//width!=(slots[pos-1]-offset)//width
            if stop:
                length=pos-start
                for j in range(len(thresholds)):
                    if length>=thresholds[j]:
                        result[definition,periods[start],labels[start],j]+=1
                start=pos
    return result


@njit(cache=True)
def falling_probabilities(counts,size,max_order):
    result=np.zeros((len(counts),max_order+1),dtype=np.float64)
    for label in range(len(counts)):
        result[label,0]=1.
        for order in range(1,max_order+1):
            if order>size or order>counts[label]:
                break
            result[label,order]=result[label,order-1]*(counts[label]-order+1)/(size-order+1)
    return result


@njit(cache=True)
def expected_by_start(slots,labels,periods,width,offset,n_periods,n_labels,thresholds):
    max_k=int(thresholds.max())
    n=len(slots)
    window_ids=(slots-offset)//width
    starts_list=[0]
    for i in range(1,n):
        if window_ids[i]!=window_ids[i-1]:
            starts_list.append(i)
    starts=np.array(starts_list)
    period_start=np.full(n_periods,n,dtype=np.int64)
    period_end=np.zeros(n_periods,dtype=np.int64)
    for i in range(n):
        p=periods[i]
        if period_start[p]==n:
            period_start[p]=i
        period_end[p]=i+1
    bounded=np.zeros((n_periods,n_labels,len(thresholds)),dtype=np.float64)
    correction=np.zeros_like(bounded)
    previous=np.zeros((n_labels,max_k+2),dtype=np.float64)
    previous_tail=0
    for w in range(len(starts)):
        start=starts[w]
        end=starts[w+1] if w+1<len(starts) else n
        counts=np.zeros(n_labels,dtype=np.int64)
        initial=np.zeros((n_periods,len(thresholds)),dtype=np.int64)
        interior=np.zeros_like(initial)
        segment_start=start
        for pos in range(start,end):
            counts[labels[pos]]+=1
            segment_end=pos+1==end
            if not segment_end:
                segment_end=slots[pos+1]!=slots[pos]+1
            if segment_end:
                length=pos+1-segment_start
                for j in range(len(thresholds)):
                    k=thresholds[j]
                    if length<k:
                        continue
                    initial[periods[segment_start],j]+=1
                    for p in range(n_periods):
                        lo=max(segment_start+1,period_start[p])
                        hi=min(pos+2-k,period_end[p])
                        if hi>lo:
                            interior[p,j]+=hi-lo
                segment_start=pos+1
        # The first and last uninterrupted segment lengths, capped at max_k.
        right=1
        while start+right<end and right<max_k and slots[start+right]==slots[start+right-1]+1:
            right+=1
        tail=1
        while end-tail-1>=start and tail<max_k and slots[end-tail]==slots[end-tail-1]+1:
            tail+=1
        probability=falling_probabilities(counts,end-start,max_k+1)
        for label in range(n_labels):
            if counts[label]==0:
                continue
            for j in range(len(thresholds)):
                k=thresholds[j]
                first=probability[label,k]
                internal=first-probability[label,k+1]
                for p in range(n_periods):
                    bounded[p,label,j]+=initial[p,j]*first+interior[p,j]*internal
        if start>0 and slots[start]==slots[start-1]+1:
            for label in range(n_labels):
                if probability[label,1]==0 or previous[label,1]==0:
                    continue
                for j in range(len(thresholds)):
                    k=thresholds[j]
                    if right>=k:
                        correction[periods[start],label,j]-=previous[label,1]*probability[label,k]
                    for left_count in range(1,k):
                        right_count=k-left_count
                        if left_count>previous_tail or right_count>right:
                            continue
                        i=start-left_count
                        prefix=previous[label,left_count]
                        if i>0 and slots[i]==slots[i-1]+1:
                            if window_ids[i]!=window_ids[i-1]:
                                raise AssertionError("Prefix predecessor escaped the previous window")
                            prefix-=previous[label,left_count+1]
                        correction[periods[i],label,j]+=prefix*probability[label,right_count]
        previous=probability
        previous_tail=tail
    return bounded,correction


def direct_contributions(slots,labels,periods,width,offset,thresholds=THRESHOLDS):
    slots=np.asarray(slots)
    labels=np.asarray(labels)
    periods=np.asarray(periods)
    thresholds=np.asarray(thresholds)
    for value in (slots,labels,periods,thresholds):
        if value.ndim!=1 or not np.issubdtype(value.dtype,np.integer):
            raise ValueError("One-dimensional integer arrays required")
    if len(slots)==0 or len(labels)!=len(slots) or len(periods)!=len(slots):
        raise ValueError("Nonempty aligned arrays required")
    if np.any(np.diff(slots)<=0) or np.any(labels<0) or np.any(periods<0) or np.any(np.diff(periods)<0):
        raise ValueError("Invalid support, identity codes or period order")
    if len(thresholds)==0 or np.any(thresholds<1) or np.any(np.diff(thresholds)<=0):
        raise ValueError("Invalid thresholds")
    if width<int(thresholds.max()) or not 0<=offset<width:
        raise ValueError("Windows must fit the longest prefix")
    n_periods=int(periods.max())+1
    n_labels=int(labels.max())+1
    slots=slots.astype(np.int64)
    labels=labels.astype(np.int64)
    periods=periods.astype(np.int64)
    observed=observed_by_start(slots,labels,periods,width,offset,n_periods,n_labels,thresholds)
    bounded,bridge=expected_by_start(slots,labels,periods,width,offset,n_periods,n_labels,thresholds)
    expected=np.stack([bounded,bounded+bridge])
    minimum=float(expected.min())
    if minimum < -1e-9:
        raise ArithmeticError("Negative expected run count")
    rounding_corrections=int(np.sum(expected<0))
    expected=np.maximum(expected,0.)
    if np.any((observed>0)&(expected<=0)):
        raise ArithmeticError("Observed event assigned zero probability")
    return observed,expected,bridge,dict(minimum_unclipped_expected=minimum,rounding_corrections=rounding_corrections)
