"""Window layouts, joint permutations, and finite-population expectations."""

import numpy as np
from numba import njit

import persistence_test as core

WIDTHS = (10000,5000,2500,1000,500,250,100,50)
DEFINITIONS = ('window_bounded','gap_only')
THRESHOLDS = core.THRESHOLDS


def layout(slots, width, offset):
    if width < int(THRESHOLDS.max()) or not 0 <= offset < width:
        raise ValueError('Invalid width or origin')
    if len(slots)==0 or np.any(np.diff(slots)<=0):
        raise ValueError('Slots must be unique, sorted, and nonempty')
    ids=(slots-offset)//width
    boundary=np.r_[True,np.diff(ids)!=0]
    gaps=np.r_[True,np.diff(slots)!=1]
    starts=np.flatnonzero(boundary).astype(np.int64)
    ends=np.r_[starts[1:],len(slots)].astype(np.int64)
    return gaps|boundary,gaps,starts,ends


@njit(cache=True)
def shuffled_order(rng,size,starts,ends):
    order=np.arange(size,dtype=np.int64)
    for index in range(len(starts)):
        rng.shuffle(order[starts[index]:ends[index]])
    return order


@njit(cache=True)
def draw_batch(labels,bounded,gaps,starts,ends,repetitions,rng):
    output=np.empty((repetitions,2,2,7),dtype=np.int64)
    for repetition in range(repetitions):
        order=shuffled_order(rng,len(labels),starts,ends)
        output[repetition,0]=core.ordered_counts(labels,order,bounded)
        output[repetition,1]=core.ordered_counts(labels,order,gaps)
    return output


@njit(cache=True)
def exact_window_expectation(labels,bounded,starts,ends):
    expectation=np.zeros((2,7),dtype=np.float64)
    if len(labels)==0:
        return expectation
    counts=np.zeros(int(labels.max())+1,dtype=np.int64)
    touched=np.empty(int(np.max(ends-starts)),dtype=np.int64)
    for window in range(len(starts)):
        start,end=starts[window],ends[window]
        size=end-start
        initial=np.zeros(7,dtype=np.int64)
        interior=np.zeros(7,dtype=np.int64)
        segment_start=start
        for position in range(start+1,end+1):
            if position==end or bounded[position]:
                length=position-segment_start
                for k in range(7):
                    if length>=THRESHOLDS[k]:
                        initial[k]+=1
                        interior[k]+=length-THRESHOLDS[k]
                segment_start=position
        for unit in range(2):
            unique=0
            for position in range(start,end):
                label=labels[position,unit]
                if counts[label]==0:
                    touched[unique]=label
                    unique+=1
                counts[label]+=1
            for k in range(7):
                threshold=THRESHOLDS[k]
                if initial[k]==0:
                    continue
                a,b=0.,0.
                for index in range(unique):
                    number=counts[touched[index]]
                    if number<threshold:
                        continue
                    probability=1.
                    for j in range(threshold):
                        probability*= (number-j)/(size-j)
                    a+=probability
                    if size>threshold:
                        b+=probability*(size-number)/(size-threshold)
                expectation[unit,k]+=initial[k]*a+interior[k]*b
            for index in range(unique):
                counts[touched[index]]=0
    return expectation
