"""Independent positive-probability reference, classifying every candidate start.

Unlike the producer, this does not derive gap-only means by correcting bounded
means. It directly classifies each valid prefix and computes nonnegative terms.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def probabilities(labels,start,end,n_labels,max_k):
    counts=np.zeros(n_labels,dtype=np.int64)
    for i in range(start,end):
        counts[labels[i]]+=1
    table=np.zeros((n_labels,max_k+2),dtype=np.float64)
    for c in range(n_labels):
        table[c,0]=1.
        for k in range(1,max_k+2):
            if k>counts[c] or k>end-start:
                break
            value=1.
            for r in range(k):
                value*=float(counts[c]-r)/float(end-start-r)
            table[c,k]=value
    return table


@njit(cache=True)
def reference_expectation(slots,labels,periods,width,offset,thresholds):
    n=len(slots)
    n_labels=int(labels.max())+1
    n_periods=int(periods.max())+1
    max_k=int(thresholds.max())
    ids=(slots-offset)//width
    starts_list=[0]
    for i in range(1,n):
        if ids[i]!=ids[i-1]:
            starts_list.append(i)
    starts=np.array(starts_list)
    output=np.zeros((2,n_periods,n_labels,len(thresholds)),dtype=np.float64)
    previous=np.zeros((n_labels,max_k+2),dtype=np.float64)
    for w in range(len(starts)):
        start=starts[w]
        end=starts[w+1] if w+1<len(starts) else n
        next_end=starts[w+2] if w+2<len(starts) else n
        current=probabilities(labels,start,end,n_labels,max_k)
        following=probabilities(labels,end,next_end,n_labels,max_k)
        kinds=np.zeros((3,n_periods,len(thresholds)),dtype=np.int64)
        for i in range(start,end):
            p=periods[i]
            predecessor=i>0 and slots[i-1]==slots[i]-1
            for j in range(len(thresholds)):
                k=thresholds[j]
                last=i+k-1
                if last>=n or slots[last]-slots[i]!=k-1:
                    continue
                if last<end:
                    kind=0 if not predecessor else (1 if i>start else 2)
                    kinds[kind,p,j]+=1
                else:
                    # A gap-free prefix shorter than the window can cross only
                    # the immediately following conditioning window.
                    if last>=next_end:
                        raise AssertionError("Unexpected third window")
                    left=end-i
                    right=k-left
                    if predecessor and i==start:
                        raise AssertionError("Crossing prefix predecessor in another window")
                    for c in range(n_labels):
                        first=current[c,left]
                        if predecessor:
                            first-=current[c,left+1]
                        output[1,p,c,j]+=first*following[c,right]
        for p in range(n_periods):
            for c in range(n_labels):
                for j in range(len(thresholds)):
                    k=thresholds[j]
                    natural=kinds[0,p,j]*current[c,k]
                    interior=kinds[1,p,j]*(current[c,k]-current[c,k+1])
                    boundary=kinds[2,p,j]*current[c,k]
                    output[0,p,c,j]+=natural+interior+boundary
                    output[1,p,c,j]+=natural+interior+boundary*(1-previous[c,1])
        previous=current
    return output
