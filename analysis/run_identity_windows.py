"""Local-window inference for two externally registered identity units."""

import argparse
import json
import multiprocessing as mp
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

import persistence_test as core
import window_engine as engine

_ARRAYS=None


def worker(task):
    seed_words,repetitions=task
    return engine.draw_batch(*_ARRAYS,repetitions,np.random.default_rng(seed_words))


def tasks_for(permutations,seed,width,offset):
    batches=min(128,permutations)
    sizes=np.full(batches,permutations//batches,dtype=int)
    sizes[:permutations%batches]+=1
    children=np.random.SeedSequence([seed,width,offset]).spawn(batches)
    return [(child.generate_state(4),int(size)) for child,size in zip(children,sizes)]


def configuration_rows(labels,bounded,gaps,simulations,width,offset,alpha,families,exact,columns):
    records=[]
    order=np.arange(len(labels),dtype=np.int64)
    for d,definition in enumerate(engine.DEFINITIONS):
        observed=core.ordered_counts(labels,order,bounded if d==0 else gaps)
        result=core.infer_from_counts(observed,simulations[:,d],alpha)
        for row in core.result_rows(result,width_slots=width,offset_slots=offset,definition=definition):
            u=core.UNITS.index(row['unit'])
            k=list(core.THRESHOLDS).index(row['threshold'])
            mean=row['permutation_mean']
            row['window_minutes']=width/5
            row['observed_expected_ratio']=row['observed']/mean if mean else np.nan
            row['excess_runs']=row['observed']-mean
            row['across_settings_p']=min(1.,row['family_adjusted_p']*families)
            row['exact_conditional_mean']=float(exact[u,k]) if d==0 else np.nan
            row['mc_mean_se']=row['permutation_sd']/np.sqrt(len(simulations))
            row['mc_mean_zerror']=((mean-exact[u,k])/row['mc_mean_se']
                                   if d==0 and row['mc_mean_se']>0 else np.nan)
            row['unit']=columns[u]
            records.append(row)
    return records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--payloads',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--columns',nargs=2,default=['historical_service','historical_network'])
    parser.add_argument('--windows',nargs='+',type=int,default=list(engine.WIDTHS))
    parser.add_argument('--permutations',type=int,default=10000)
    parser.add_argument('--workers',type=int,default=48)
    parser.add_argument('--seed',type=int,default=20261024)
    parser.add_argument('--alpha',type=float,default=.05)
    args=parser.parse_args()
    if args.permutations<2 or args.workers<1 or not 0<args.alpha<1:
        parser.error('Invalid permutations, workers, or alpha')
    if len(set(args.windows))!=len(args.windows) or any(w<10 or w%2 for w in args.windows):
        parser.error('Window widths must be distinct even integers of at least ten')
    args.out_dir.mkdir(parents=True,exist_ok=False)
    start_time=time.time()
    input_hash=core.sha256(args.payloads)
    frame=pd.read_parquet(args.payloads,columns=['slot',*args.columns]).sort_values('slot')
    if frame.empty or frame.isna().any().any() or frame.slot.duplicated().any():
        raise ValueError('Invalid frozen identity input')
    labels=frame[args.columns].to_numpy(dtype=np.int32)
    slots=frame.slot.to_numpy(dtype=np.int64)
    specs=[(width,offset) for width in args.windows for offset in (0,width//2)]
    metadata={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    metadata.update(rows=len(frame),slot_min=int(slots[0]),slot_max=int(slots[-1]),
                    input_sha256=input_hash,settings=specs,definitions=engine.DEFINITIONS,
                    settings_family_count=len(specs)*len(engine.DEFINITIONS),
                    source_sha256={name:core.sha256(Path(__file__).parent/name) for name in
                                   ('persistence_test.py','window_engine.py','run_identity_windows.py')},
                    python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,status='running')
    (args.out_dir/'configuration.json').write_text(json.dumps(metadata,indent=2)+'\n')
    warm_labels=labels[:16].copy()
    warm_breaks=np.r_[True,np.zeros(15,dtype=bool)]
    engine.draw_batch(warm_labels,warm_breaks,warm_breaks,np.array([0]),np.array([16]),2,np.random.default_rng(0))
    engine.exact_window_expectation(warm_labels,warm_breaks,np.array([0]),np.array([16]))
    records=[]
    coverage=[]
    global _ARRAYS
    for index,(width,offset) in enumerate(specs,1):
        case_start=time.time()
        bounded,gaps,starts,ends=engine.layout(slots,width,offset)
        _ARRAYS=(labels,bounded,gaps,starts,ends)
        tasks=tasks_for(args.permutations,args.seed,width,offset)
        with mp.get_context('fork').Pool(min(args.workers,len(tasks))) as pool:
            simulations=np.concatenate(list(pool.imap(worker,tasks)),axis=0)
        exact=engine.exact_window_expectation(labels,bounded,starts,ends)
        case_records=configuration_rows(labels,bounded,gaps,simulations,width,offset,args.alpha,
                                        metadata['settings_family_count'],exact,args.columns)
        records.extend(case_records)
        case_dir=args.out_dir/f'w{width}_o{offset}'
        case_dir.mkdir()
        np.save(case_dir/'permutation_counts.npy',simulations.astype(np.int32))
        pd.DataFrame(case_records).to_csv(case_dir/'inference.csv',index=False)
        sizes=ends-starts
        coverage.append(dict(width_slots=width,offset_slots=offset,rows=len(labels),windows=len(starts),
                             minimum_window_rows=int(sizes.min()),median_window_rows=float(np.median(sizes)),
                             p05_window_rows=float(np.quantile(sizes,.05)),
                             windows_under_10_rows=int((sizes<10).sum()),
                             gap_breaks=int(gaps.sum()),bounded_breaks=int(bounded.sum()),
                             elapsed_seconds=time.time()-case_start))
        pd.DataFrame(records).to_csv(args.out_dir/'all_inference.csv',index=False)
        pd.DataFrame(coverage).to_csv(args.out_dir/'window_coverage.csv',index=False)
        selected=pd.DataFrame(case_records)
        selected=selected[selected.threshold.isin([8,10]) & selected.definition.eq('window_bounded')]
        print(f'completed {index}/{len(specs)} width={width} offset={offset} seconds={time.time()-case_start:.1f}',flush=True)
        print(selected[['definition','unit','threshold','observed','observed_expected_ratio','family_adjusted_p','across_settings_p']].to_string(index=False),flush=True)
    if core.sha256(args.payloads)!=input_hash:
        raise RuntimeError('Input hash changed during execution')
    metadata.update(status='complete',elapsed_seconds=time.time()-start_time,
                    output_sha256={str(path.relative_to(args.out_dir)):core.sha256(path)
                                   for path in sorted(args.out_dir.rglob('*')) if path.is_file() and path.name!='configuration.json'})
    (args.out_dir/'completion.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'complete: {time.time()-start_time:.1f}s',flush=True)


if __name__=='__main__':
    main()
