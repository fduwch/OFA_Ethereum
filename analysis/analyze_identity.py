"""External-key validation and concentration sensitivity on the full support."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
from identity_evidence import load_evidence, resolve_key, sha256, concentration_from_counts

INPUT_SHA = '06bac99220ff55814d91d43a7d2082f6409968bd98078f4dec946bab485c7c78'
VARIANTS = [
    ('official_historical_service','official_historical','service',False),
    ('official_historical_network','official_historical','network',False),
    ('historical_service','historical','service',False),
    ('historical_network','historical','network',False),
    ('forward_historical_service','historical','service',True),
    ('forward_historical_network','historical','network',True),
    ('official_all_service','official_all','service',False),
    ('all_service','all','service',False),
    ('all_network','all','network',False),
]


def create_mapping(keys, ledger):
    bykey = defaultdict(list)
    for row in ledger:
        bykey[row['builder_pubkey']].append(row)
    rows=[]
    for key in keys:
        for variant, mode, level, forward in VARIANTS:
            name, date, basis, sources = resolve_key(bykey[key],mode,level)
            rows.append(dict(builder_pubkey=key,variant=variant,name=name,
                             documentary_date=date,basis=basis,sources=';'.join(sources)))
    return pd.DataFrame(rows)


def attach_variant(frame,mapping,variant,forward=False):
    m=mapping[mapping.variant.eq(variant)].set_index('builder_pubkey')
    names=frame.builder_pubkey.map(m.name).fillna('')
    known=names.ne('')
    if forward:
        source_dates=frame.builder_pubkey.map(m.documentary_date).fillna('')
        dates=pd.to_datetime(source_dates,format='ISO8601',utc=True,errors='coerce')
        if (source_dates.ne('') & dates.isna()).any():
            raise ValueError('Unparsed nonempty documentary date')
        known &= dates.notna() & frame.slot_start_date_time.ge(dates)
    identities=('key:'+frame.builder_pubkey).to_numpy(dtype=object)
    identities[known]=('external:'+names[known]).to_numpy()
    return identities,known.to_numpy()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'completion.json').exists():
        raise FileExistsError(a.out)
    digest=sha256(a.input)
    assert digest==INPUT_SHA
    sources,ledger=load_evidence(a.root)
    ld=pd.DataFrame(ledger)
    ld.to_csv(a.out/'evidence_ledger.csv',index=False)
    (a.out/'source_manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
    f=pd.read_parquet(a.input,columns=['slot','slot_start_date_time','builder_pubkey','extra_data_tag'])
    f=f.sort_values('slot').reset_index(drop=True)
    f['builder_pubkey']=f.builder_pubkey.str.casefold()
    f['extra_data_tag']=f.extra_data_tag.fillna('')
    f['slot_start_date_time']=pd.to_datetime(f.slot_start_date_time,utc=True)
    f['month']=f.slot_start_date_time.dt.strftime('%Y-%m')
    assert len(f)==2332997 and not f.slot.duplicated().any()
    keys=sorted(f.builder_pubkey.unique())
    mapping=create_mapping(keys,ledger)
    keyinfo=f.groupby('builder_pubkey').agg(slots=('slot','size'),first_seen=('slot_start_date_time','min'),last_seen=('slot_start_date_time','max')).reset_index()
    mapping=mapping.merge(keyinfo,on='builder_pubkey',validate='many_to_one')
    mapping.to_csv(a.out/'key_mapping.csv',index=False)
    f.groupby(['builder_pubkey','extra_data_tag'],dropna=False).size().rename('slots').reset_index().to_csv(a.out/'key_tag_counts.csv',index=False)
    source_coverage=[]
    for sid,g in ld.groupby('source_id'):
        matched=f.builder_pubkey.isin(g.loc[g.status.eq('mapped'),'builder_pubkey'])
        source_coverage.append(dict(source_id=sid,evidence_rows=len(g),valid_keys=g.loc[g.status.eq('mapped'),'builder_pubkey'].nunique(),matched_keys=f.loc[matched,'builder_pubkey'].nunique(),matched_slots=int(matched.sum()),slot_coverage=float(matched.mean()),invalid_rows=int(g.status.eq('invalid_key').sum())))
    pd.DataFrame(source_coverage).to_csv(a.out/'source_coverage.csv',index=False)
    # Source-to-source label disagreement is retained, including non-entity flags.
    conflicts=[]
    for key,g in ld[ld.builder_pubkey.isin(keys)].groupby('builder_pubkey'):
        latest=g[~g.evidence_class.str.endswith('_earlier')]
        if latest.network.replace('',np.nan).nunique()>1 or latest.raw_label.str.casefold().isin(['imposter','default key']).any():
            for row in latest.to_dict('records'):
                conflicts.append(row)
    pd.DataFrame(conflicts,columns=ld.columns).to_csv(a.out/'source_disagreements.csv',index=False)
    counts,coverage,groups,runrows=[],[],[],[]
    variants=[('raw_key',('key:'+f.builder_pubkey).to_numpy(),np.zeros(len(f),bool)),
              ('slot_tag',np.where(f.extra_data_tag.ne(''),'tag:'+f.extra_data_tag,'key:'+f.builder_pubkey),f.extra_data_tag.ne('').to_numpy())]
    for name,mode,level,forward in VARIANTS:
        labels,known=attach_variant(f,mapping,name,forward)
        variants.append((name,labels,known))
    # Shared run counter and expectation implementation, copied byte-for-byte.
    import persistence_test as core
    from window_engine import layout,exact_window_expectation
    slots=f.slot.to_numpy(dtype=np.int64)
    persisted=pd.DataFrame({'slot':slots})
    for variant,labels,known in variants:
        work=f[['month','builder_pubkey','extra_data_tag']].copy()
        work['identity']=labels
        work['known']=known
        for month,g in work.groupby('month',sort=True):
            vc=g.identity.value_counts()
            counts.append(dict(variant=variant,month=month,**concentration_from_counts(vc)))
            coverage.append(dict(variant=variant,month=month,slots=len(g),matched_slots=int(g.known.sum()),slot_coverage=float(g.known.mean()),keys=g.builder_pubkey.nunique(),matched_keys=g.loc[g.known,'builder_pubkey'].nunique()))
        counts.append(dict(variant=variant,month='all',**concentration_from_counts(work.identity.value_counts())))
        coverage.append(dict(variant=variant,month='all',slots=len(work),matched_slots=int(known.sum()),slot_coverage=float(known.mean()),keys=len(keys),matched_keys=work.loc[known,'builder_pubkey'].nunique()))
        vc=work.groupby(['identity','extra_data_tag'],sort=True).size().rename('slots').reset_index()
        vc.insert(0,'variant',variant)
        groups.append(vc)
        codes=pd.factorize(labels,sort=True)[0].astype(np.int32)
        if variant in ('raw_key','slot_tag','historical_service','historical_network'):
            persisted[variant]=codes
        pair=np.column_stack([codes,codes])
        for width in (10000,5000,2500,1000,500,250,100,50):
            bounded,gaps,starts,ends=layout(slots,width,0)
            obs=core.count_runs(pair,bounded)[0]
            gap_obs=core.count_runs(pair,gaps)[0]
            expected=exact_window_expectation(pair,bounded,starts,ends)[0]
            for i,k in enumerate(core.THRESHOLDS):
                runrows.append(dict(variant=variant,width_slots=width,threshold=int(k),observed=int(obs[i]),gap_only_observed=int(gap_obs[i]),exact_expectation=float(expected[i]),ratio=float(obs[i]/expected[i]) if expected[i]>0 else None))
        print(variant, 'coverage',float(known.mean()),flush=True)
    pd.DataFrame(counts).to_csv(a.out/'concentration.csv',index=False)
    conc=pd.DataFrame(counts)
    conc.loc[conc.month.ne('all')].groupby('variant')[['hhi','cr3','cr5']].median().reset_index().to_csv(a.out/'monthly_median_concentration.csv',index=False)
    pd.DataFrame(coverage).to_csv(a.out/'coverage.csv',index=False)
    pd.concat(groups,ignore_index=True).to_csv(a.out/'identity_tag_contingency.csv',index=False)
    pd.DataFrame(runrows).to_csv(a.out/'analytic_runs.csv',index=False)
    persisted.to_parquet(a.out/'analysis_labels.parquet',index=False)
    # All external map versions, including absent mappings, stay available.
    old=mapping[mapping.variant.eq('historical_service')].set_index('builder_pubkey')
    new=mapping[mapping.variant.eq('all_service')].set_index('builder_pubkey')
    delta=old[['name','basis','slots']].rename(columns={'name':'historical_name','basis':'historical_basis'}).join(new[['name','basis']].rename(columns={'name':'updated_name','basis':'updated_basis'}))
    delta[delta.historical_name.ne(delta.updated_name)].reset_index().to_csv(a.out/'registry_version_changes.csv',index=False)
    completion=dict(status='computed',rows=len(f),keys=len(keys),min_time=str(f.slot_start_date_time.min()),max_time=str(f.slot_start_date_time.max()),input_sha256=digest,
                    finished_at=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,
                    outputs={p.name:sha256(p) for p in sorted(a.out.iterdir()) if p.is_file()},
                    code={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    (a.out/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')


if __name__=='__main__':
    main()
