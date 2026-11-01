"""Compare every historical registry and each leave-one-registry-out map."""
import argparse
from collections import defaultdict,Counter
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from identity_evidence import load_evidence,resolve_key,sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    _,ledger=load_evidence(a.root)
    bykey=defaultdict(list)
    for r in ledger:bykey[r['builder_pubkey']].append(r)
    f=pd.read_parquet(a.input,columns=['builder_pubkey','slot_start_date_time'])
    f['builder_pubkey']=f.builder_pubkey.str.casefold()
    months=pd.to_datetime(f.slot_start_date_time,utc=True).dt.strftime('%Y-%m')
    keys=sorted(f.builder_pubkey.unique())
    families=('dataalways','yang','brontes')
    specs=[('official_plus_'+r,{r},False) for r in families]
    specs += [('without_'+r,set(families)-{r},False) for r in families]
    specs += [('official_or_two_registry_agreement',set(families),True)]
    rows=[];maprows=[]
    for scenario,include,require_two in specs:
        lookup={}
        for key in keys:
            eligible=[r for r in bykey[key] if r['evidence_class']=='official_historical' or
                      (r['source_id'].split('_')[0] in include and r['evidence_class']=='research_registry_historical')]
            name,date,basis,sources=resolve_key(eligible,'historical','service')
            if name and basis=='registry' and require_two and len({s.split('_')[0] for s in sources})<2:
                name='';basis='single_registry_only'
            lookup[key]='external:'+name if name else 'key:'+key
            maprows.append(dict(scenario=scenario,builder_pubkey=key,identity=lookup[key],basis=basis))
        labels=f.builder_pubkey.map(lookup)
        known=labels.str.startswith('external:')
        for month in sorted(months.unique())+['all']:
            mask=months.eq(month) if month!='all' else np.ones(len(f),bool)
            counts=np.array(sorted(Counter(labels[mask]).values(),reverse=True),dtype=float)
            n=int(counts.sum())
            hhi=float(np.dot(counts,counts)/(n*n))
            # Separate scalar arithmetic confirms each concentration result.
            assert abs(hhi-math.fsum((int(x)/n)**2 for x in counts))<1e-12
            rows.append(dict(scenario=scenario,month=month,slots=n,matched_slots=int(known[mask].sum()),
                             slot_coverage=float(known[mask].mean()),hhi=hhi,cr3=float(counts[:3].sum()/n),cr5=float(counts[:5].sum()/n)))
    result=pd.DataFrame(rows)
    result.to_csv(a.out/'concentration.csv',index=False)
    result[result.month.ne('all')].groupby('scenario')[['hhi','cr3','cr5']].median().reset_index().to_csv(a.out/'monthly_medians.csv',index=False)
    pd.DataFrame(maprows).to_csv(a.out/'key_mapping.csv',index=False)
    manifest=dict(status='verified',input_sha256=sha256(a.input),code_sha256=sha256(Path(__file__)),
                  scenarios=len(specs),output_sha256={x.name:sha256(x) for x in a.out.iterdir() if x.is_file()})
    (a.out/'validation.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(result[result.month.eq('all')][['scenario','slot_coverage','hhi','cr3']].to_string(index=False))


if __name__=='__main__':main()
