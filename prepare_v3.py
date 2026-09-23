import sys,json,collections,copy,random
from pathlib import Path
ROOT=Path('/Users/hassan/dev/open-jev');sys.path.insert(0,str(ROOT));import build_v2 as b
from datasets import load_dataset
v=b.v1
OUT=ROOT/'data/v3';OUT.mkdir(exist_ok=False,parents=True)
old=[]
for version in ['v1','v2','v2.1']:
 for p in (ROOT/'data'/version/'records').glob('*.jsonl'):old.extend(b.read(p))
forbidden={b.statehash(r) for r in old};lock=json.loads((ROOT/'sources.lock.json').read_text());audit={};parts={'candidates':[],'dev':[],'holdout':[]}
counts={'sst5':800,'banking77':1000,'ag_news':600,'mnli':1200,'boolq':400}
for name,n in counts.items():
 source=lock['datasets'][name];ds=load_dataset(source['repo'],name=source['config'],revision=source['revision'])
 tests=['validation_matched','validation_mismatched'] if name=='mnli' else ['validation' if name=='boolq' else 'test']
 seen=forbidden|{x['group'] for s in tests for x in v.prepare_rows(name,ds[s],s,source)}
 pool=[]
 for x in v.prepare_rows(name,ds['train'],'train',source):
  if x['group'] not in seen:pool.append(x);seen.add(x['group'])
 audit[name]={'available_fresh':len(pool),'official_heldout_excluded':tests}
 for split,k in [('holdout',100),('dev',100),('candidates',n)]:
  picked=v.balanced_take(pool,k,'v3-classification:'+name+split);taken={x['group'] for x in picked};pool=[x for x in pool if x['group'] not in taken]
  for x in picked:
   row=v.public_record(name,x,split,source);row['provenance']['v3_role']='fresh_public_'+split;parts[split].append(row)
 print(name,audit[name],flush=True)
for split,rows in parts.items():v.write_jsonl(OUT/'staging'/f'{split}.jsonl',rows)
v.write_json(OUT/'holdout-lock.json',{'seed':'v3-classification','counts':{s:len(r) for s,r in parts.items()},'hashes':{s:v.digest((OUT/'staging'/f'{s}.jsonl').read_text()) for s in parts},'source_audit':audit,'holdout_policy':'500 source-labeled unused upstream-training records reserved before mining. No holdout model predictions used for selection. Existing upstream official heldout text and all prior splits excluded. Same domains, not an OOD claim.'})
