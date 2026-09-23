"""Build v3 continuation exports from frozen candidates and model-error mining."""
import json,copy,random,string,collections
from pathlib import Path
import build_v2 as b
import v3_contrasts
v=b.v1;ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/v3'
def rng(k):return random.Random(int(v.digest('v3-final:'+k),16))
def main():
 assert not (OUT/'manifest.json').exists(),'Refuse to overwrite completed build'
 candidates=b.read(OUT/'staging/candidates.jsonl');mined={r['id']:r for r in b.read(OUT/'mining.jsonl')}
 assert len(mined)==len(candidates)==4000 and all(r['status']==200 for r in mined.values())
 lock=json.loads((OUT/'holdout-lock.json').read_text())
 for split,h in lock['hashes'].items():assert v.digest((OUT/'staging'/f'{split}.jsonl').read_text())==h
 quarantine=json.loads((OUT/'quarantine.json').read_text())['excluded']
 train=[];audit={};quotas={'sst5':500,'banking77':650,'ag_news':350,'mnli':700,'boolq':200}
 # Cap mined errors at half of each source. Labels always remain upstream gold.
 for source,n in quotas.items():
  rr=[r for r in candidates if r['source']==source and r['id'] not in quarantine];wrong=[r for r in rr if not mined[r['id']]['correct']];right=[r for r in rr if mined[r['id']]['correct']]
  rng(source+'wrong').shuffle(wrong);rng(source+'right').shuffle(right)
  nw=min(n//2,len(wrong));chosen=wrong[:nw]+right[:n-nw]
  if len(chosen)<n:chosen+=wrong[nw:nw+n-len(chosen)]
  assert len(chosen)==n
  for r in chosen:
   r=copy.deepcopy(r);r['split']='train';r['provenance'].update(v3_role='fresh_error_enriched',mining_correct=mined[r['id']]['correct']);train.append(r)
  audit[source]={'candidate_n':len(rr),'candidate_errors':len(wrong),'quarantined':sum(r['source']==source and r['id'] in quarantine for r in candidates),'selected':n,'selected_errors':sum(not mined[r['id']]['correct'] for r in chosen)}
 # Banking distractors use observed semantic confusion edges, never benchmark text.
 analysis=json.loads((ROOT/'evaluation/v3-analysis/error-analysis.json').read_text())
 edges=collections.defaultdict(set)
 for c in analysis['banking77']['confusions']:
  if 'none' not in (c['gold'],c['predicted']):edges[c['gold']].add(c['predicted']);edges[c['predicted']].add(c['gold'])
 from datasets import load_dataset
 sources=json.loads((ROOT/'sources.lock.json').read_text());s=sources['datasets']['banking77']
 ds=load_dataset(s['repo'],name=s['config'],revision=s['revision']);names=ds['train'].features['label'].names
 bank=[copy.deepcopy(r) for r in train if r['source']=='banking77']
 for i,r in enumerate(bank):
  gold=names[r['provenance']['original_label']];rd=rng(r['id']);others=[k for k in names if k!=gold];rd.shuffle(others)
  others.sort(key=lambda k:(k not in edges[gold],-len(set(k.split('_'))&set(gold.split('_')))))
  keys=[gold]+others[:23];rd.shuffle(keys)
  for removed in ([False,True] if i%4==0 else [False]):
   q=copy.deepcopy(r);q['id']+=':v3-24'+('-absent' if removed else '');kk=['none' if removed and k==gold else k for k in keys]
   q['options']=[{'label':string.ascii_uppercase[j],'key':k,'description':'None of the listed intents matches.' if k=='none' else k.replace('_',' ')+'.'} for j,k in enumerate(kk)]
   q['answer_key']='none' if removed else gold;q['answer']=next(o['label'] for o in q['options'] if o['key']==q['answer_key'])
   q['provenance'].update(v3_role='banking_hard_candidates',underlying_intent=gold,gold_removed=removed,candidate_sampling='24 options; observed confusion neighbors then lexical overlap; paired absent-gold cases on 1/4 of groups')
   train.append(q)
 train+=v3_contrasts.build()
 # Group-preserving replay across all existing skills. Do not copy old validation.
 old=b.read(ROOT/'data/v2.1/records/train.jsonl');replay={}
 targets={'policy_v2':4000,'routing_v2':2000,'research_taxonomy_v21':2000,'mnli':1500,'boolq':800,'banking77':700,'ag_news':500,'sst5':500}
 for source,target in targets.items():
  groups=collections.defaultdict(list)
  for r in old:
   if r['source']==source:groups[r['group_id']].append(r)
  keys=list(groups);rng('replay'+source).shuffle(keys);selected=[]
  for key in keys:
   if len(selected)>=target:break
   selected+=groups[key]
  for r in selected:
   q=copy.deepcopy(r);q['provenance']['v3_role']='replay_v21_train';train.append(q)
  replay[source]=len(selected)
 parts={'train':train,'dev':b.read(OUT/'staging/dev.jsonl'),'holdout':b.read(OUT/'staging/holdout.jsonl')}
 # Keep unchanged development tasks available separately for regression selection.
 for r in b.read(ROOT/'data/v2.1/records/dev.jsonl'):
  r['provenance']['v3_role']='regression_dev_v21';parts['dev'].append(r)
 for name in parts:rng('export'+name).shuffle(parts[name])
 integrity=v.validate_partitions(parts)
 # New training may replay old training only; all older non-training states protected.
 protected=set()
 for version in ['v1','v2','v2.1']:
  for p in (ROOT/'data'/version/'records').glob('*.jsonl'):
   if p.stem!='train':protected.update(b.statehash(r) for r in b.read(p))
 assert not {b.statehash(r) for r in train}&protected
 from transformers import AutoTokenizer
 pin=sources['tokenizer'];tok=AutoTokenizer.from_pretrained(pin['repo'],revision=pin['revision'],local_files_only=True)
 labels={c:tok.encode(c,add_special_tokens=False) for c in string.ascii_uppercase[:24]};assert all(len(a)==1 for a in labels.values())
 manifest={'version':'3','base':'v2.1','model_mined':'hassan/Qwen3.5-4B-jev-2-1-4286f48b-c0ad7564','sources':sources,'mining_audit':audit,'replay':replay,'integrity':integrity,'files':{},'splits':{},'max_tokens':2048,'label_token_ids':labels,'holdout_lock_sha256':v.digest((OUT/'holdout-lock.json').read_text()),'mining_sha256':v.digest((OUT/'mining.jsonl').read_text()),'builder_hashes':{f:v.digest((ROOT/f).read_text()) for f in ['prepare_v3.py','mine_v3.py','build_v3.py','v3_contrasts.py']}}
 for split,rows in parts.items():
  raw=[];chats=[];lengths=[]
  for r in rows:
   chat=v.messages(r);prefix=tok.apply_chat_template(chat[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=tok.encode(prefix,add_special_tokens=False)
   assert tok.encode(prefix+r['answer'],add_special_tokens=False)==ids+labels[r['answer']]
   completion=r['answer']+tok.eos_token;length=len(tok.encode(prefix+completion,add_special_tokens=False));assert length<=2048,(r['id'],length)
   r['token_count']=length;lengths.append(length);raw.append({'prompt':prefix,'completion':completion});chats.append({'messages':chat})
  for folder,items in [('records',rows),('instruction',raw),('sft',chats)]:
   p=OUT/folder/f'{split}.jsonl';v.write_jsonl(p,items);manifest['files'][str(p.relative_to(OUT))]=v.digest(p.read_text())
  manifest['splits'][split]={'count':len(rows),'tokens':sum(lengths),'max_tokens':max(lengths),'sources':dict(collections.Counter(r['source'] for r in rows)),'roles':dict(collections.Counter(r['provenance'].get('v3_role','authored_contrast') for r in rows)),'groups':len({r['group_id'] for r in rows})}
  print(split,manifest['splits'][split],flush=True)
 v.write_json(OUT/'manifest.json',manifest)
if __name__=='__main__':main()
