"""Independent integrity checks for the v3 export; no API calls."""
import json,collections
from pathlib import Path
import build_v2 as b
v=b.v1;ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/v3'
def main():
 m=json.loads((OUT/'manifest.json').read_text());parts={s:b.read(OUT/'records'/f'{s}.jsonl') for s in m['splits']}
 assert v.validate_partitions(parts)==m['integrity']
 for file,h in m['files'].items():assert v.digest((OUT/file).read_text())==h,file
 lock=json.loads((OUT/'holdout-lock.json').read_text())
 assert v.digest((OUT/'holdout-lock.json').read_text())==m['holdout_lock_sha256']
 assert v.digest((OUT/'mining.jsonl').read_text())==m['mining_sha256']
 for split,h in lock['hashes'].items():assert v.digest((OUT/'staging'/f'{split}.jsonl').read_text())==h
 staged={r['id']:r for r in b.read(OUT/'staging/candidates.jsonl')}
 old={r['id']:r for r in b.read(ROOT/'data/v2.1/records/train.jsonl')}
 quarantined=json.loads((OUT/'quarantine.json').read_text())['excluded']
 hold={r['id']:r for r in b.read(OUT/'staging/holdout.jsonl')}
 for r in parts['holdout']:
  q=dict(r);q.pop('token_count',None);z=dict(hold[r['id']]);z.pop('token_count',None);assert q==z
 for r in parts['train']:
  role=r['provenance'].get('v3_role')
  if role in ('fresh_error_enriched','replay_v21_train'):
   ref=(staged if role=='fresh_error_enriched' else old)[r['id']]
   for k in ('state','question','options','answer','answer_key','group_id'):assert r[k]==ref[k],r['id']
   assert r['id'] not in quarantined
  elif role=='banking_hard_candidates':
   original=r['id'].split(':v3-24')[0];assert original not in quarantined
   assert r['state']==staged[original]['state'] and r['group_id']==staged[original]['group_id']
   assert len(r['options'])==24
   keys={o['key'] for o in r['options']};p=r['provenance'];assert ('none' in keys)==p['gold_removed']
   assert (p['underlying_intent'] in keys)!=p['gold_removed']
   assert r['answer_key']==('none' if p['gold_removed'] else p['underlying_intent'])
  elif r['source']=='entailment_contrast_v3':
   # Independently check the explicit claims against their complete ledger facts.
   import re
   n=int(re.search(r'shipped exactly (\d+) crates',r['state']).group(1));day=int(re.search(r'on day (\d+)',r['state']).group(1));claim=r['question'].split('claim: ',1)[1]
   if 'contained books' in claim or f'on day {day+1}.' in claim:want='neutral'
   elif 'No crate' in claim or f'at least {n} crates' in claim:want='entailed'
   elif 'At least one crate' in claim:want='contradicted'
   else:
    amount=int(re.search(r'exactly (\d+) crates',claim).group(1));want='entailed' if amount==n else 'contradicted'
   assert r['answer_key']==want,r['id']
  elif r['source']!='news_contrast_v3':raise AssertionError(r['id'])
 # Exclude every previous evaluation's normalized states from training.
 trainstates={b.statehash(r) for r in parts['train']}
 for version in ['v1','v2','v2.1']:
  for p in (ROOT/'data'/version/'records').glob('*.jsonl'):
   if p.stem!='train':assert not trainstates&{b.statehash(r) for r in b.read(p)},str(p)
 from transformers import AutoTokenizer
 pin=m['sources']['tokenizer'];tok=AutoTokenizer.from_pretrained(pin['repo'],revision=pin['revision'],local_files_only=True)
 for split,rows in parts.items():
  raw=b.read(OUT/'instruction'/f'{split}.jsonl');chats=b.read(OUT/'sft'/f'{split}.jsonl');assert len(rows)==len(raw)==len(chats)==m['splits'][split]['count']
  for r,x,c in zip(rows,raw,chats):
   assert c['messages']==v.messages(r)
   assert x['completion']==r['answer']+tok.eos_token
   assert x['prompt']==tok.apply_chat_template(c['messages'][:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   assert r['token_count']<=2048
 print(json.dumps({'validation':'PASS','splits':{s:len(r) for s,r in parts.items()},'quarantined':len(quarantined),'train_vs_prior_eval_state_overlap':0,'checks':'source gold preservation, candidate construction, NLI oracle, frozen holdout, group/state/prompt separation, file hashes, chat/instruction alignment'}))
if __name__=='__main__':main()
