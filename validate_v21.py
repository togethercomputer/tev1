"""Validate extension preservation, export alignment, and 24-label supervision."""
import json,string,sys
from pathlib import Path
from collections import defaultdict,Counter
sys.path.insert(0,'/Users/hassan/dev/open-jev')
import build_v21 as b
from transformers import AutoTokenizer
root=Path(sys.argv[1]) if len(sys.argv)>1 else b.PROJECT/'data/v2.1'
m=json.loads((root/'manifest.json').read_text());base=b.PROJECT/'data/v2'
for name,h in m['files'].items():assert b.v.digest((root/name).read_text())==h,name
parts={s:b.b.read(root/'records'/f'{s}.jsonl') for s in m['splits']}
assert b.v.validate_partitions(parts)==m['integrity']
for s in ['train','dev','calibration','test','transfer']:
 old={r['id']:r for r in b.b.read(base/'records'/f'{s}.jsonl')};new={r['id']:r for r in parts[s]}
 assert all(new[k]==v for k,v in old.items()),('changed base example',s)
 if s in ['calibration','test','transfer']:
  assert (root/'records'/f'{s}.jsonl').read_bytes()==(base/'records'/f'{s}.jsonl').read_bytes()
print('PASS: all v2 examples preserved; reserved v2 records unchanged',flush=True)
pin=m['sources']['tokenizer'];tok=AutoTokenizer.from_pretrained(pin['repo'],revision=pin['revision'],local_files_only=True)
counts={'train':33840,'dev':3568,'calibration':2800,'test':2800,'transfer':1800,'research_challenge':768}
for s,rows in parts.items():
 assert len(rows)==counts[s]
 chats=b.b.read(root/'sft'/f'{s}.jsonl');raw=b.b.read(root/'instruction'/f'{s}.jsonl');assert len(rows)==len(chats)==len(raw)
 groups=defaultdict(list);letters=Counter()
 for r,c,x in zip(rows,chats,raw):
  chat=b.v.messages(r);assert c=={'messages':chat};assert x['completion']==r['answer']+tok.eos_token
  if r['source']!='research_taxonomy_v21':continue
  prefix=tok.apply_chat_template(chat[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  assert x['prompt']==prefix
  token=tok.encode(r['answer'],add_special_tokens=False);assert len(token)==1
  assert tok.encode(prefix+r['answer'],add_special_tokens=False)==tok.encode(prefix,add_special_tokens=False)+token
  assert len(tok.encode(prefix+x['completion'],add_special_tokens=False))==r['token_count']<=2048
  p=r['provenance'];assert r['answer_key']==p['main_contribution_key']!=p['background_key']
  assert p['contribution'] in b.CONTRIBUTIONS[r['answer_key']]
  assert r['state']['summary']==p['template'].format(main=p['contribution'],method=p['method'],outcome=b.OUTCOMES[r['answer_key']],other=p['background'])
  groups[r['group_id']].append(r);letters[r['answer']]+=1
 for g,rr in groups.items():
  assert len(rr)==8
  sides={i:[r for r in rr if r['provenance']['pair_side']==i] for i in [0,1]}
  for side,xx in sides.items():
   assert len({r['answer_key'] for r in xx})==1
   assert sorted(len(r['options']) for r in xx)==[8,12,24,24]
  assert sides[0][0]['answer_key']!=sides[1][0]['answer_key']
 if groups:assert set(letters)==set(string.ascii_uppercase[:24])
 print('PASS',s,len(rows),'new research groups',len(groups),flush=True)
print('PASS: complete coverage, preserved base records, all A-X labels, grouped pairs, and aligned exports')
