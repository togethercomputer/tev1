"""Preserve v3, append research augmentation, export v3.1 with frozen holdout."""
import json,copy,random,collections,string
from pathlib import Path
import build_v2 as b
import research_v31
v=b.v1;ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/v3.1'
BENCH=Path('/Users/hassan/dev/1kpapers/experiments/qwen35-4b-jev-v21-c0ad7564')
def main():
 assert not OUT.exists(),'Preserve completed versions; output exists'
 base=ROOT/'data/v3';base_manifest=json.loads((base/'manifest.json').read_text())
 config=json.loads((BENCH/'config.json').read_text());question=json.loads(config['inputs'][0]['state'])['question'];options=config['options']
 assert set(research_v31.CORES)=={o['key'] for o in options} and all(len(x)==3 for x in research_v31.CORES.values())
 analysis=json.loads((ROOT/'evaluation/v3-analysis/paper-error-analysis.json').read_text())
 parts={s:b.read(base/'records'/f'{s}.jsonl') for s in ['train','dev','holdout']}
 extras={s:research_v31.build(s,options,question,analysis['confusions']) for s in ['train','dev']}
 for s,rows in extras.items():
  # Same prose may appear with another option order, never exact duplicate prompts.
  assert len({v.digest(v.messages(r)[:-1]) for r in rows})==len(rows)
  parts[s]+=rows;random.Random(31021).shuffle(parts[s])
 integrity=v.validate_partitions(parts)
 # Exclude benchmark state, title and long exact spans; allowed rubric shared by design.
 import re
 def words(text):return re.findall(r'\w+',text.lower())
 bench_states=[json.loads(i['state'])['state'] for i in config['inputs']]
 titles=[r['title'].lower() for r in json.loads((BENCH/'reference.json').read_text())]
 spans=set()
 for state in bench_states:
  ww=words(state);spans.update(tuple(ww[i:i+16]) for i in range(len(ww)-15))
 for r in extras['train']+extras['dev']:
  assert r['state'] not in bench_states
  assert not any(t in r['state'].lower() for t in titles if len(t)>20)
  ww=words(r['state']);assert not any(tuple(ww[i:i+16]) in spans for i in range(len(ww)-15)),r['id']
 from transformers import AutoTokenizer
 pin=base_manifest['sources']['tokenizer'];tok=AutoTokenizer.from_pretrained(pin['repo'],revision=pin['revision'],local_files_only=True)
 ids={c:tok.encode(c,add_special_tokens=False) for c in string.ascii_uppercase[:24]};assert all(len(x)==1 for x in ids.values())
 manifest={'version':'3.1','base_manifest_sha256':v.digest((base/'manifest.json').read_text()),'sources':base_manifest['sources'],'integrity':integrity,'max_tokens':2048,'files':{},'splits':{},'research':{'benchmark_n':891,'benchmark_disagreements':124,'label_source':'authored fictional artifact/evidence targets, not judge answers','new_train_cores':48,'new_dev_cores':24,'new_counts':{s:len(r) for s,r in extras.items()},'all_24_topics':True,'limitations':'Controlled variants; not independent real papers. Dev shares rubric and topic boundaries. Existing benchmark is development evidence. No claim of improved accuracy before training.'},'benchmark_config_sha256':v.digest((BENCH/'config.json').read_text()),'analysis_sha256':v.digest((ROOT/'evaluation/v3-analysis/paper-error-analysis.json').read_text()),'builder_hashes':{f:v.digest((ROOT/f).read_text()) for f in ['research_v31.py','build_v31.py']}}
 for split,rows in parts.items():
  chats=[];raw=[];lengths=[]
  for row in rows:
   chat=v.messages(row);prefix=tok.apply_chat_template(chat[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   pre=tok.encode(prefix,add_special_tokens=False);assert tok.encode(prefix+row['answer'],add_special_tokens=False)==pre+ids[row['answer']]
   completion=row['answer']+tok.eos_token;length=len(tok.encode(prefix+completion,add_special_tokens=False));assert length<=2048,(row['id'],length)
   row['token_count']=length;lengths.append(length);chats.append({'messages':chat});raw.append({'prompt':prefix,'completion':completion})
  for folder,items in [('records',rows),('sft',chats),('instruction',raw)]:
   p=OUT/folder/f'{split}.jsonl';v.write_jsonl(p,items);manifest['files'][str(p.relative_to(OUT))]=v.digest(p.read_text())
  manifest['splits'][split]={'count':len(rows),'tokens':sum(lengths),'max_tokens':max(lengths),'sources':dict(collections.Counter(r['source'] for r in rows)),'new_research_topic_counts':dict(collections.Counter(r['answer_key'] for r in rows if r['source']=='research_taxonomy_v31'))}
  print(split,manifest['splits'][split],flush=True)
 # All original holdout bytes must be identical across every exported format.
 for folder in ['records','sft','instruction']:assert (OUT/folder/'holdout.jsonl').read_bytes()==(base/folder/'holdout.jsonl').read_bytes()
 v.write_json(OUT/'manifest.json',manifest)
if __name__=='__main__':main()
