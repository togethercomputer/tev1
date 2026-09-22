"""Validate the research-augmented exports and preserved v3 data."""
from pathlib import Path
import json,collections
import build_v2 as b
import research_v31
v=b.v1;ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/v3.1'
def main():
 m=json.loads((OUT/'manifest.json').read_text());base=ROOT/'data/v3'
 assert v.digest((base/'manifest.json').read_text())==m['base_manifest_sha256']
 for f,h in m['files'].items():assert v.digest((OUT/f).read_text())==h
 for f,h in m['builder_hashes'].items():assert v.digest((ROOT/f).read_text())==h
 parts={s:b.read(OUT/'records'/f'{s}.jsonl') for s in m['splits']};assert v.validate_partitions(parts)==m['integrity']
 for s,rows in parts.items():
  indexed={r['id']:r for r in rows}
  for old in b.read(base/'records'/f'{s}.jsonl'):assert indexed[old['id']]==old,old['id']
 for folder in ['records','sft','instruction']:assert (OUT/folder/'holdout.jsonl').read_bytes()==(base/folder/'holdout.jsonl').read_bytes()
 extras={s:[r for r in rows if r['source']=='research_taxonomy_v31'] for s,rows in parts.items()}
 for s in ['train','dev']:
  rr=extras[s];assert len(rr)==m['research']['new_counts'][s]
  assert set(r['answer_key'] for r in rr)==set(research_v31.CORES)
  assert len({v.digest(v.messages(r)[:-1]) for r in rr})==len(rr)
  for r in rr:
   p=r['provenance'];core=research_v31.CORES[p['topic']][p['core_id']]
   assert p['core']==core and core in r['state'] and p['context'] in r['state']
   assert r['answer_key']==p['topic']!=p['background_topic']
   assert p['core_id'] in ([0,1] if s=='train' else [2])
   assert len(r['options'])==24 and set(o['key'] for o in r['options'])==set(research_v31.CORES)
   assert [o['label'] for o in r['options']]==list('ABCDEFGHIJKLMNOPQRSTUVWX')
   if p['option_variant']=='canonical24':assert [o['key'] for o in r['options']]==list(research_v31.CORES)
 for topic in research_v31.CORES:
  # Dev cores are different descriptions; neither appears verbatim in training text.
  text=research_v31.CORES[topic][2]
  assert all(text not in r['state'] for r in extras['train'])
 protected=set()
 for version in ['v1','v2','v2.1','v3']:
  for p in (ROOT/'data'/version/'records').glob('*.jsonl'):
   if p.stem!='train':protected.update(b.statehash(r) for r in b.read(p))
 assert not {b.statehash(r) for r in parts['train']}&protected
 from transformers import AutoTokenizer
 pin=m['sources']['tokenizer'];tok=AutoTokenizer.from_pretrained(pin['repo'],revision=pin['revision'],local_files_only=True)
 for s,rows in parts.items():
  chats=b.read(OUT/'sft'/f'{s}.jsonl');raw=b.read(OUT/'instruction'/f'{s}.jsonl');assert len(rows)==len(chats)==len(raw)
  for r,c,x in zip(rows,chats,raw):
   assert c['messages']==v.messages(r);assert x['completion']==r['answer']+tok.eos_token
   assert x['prompt']==tok.apply_chat_template(c['messages'][:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   assert r['token_count']<=2048
 print(json.dumps({'validation':'PASS','counts':{s:len(r) for s,r in parts.items()},'new_research':{s:len(extras[s]) for s in ['train','dev']},'checks':'all v3 records preserved, exact holdout bytes preserved, hashes, gold/option alignment, 24-topic coverage, distinct train/dev cores, no prior-eval states in train, export alignment'}))
if __name__=='__main__':main()
