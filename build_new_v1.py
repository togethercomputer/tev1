"""Merge v1 and v2.1 train/dev, removing exact replay duplicates; no new labels."""
import json,random,collections
from pathlib import Path
import build_v2 as b
v=b.v1;ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/new-v1'
def main():
 assert not OUT.exists(),'Refuse to overwrite existing dataset'
 parts={};exports={};audit={};inputs={};membership={}
 for split in ['train','dev']:
  selected={};by_id={};duplicates=0;input_n=0
  for version in ['v2.1','v1']:
   base=ROOT/'data'/version;rows=b.read(base/'records'/f'{split}.jsonl');raw=b.read(base/'instruction'/f'{split}.jsonl');chats=b.read(base/'sft'/f'{split}.jsonl')
   assert len(rows)==len(raw)==len(chats)
   for folder in ['records','instruction','sft']:
    p=base/folder/f'{split}.jsonl';inputs[f'{version}/{folder}/{split}.jsonl']=v.digest(p.read_text())
   for row,item,chat in zip(rows,raw,chats):
    input_n+=1;assert chat['messages']==v.messages(row)
    key=v.digest(v.messages(row)[:-1])
    if row['id'] in by_id:assert by_id[row['id']]==key,'Same ID different prompt'
    by_id[row['id']]=key
    if key in selected:
     prior=selected[key];assert prior[0]['answer']==row['answer'] and prior[1]==item,'Conflicting duplicate'
     duplicates+=1;membership[split+':'+key].append({'version':version,'id':row['id']});continue
    selected[key]=(row,item,chat);membership[split+':'+key]=[{'version':version,'id':row['id']}]
  items=list(selected.values());random.Random(42).shuffle(items);parts[split]=[x[0] for x in items];exports[split]=items
  audit[split]={'input_rows':input_n,'exact_duplicates_removed':duplicates,'unique_rows':len(items)}
 integrity=v.validate_partitions(parts)
 protected=set()
 for version in ['v1','v2','v2.1','v3','v3.1']:
  for p in (ROOT/'data'/version/'records').glob('*.jsonl'):
   if p.stem!='train':protected.update(b.statehash(r) for r in b.read(p))
 assert not {b.statehash(r) for r in parts['train']}&protected
 manifest={'version':'new-v1','description':'Unique union of v1 and v2.1; original supervised examples and exports unchanged; no v3/v3.1 augmentation','seed':42,'input_hashes':inputs,'deduplication':audit,'integrity':integrity,'files':{},'splits':{},'builder_sha256':v.digest(Path(__file__).read_text())}
 for split,items in exports.items():
  for folder,col in [('records',0),('instruction',1),('sft',2)]:
   p=OUT/folder/f'{split}.jsonl';v.write_jsonl(p,[r[col] for r in items]);manifest['files'][str(p.relative_to(OUT))]=v.digest(p.read_text())
  rows=parts[split];manifest['splits'][split]={'count':len(rows),'tokens':sum(r['token_count'] for r in rows),'max_tokens':max(r['token_count'] for r in rows),'sources':dict(collections.Counter(r['source'] for r in rows))}
  print(split,audit[split],manifest['splits'][split],flush=True)
 v.write_json(OUT/'membership.json',membership);manifest['files']['membership.json']=v.digest((OUT/'membership.json').read_text());v.write_json(OUT/'manifest.json',manifest)
 # Independent export checks using tokenizer rather than trusting cached counts.
 from transformers import AutoTokenizer
 source=json.loads((ROOT/'sources.lock.json').read_text())['tokenizer'];tok=AutoTokenizer.from_pretrained(source['repo'],revision=source['revision'],local_files_only=True)
 for split,items in exports.items():
  for row,item,chat in items:
   expected=tok.apply_chat_template(chat['messages'][:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   assert item['prompt']==expected and item['completion']==row['answer']+tok.eos_token
   assert len(tok.encode(item['prompt']+item['completion'],add_special_tokens=False))==row['token_count']<=2048
 v.write_json(OUT/'validation.json',{'status':'PASS','checks':['gold conflicts','ID/prompt consistency','source export preservation','group/state/prompt split separation','all prior heldout states excluded from train','Qwen template and completion alignment','token counts independently recomputed <=2048'],'deduplication':audit})
 print('VALIDATION PASS',flush=True)
if __name__=='__main__':main()
