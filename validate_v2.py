"""Validate a finished continuation dataset and every exported prompt."""
import argparse,json
from pathlib import Path
import build_v2 as b

def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path,nargs='?',default=b.PROJECT/'data/v2');args=ap.parse_args();root=args.directory
 manifest=json.loads((root/'manifest.json').read_text())
 for name,h in manifest['files'].items():assert b.v1.digest((root/name).read_text())==h,name
 parts={s:b.read(root/'records'/f'{s}.jsonl') for s in manifest['splits']}
 old={p.stem:b.read(p) for p in (b.PROJECT/'data/v1/records').glob('*.jsonl')}
 assert b.validate(parts,old)==manifest['integrity']
 from transformers import AutoTokenizer
 pin=manifest['sources']['tokenizer'];tok=AutoTokenizer.from_pretrained(pin['repo'],revision=pin['revision'],local_files_only=True)
 for split,rows in parts.items():
  assert len(rows)==(30000 if split=='train' else 1800 if split=='transfer' else 2800)
  chats=b.read(root/'sft'/f'{split}.jsonl');instructions=b.read(root/'instruction'/f'{split}.jsonl')
  assert len(rows)==len(chats)==len(instructions)
  for row,chat,raw in zip(rows,chats,instructions):
   expected=b.v1.messages(row);assert chat=={'messages':expected}
   prefix=tok.apply_chat_template(expected[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   assert raw=={'prompt':prefix,'completion':row['answer']+tok.eos_token}
   assert len(tok.encode(prefix+raw['completion'],add_special_tokens=False))==row['token_count']<=2048
  print('PASS',split,len(rows),flush=True)
 print('PASS: hashes, independent labels, partitions, v1 exclusions, and all exports',flush=True)
if __name__=='__main__':main()
