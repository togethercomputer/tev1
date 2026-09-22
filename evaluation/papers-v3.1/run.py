import asyncio,json,os,time,hashlib,statistics,collections
from pathlib import Path
import httpx
BASE=Path('/Users/hassan/dev/1kpapers/experiments/qwen35-4b-jev-v21-c0ad7564');OUT=Path(__file__).resolve().parent;MODEL='hassan/Qwen3.5-4B-v3.1-a9db1708-e46d1350'
async def main():
 config=json.loads((BASE/'config.json').read_text());refs={r['id']:r for r in json.loads((BASE/'comparison.json').read_text())};sem=asyncio.Semaphore(15);results=[]
 assert len(config['inputs'])==len(refs)==891
 async with httpx.AsyncClient(headers={'Authorization':'Bearer '+os.environ['TOGETHER_API_KEY'],'User-Agent':'open-jev/0.1'},timeout=60) as c:
  with (OUT/'results.jsonl').open('x') as f:
   async def one(r):
    async with sem:
     labels=[o['label'] for o in config['options']]
     body={'model':MODEL,'temperature':0,'max_tokens':8,'logprobs':5,'chat_template_kwargs':{'enable_thinking':False},'response_format':{'type':'regex','pattern':'('+'|'.join(labels)+')'},'messages':[{'role':'system','content':config['system']},{'role':'user','content':r['state']}]}
     x={'id':r['id'],'status':0,'valid':False,'correct':False};t=time.perf_counter()
     try:
      res=await c.post('https://api.together.ai/v1/chat/completions',json=body);x['status']=res.status_code
      if res.status_code==200:
       d=res.json();ch=d['choices'][0];answer=(ch['message'].get('content') or '').strip();key=next((o['key'] for o in config['options'] if o['label']==answer),None)
       x.update(answer=answer,predicted=key,valid=key is not None and ch.get('finish_reason')=='stop',resolved_model=d.get('model'),logprobs=ch.get('logprobs'),usage=d.get('usage'))
      else:x['error']=res.text[:300]
     except Exception as e:x['error']=type(e).__name__
     x['elapsed_ms']=(time.perf_counter()-t)*1000;x['gold']=refs[r['id']]['consensus'];x['correct']=x['valid'] and x.get('predicted')==x['gold'];results.append(x);f.write(json.dumps(x)+'\n');f.flush()
     if len(results)%150==0:print(json.dumps({'completed':len(results),'errors':sum(r['status']!=200 for r in results)}),flush=True)
   await asyncio.gather(*(one(r) for r in config['inputs']))
 times=sorted(r['elapsed_ms'] for r in results if r['status']==200)
 report={'model':MODEL,'n':len(results),'correct':sum(r['correct'] for r in results),'valid':sum(r['valid'] for r in results),'errors':sum(r['status']!=200 for r in results),'median_ms':statistics.median(times),'p95_ms':times[int(.95*len(times))],'config_sha256':hashlib.sha256((BASE/'config.json').read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256((BASE/'comparison.json').read_bytes()).hexdigest(),'method':'Same 891 paper states, rubric, option order, system instruction and consensus as prior run. Regex A-X, logprobs5, temperature0, thinking off, max_tokens8, concurrency15, no retries. Historical comparators used unconstrained output. Consensus agreement, not human ground truth. Feedback-informed development benchmark; holdout untouched.','comparators':{k:sum(r[k]==r['consensus'] for r in refs.values()) for k in ['qwenPrimary','qwen4bPrimary','qwen2bv21Primary','jevPrimary']},'paired_vs_v21':{'fixed':sum(r['correct'] and refs[r['id']]['qwenPrimary']!=r['gold'] for r in results),'regressed':sum(not r['correct'] and refs[r['id']]['qwenPrimary']==r['gold'] for r in results)},'by_topic':{k:{'n':sum(r['gold']==k for r in results),'correct':sum(r['gold']==k and r['correct'] for r in results),'v21_correct':sum(r['consensus']==k and r['qwenPrimary']==k for r in refs.values())} for k in sorted({r['gold'] for r in results})}}
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
asyncio.run(main())
