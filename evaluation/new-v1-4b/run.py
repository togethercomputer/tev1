import asyncio,json,os,sys,time,hashlib,statistics,random,collections
from pathlib import Path
import httpx
ROOT=Path('/Users/hassan/dev/open-jev');sys.path.insert(0,str(ROOT))
from examples.decide import payload,select
OUT=Path(__file__).resolve().parent;MODEL='hassan/Qwen3.5-4B-v1-new-69617472-bdc3c2fc'
def read(p):return list(map(json.loads,p.read_text().splitlines()))
def summary(rs):
 times=sorted(r['latency_ms'] for r in rs if r['status']==200)
 return {'n':len(rs),'correct':sum(r['correct'] for r in rs),'accuracy':sum(r['correct'] for r in rs)/len(rs),'valid':sum(r['valid'] for r in rs),'errors':sum(r['status']!=200 for r in rs),'median_ms':statistics.median(times) if times else None,'p95_ms':times[min(len(times)-1,int(.95*len(times)))] if times else None}
async def main():
 async with httpx.AsyncClient(headers={'Authorization':'Bearer '+os.environ['TOGETHER_API_KEY'],'User-Agent':'open-jev/0.1'},timeout=45) as client:
  probe=read(ROOT/'data/v1/records/dev.jsonl')[0]
  for attempt in range(3):
   try:
    res=await client.post('https://api.together.ai/v1/chat/completions',json=payload(probe,MODEL))
    if res.status_code==200:
     d=res.json();select(d,probe['options']);print('READY '+str(d.get('model')),flush=True);(OUT/'readiness.json').write_text(json.dumps({'attempt':attempt+1,'model':d.get('model'),'usage':d.get('usage')}));break
    print('WAIT '+str(res.status_code)+' '+res.text[:160],flush=True)
   except Exception as e:print('WAIT '+type(e).__name__,flush=True)
   if attempt==2:raise SystemExit('Endpoint not ready after bounded polling')
   await asyncio.sleep(30)
  jobs=[];hashes={}
  for split in ['test','policy_transfer']:
   p=ROOT/'data/v1/records'/f'{split}.jsonl';hashes[split]=hashlib.sha256(p.read_bytes()).hexdigest();jobs.extend((split,r) for r in read(p))
  random.Random(42).shuffle(jobs);sem=asyncio.Semaphore(8);results=[]
  with (OUT/'results.jsonl').open('x') as f:
   async def run(split,row):
    async with sem:
     if sum(r['status']!=200 for r in results)>=3:return
     x={'split':split,'id':row['id'],'source':row['source'],'group_id':row.get('group_id'),'gold':row['answer'],'gold_key':row['answer_key'],'valid':False,'correct':False,'status':0};start=time.perf_counter()
     try:
      res=await client.post('https://api.together.ai/v1/chat/completions',json=payload(row,MODEL));x['status']=res.status_code
      if res.status_code==200:
       d=res.json();ch=d['choices'][0];x.update(output=ch['message'].get('content'),usage=d.get('usage'),logprobs=ch.get('logprobs'),resolved_model=d.get('model'),finish_reason=ch.get('finish_reason'))
       chosen=select(d,row['options']);x.update(predicted=chosen['label'],predicted_key=chosen['key'],valid=ch.get('finish_reason')=='stop',correct=ch.get('finish_reason')=='stop' and chosen['label']==row['answer'])
      else:x['error']=res.text[:300]
     except Exception as e:x['error']=type(e).__name__
     x['latency_ms']=round((time.perf_counter()-start)*1000,2);results.append(x);f.write(json.dumps(x)+'\n');f.flush()
     if len(results)%200==0:print(json.dumps({'completed':len(results),'errors':sum(r['status']!=200 for r in results)}),flush=True)
   await asyncio.gather(*(run(s,r) for s,r in jobs))
 if len(results)!=1300:raise SystemExit(f'Stopped after {len(results)}/1300 due to provider errors; partial results preserved')
 old={r['id']:r for r in read(ROOT/'evaluation/decoding/results.jsonl') if r['mode']=='regex'}
 out={'model':MODEL,'hashes':hashes,'method':'1300 v1 development records; regex per option list, logprobs5, max_tokens8, temperature0, thinking off, concurrency8, shuffle42, pooled HTTP, no quality retries; readiness probes on dev excluded. Historical v2.1 regex baseline reused; latency is not a controlled hardware comparison.'}
 for split in ['test','policy_transfer']:
  rr=[r for r in results if r['split']==split];assert all(r['gold']==old[r['id']]['gold'] for r in rr)
  out[split]=summary(rr);out[split]['by_source']={s:summary([r for r in rr if r['source']==s]) for s in sorted({r['source'] for r in rr})};out[split]['paired']={'fixed':sum(r['correct'] and not old[r['id']]['correct'] for r in rr),'regressed':sum(not r['correct'] and old[r['id']]['correct'] for r in rr),'baseline_correct':sum(old[r['id']]['correct'] for r in rr)}
 out['missing_information']=summary([r for r in results if r['split']=='policy_transfer' and r['gold_key']=='unknown'])
 groups=collections.defaultdict(list)
 for r in results:
  if r['split']=='policy_transfer':groups[r['group_id']].append(r)
 out['policy_groups']={'n':len(groups),'all_correct':sum(all(r['correct'] for r in rs) for rs in groups.values())}
 (OUT/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
asyncio.run(main())
