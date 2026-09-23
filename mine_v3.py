"""Score only fresh training candidates; holdout and dev are never queried."""
import asyncio,json,os,time,sys
from pathlib import Path
import httpx
from examples.decide import payload,select
ROOT=Path(__file__).resolve().parent
MODEL='hassan/Qwen3.5-4B-jev-2-1-4286f48b-c0ad7564'
async def main():
 rows=[json.loads(l) for l in (ROOT/'data/v3/staging/candidates.jsonl').read_text().splitlines()]
 sem=asyncio.Semaphore(8);done=0;start=time.time()
 with (ROOT/'data/v3/mining.jsonl').open('x') as f:
  async with httpx.AsyncClient(headers={'Authorization':'Bearer '+os.environ['TOGETHER_API_KEY'],'User-Agent':'open-jev/0.1'},timeout=45) as c:
   async def run(r):
    nonlocal done
    async with sem:
     result={'id':r['id'],'source':r['source'],'gold':r['answer_key'],'model':MODEL}
     try:
      res=await c.post('https://api.together.ai/v1/chat/completions',json=payload(r,MODEL));result['status']=res.status_code
      if res.status_code==200:
       d=res.json();chosen=select(d,r['options']);result.update(predicted=chosen['key'],correct=chosen['key']==r['answer_key'],logprobs=d['choices'][0].get('logprobs'))
      else:result['error']=res.text[:200]
     except Exception as e:result.update(status=0,error=type(e).__name__)
     f.write(json.dumps(result)+'\n');f.flush();done+=1
     if done%400==0:print(done,len(rows),round(time.time()-start),flush=True)
   await asyncio.gather(*(run(r) for r in rows))
asyncio.run(main())
