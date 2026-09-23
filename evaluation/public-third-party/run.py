import asyncio,json,os,time,sys,hashlib
from pathlib import Path
import httpx
ROOT=Path('/Users/hassan/dev/open-jev');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from examples.decide import payload
MODELS={'qwen':('hassan/Qwen3.5-4B-v1-new-69617472-bdc3c2fc','TOGETHER_API_KEY','https://api.together.ai/v1/chat/completions',15),'jev':('typesafe/jev-1.13','OPENROUTER_API_KEY','https://openrouter.ai/api/alpha/decisions',10)}
raw=(OUT/'inputs.json').read_bytes();manifest=json.loads((OUT/'manifest.json').read_text());assert hashlib.sha256(raw).hexdigest()==manifest['inputs_sha256']
rows=json.loads(raw);rows.sort(key=lambda r:({'ticket_routing':0,'tool_risk':1,'phishing':2}[r['suite']],r['id']))
async def run(name):
 model,key,url,concurrency=MODELS[name];key=os.environ[key];file=OUT/f'{name}.jsonl'
 prior=[json.loads(l) for l in file.read_text().splitlines()] if file.exists() else []
 done={r['id'] for r in prior if r['ok'] or '--retry-failed' not in sys.argv};todo=[r for r in rows if r['id'] not in done]
 if '--probe' in sys.argv:todo=todo[:1]
 queue=asyncio.Queue()
 for row in todo:queue.put_nowait(row)
 errors=0;completed=0
 async with httpx.AsyncClient(headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},timeout=60) as client:
  with file.open('a') as f:
   async def worker():
    nonlocal errors,completed
    while not queue.empty() and errors<3:
     row=queue.get_nowait();task=row['task'];out={'id':row['id'],'suite':row['suite'],'model':model,'ok':False,'status':0};start=time.perf_counter()
     if name=='qwen':body=payload(task,model)
     else:body={'model':model,'state':task['state'],'questions':{'decision':{'type':'choice','instructions':task['question'],'criteria':{o['key']:o['description'] for o in task['options']}}}}
     try:
      res=await client.post(url,json=body);out['status']=res.status_code
      if res.status_code!=200:raise ValueError(f'HTTP {res.status_code}: {res.text[:200]}')
      data=res.json();out['response']=data
      if name=='qwen':
       ch=data['choices'][0];answer=ch['message']['content'].strip();assert ch['finish_reason']=='stop';prediction=next(o['key'] for o in task['options'] if o['label']==answer)
      else:prediction=data['answers']['decision']['choice'];assert prediction in [o['key'] for o in task['options']]
      out.update(ok=True,prediction=prediction,correct=prediction==row['gold'])
     except Exception as e:errors+=1;out['error']=str(e);print(name+' ERROR '+str(e),flush=True)
     out['elapsed_ms']=(time.perf_counter()-start)*1000;f.write(json.dumps(out)+'\n');f.flush();completed+=1
     if completed%200==0:print(json.dumps({'model':name,'completed':completed,'pending':queue.qsize(),'errors':errors}),flush=True)
   await asyncio.gather(*(worker() for _ in range(min(concurrency,len(todo)))))
 print(json.dumps({'model':name,'completed_this_run':completed,'pending':queue.qsize(),'errors':errors}),flush=True)
async def main():await asyncio.gather(*(run(n) for n in MODELS))
asyncio.run(main())
