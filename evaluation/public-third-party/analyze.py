import json,math,statistics,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent
rows=json.loads((OUT/'inputs.json').read_text());ref={r['id']:r for r in rows};manifest=json.loads((OUT/'manifest.json').read_text())
assert hashlib.sha256((OUT/'inputs.json').read_bytes()).hexdigest()==manifest['inputs_sha256']
results={};attempt_stats={}
for name in ['qwen','jev']:
 attempts=[json.loads(l) for l in (OUT/f'{name}.jsonl').read_text().splitlines()];attempt_stats[name]={'attempts':len(attempts),'failed_attempts':sum(not r['ok'] for r in attempts)};assert len([r for r in attempts if r['ok']])==len({r['id'] for r in attempts if r['ok']});rr=list({r['id']:r for r in attempts}.values());assert len(rr)==2087;assert {r['id'] for r in rr}==set(ref)
 for r in rr:
  if r['ok']:assert r['correct']==(r['prediction']==ref[r['id']]['gold'])
 results[name]={r['id']:r for r in rr}
def wilson(k,n):
 z=1.95996398454;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [mid-half,mid+half]
def metrics(rs):
 good=[r for r in rs if r['ok']];n=len(rs);k=sum(r.get('correct',False) for r in rs);labels=sorted({ref[r['id']]['gold'] for r in rs});f1=[]
 for label in labels:
  tp=sum(r['ok'] and r['prediction']==label and ref[r['id']]['gold']==label for r in rs);fp=sum(r['ok'] and r['prediction']==label and ref[r['id']]['gold']!=label for r in rs);fn=sum(ref[r['id']]['gold']==label and (not r['ok'] or r['prediction']!=label) for r in rs);f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
 return {'n':n,'correct':k,'accuracy':k/n,'wilson95':wilson(k,n),'macro_f1':sum(f1)/len(f1),'errors':n-len(good),'median_ms':statistics.median(r['elapsed_ms'] for r in good),'reported_cost_usd':sum(r['response'].get('usage',{}).get('cost',0) for r in good) if all('cost' in r['response'].get('usage',{}) for r in good) else None}
report={'attempt_stats':attempt_stats,'manifest':manifest,'suites':{},'models':{name:sorted({r['response']['model'] for r in rr.values() if r['ok']}) for name,rr in results.items()}}
for suite in ['phishing','tool_risk','ticket_routing']:
 ids=[r['id'] for r in rows if r['suite']==suite];out={name:metrics([rr[i] for i in ids]) for name,rr in results.items()}
 a=sum(results['qwen'][i].get('correct',False) and not results['jev'][i].get('correct',False) for i in ids);b=sum(results['jev'][i].get('correct',False) and not results['qwen'][i].get('correct',False) for i in ids);n=a+b
 out['paired']={'qwen_only_correct':a,'jev_only_correct':b,'mcnemar_exact_p':min(1,2*sum(math.comb(n,k) for k in range(min(a,b)+1))/2**n) if n else 1}
 if suite=='phishing':
  for name in results:
   rr=[results[name][i] for i in ids];out[name]['phishing_recall']=sum(r['ok'] and r['prediction']=='phishing' and ref[r['id']]['gold']=='phishing' for r in rr)/1000;out[name]['false_positive_rate']=sum(r['ok'] and r['prediction']=='phishing' and ref[r['id']]['gold']=='legitimate' for r in rr)/1000
 if suite=='tool_risk':out['by_difficulty']={difficulty:{name:metrics([rr[i] for i in ids if ref[i]['difficulty']==difficulty]) for name,rr in results.items()} for difficulty in ['clear','ambiguous','adversarial']}
 report['suites'][suite]=out
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
md='# Public third-party Jev benchmarks: paired rerun\n\nModels: '+str(report['models'])+'\n\n| Benchmark | N | New Qwen 4B | Jev | Difference |\n|---|---:|---:|---:|---:|\n'
for suite,s in report['suites'].items():
 q=s['qwen'];j=s['jev'];md+=f"| {suite} | {q['n']} | {q['correct']}/{q['n']} ({100*q['accuracy']:.1f}%) | {j['correct']}/{j['n']} ({100*j['accuracy']:.1f}%) | {100*(q['accuracy']-j['accuracy']):+.1f} pp |\n"
md+='\nAttempt accounting: '+str(attempt_stats)+'\n'
md+='\n## Sources\n\n'
for owner,name in [('anisselbd','jev-phishing-bench'),('themsquared','jev-benchmark'),('WallerChen','jev-measured')]:md+=f"- [{owner}/{name}](https://github.com/{owner}/{name}/tree/{manifest['sources'][name]})\n"
md+='\n## Protocol and interpretation\n\n'+manifest['method']+'\n\nQwen: temperature 0, max_tokens 8, thinking disabled, logprobs 5, per-task letter regex, 15 concurrency. Jev: typesafe/jev-1.13 via OpenRouter Decisions API, one native choice per request, concurrency 10. Same input evidence and decision descriptions, different native API wrappers. All source labels are used only locally. Preflight successes are included once. Successful requests reused by ID. One Jev HTTP 520 attempt was retried once; original failure retained in raw logs. No Qwen retries.\n\nBoth models were run afresh; historical published results are not substituted. Phishing uses the source verdict question only, while the original report used nine questions and probability audits. This run does not reproduce its signal classifiers or calibration study. Source risk and routing scores also used different response wrappers and repetition protocols.\n\nPhishNChips contains 2,000 synthetic emails around malicious/legitimate URLs and known construction shortcuts. The source reports a simple URL heuristic at 91.6%; classify this as a dataset result, not a deployed security guarantee. Risk has only 60 hand-labelled examples with subjective/ambiguous classes; routing has 27 deliberately clear examples and a ceiling effect. No pooled headline score because suite sizes and difficulties differ. Wilson intervals and paired counts are in report.json; small score gaps are not proof of broad superiority.\n\nNo exact normalized state matches were found against locally prepared new-v1 training and validation records. This does not audit actual uploads, near-duplicates, or base-model pretraining. No prompt or label-order tuning was performed based on the outcomes.\n\nLatency uses different concurrency/provider paths and is not a controlled speed comparison. Jev cost is API-reported; Qwen dedicated-hosting cost is unknown here. See raw JSONL usage. Both raw responses and complete frozen inputs are saved for reproducibility.\n'
p=report['suites']['phishing'];md+=f"\nPhishing recall: Qwen {100*p['qwen']['phishing_recall']:.1f}%, Jev {100*p['jev']['phishing_recall']:.1f}%. False positive rate: Qwen {100*p['qwen']['false_positive_rate']:.1f}%, Jev {100*p['jev']['false_positive_rate']:.1f}%.\n"
cost=sum(s['jev']['reported_cost_usd'] for s in report['suites'].values());md+=f'\nTotal Jev reported cost: ${cost:.6f}.\n'
(OUT/'README.md').write_text(md)
print(md)
