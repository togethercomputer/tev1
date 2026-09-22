"""Feedback-informed research-classification augmentation. No benchmark papers trained on."""
import argparse,copy,json,os,random,string,sys
from pathlib import Path
from collections import Counter
PROJECT=Path(os.environ.get('OPEN_JEV_ROOT',str(Path(__file__).resolve().parent)));sys.path.insert(0,str(PROJECT))
import build_v2 as b
v=b.v1
# Frozen rubric: question text and the 24 topic options, copied from request-examples/sample.py,
# which reproduces the original paper-benchmark request format. No local benchmark checkout needed.
# Benchmark paper titles, summaries, and judge answers are not used anywhere in this builder.
QUESTION=('You are a research librarian shelving AI papers into a fixed collection. Choose the ONE topic the paper is fundamentally about — the shelf a reader looking for this work would browse.\n'
 'Judge what the paper CONTRIBUTES, not what it mentions. Ignore topics that merely appear as evaluation benchmarks, baselines, related work, or motivation.\n'
 'A paper that trains a model with reinforcement learning is only rl-for-reasoning if the RL method is the contribution.\n'
 'A benchmark, dataset, or evaluation paper belongs to the topic it evaluates, not to a generic evaluation shelf.\n'
 'arXiv categories are a hint, not the answer: cs.AI and cs.LG are catch-alls, but cs.CV, cs.RO, cs.CL, cs.SE and cs.SD are informative.')
OPTIONS=[{'label':l,'key':k,'description':d} for l,k,d in [
('A','reasoning-methods','Reasoning methods. Techniques that improve how a model deliberates: chain-of-thought, latent or implicit reasoning, self-consistency, thinking budgets, and knowing when to stop thinking.'),
('B','rl-for-reasoning','RL for reasoning. Reinforcement learning applied to train reasoning ability, including RLVR, verifiable rewards, reward design, and exploration for reasoning models.'),
('C','agent-training','Agent training and self-evolution. Training agents: environment synthesis, self-evolving and self-improving agent loops, agent RL frameworks, and agent post-training recipes.'),
('D','agent-benchmarks','Agent benchmarks and computer use. Evaluating agents on realistic tasks: computer-use and GUI agents, tool-use benchmarks, long-horizon task suites, and agent safety evaluation.'),
('E','coding-agents','Coding agents. Software engineering with LLMs: SWE-style agents and harnesses, repository-level bug fixing, code generation models, and coding benchmarks.'),
('F','search-agents','Search and deep research. Information-seeking agents: deep research systems, open-web search agents, retrieval-augmented agents, and their training data and benchmarks.'),
('G','agent-skills','Agent skills and memory. Reusable agent capability: skill acquisition, distillation and libraries, procedural memory, and transferable lessons from trajectories.'),
('H','long-context','Long context and attention. Extending or serving long context: sparse and linear attention architectures, KV-cache compression, memory-augmented generation, and long-context training.'),
('I','diffusion-lms','Diffusion LMs and decoding. Diffusion and non-autoregressive language models, parallel and speculative decoding, and sampling or decoding strategies.'),
('J','model-efficiency','Efficiency and serving. Making models cheaper to train or run: distillation, quantization, sparsity, mixture-of-experts, inference engines, and serving systems.'),
('K','model-architecture','Model architecture. Novel model architectures as the contribution: attention alternatives, linear and state-space models, hybrid or recurrent designs, biologically motivated architectures, and architectural scaling studies.'),
('L','data-curation','Data and synthetic generation. Training data itself: pretraining corpus construction, synthetic data generation, data filtering, deduplication, and data mixture studies.'),
('M','interpretability-analysis','Interpretability and analysis. Understanding why models behave as they do: mechanistic interpretability, sparse autoencoders, probing, scaling and empirical laws, emergent phenomena, and hallucination analyses.'),
('N','vlm-architectures','Vision-language models. Multimodal LLM architecture and training: unified understanding and generation, visual representation alignment, and general-purpose VLM technical reports.'),
('O','image-generation','Image generation and editing. Text-to-image and image-editing models: diffusion transformers, autoregressive image models, and image generation foundation models.'),
('P','video-generation','Video generation and world models. Generating or simulating video: long-form and interactive video generation, controllable video synthesis, and generative video world models.'),
('Q','video-understanding','Video understanding. Comprehending existing video: video question answering, temporal grounding, video reasoning, and video understanding benchmarks.'),
('R','spatial-3d','3D and spatial intelligence. 3D and 4D scene generation, reconstruction, spatial reasoning, point clouds, and 3D detection or perception.'),
('S','document-ocr','Document AI and OCR. Document parsing and intelligence: OCR models, layout and table extraction, document-to-markdown conversion, and visual document understanding.'),
('T','audio-speech','Audio, speech, and omni-modal. Speech recognition and synthesis, music and audio generation, and omni-modal models that jointly handle audio with text or vision.'),
('U','robot-policies','Robot policies. Vision-language-action models, robot manipulation and navigation policies, embodied control, and robot learning from demonstration or simulation.'),
('V','science-medicine','Science and medicine. AI applied to scientific and medical domains: scientific foundation models, discovery and hypothesis generation, clinical reasoning, and medical imaging.'),
('W','research-automation','Research automation. Automating the research workflow itself: paper writing and review assistance, literature synthesis, and paper-to-artifact generation.'),
('X','safety-alignment','Safety and alignment. Making models behave: alignment and preference-tuning methods, jailbreak and adversarial robustness, guardrails and refusal behaviour, and misuse or societal-risk evaluation.'),
]]
# Authored fictional contributions; no titles, summaries, predictions, or judge rationales copied.
CONTRIBUTIONS={
'reasoning-methods':['an inference-time controller that allocates deliberation steps according to uncertainty','a latent-state deliberation procedure that revises intermediate solutions without emitting a long trace','a consistency-based stopping rule that terminates redundant reasoning paths','a search procedure over intermediate reasoning states that improves answer selection'],
'rl-for-reasoning':['a new verifiable reward objective for training multi-step mathematical reasoning','an exploration algorithm for reinforcement learning of reasoning policies','a reward assignment method that credits useful intermediate reasoning steps','an RL training recipe that improves reasoning with sparse final-answer rewards'],
'agent-training':['a curriculum of synthesized interaction environments for training tool-using agents','a trajectory-based agent post-training framework with reinforcement learning across environments','a self-improvement training loop in which agents generate and learn from new interaction tasks','a method for turning successful environment interactions into an agent training curriculum'],
'agent-benchmarks':['a benchmark of graphical-interface agent tasks with hidden execution checks','a task suite measuring tool-using agents over long interaction horizons','an evaluation environment that tests whether computer-use agents recover from failed actions','a benchmark measuring unsafe actions taken by agents during realistic browsing tasks'],
'coding-agents':['a repository-level repair agent that locates faults and validates patches with tests','a software-engineering harness that coordinates code editing and test execution','a code-generation model evaluated on executable programming tasks','a benchmark of repository changes requiring coordinated edits across multiple files'],
'search-agents':['an information-seeking agent that plans web searches and checks source support','a retrieval-augmented research system that iteratively fills gaps in collected evidence','a benchmark of multi-hop open-web research tasks with verifiable source trails','a training dataset for agents that find and synthesize evidence from web sources'],
'agent-skills':['a reusable library of agent procedures distilled from successful interaction traces','a memory retrieval mechanism that transfers lessons from past agent trajectories','a method for acquiring portable tool-use skills across different environments','a distillation process that converts repeated agent behaviors into reusable skills'],
'long-context':['a KV-cache compression method that preserves evidence across long documents','a long-context training recipe that improves retrieval from distant input positions','a sparse attention pattern designed specifically to extend the usable context window','a serving technique for reducing the memory required by long-context generation'],
'diffusion-lms':['a diffusion language model that refines masked text in parallel','a speculative decoding algorithm that validates multiple proposed text tokens at once','a non-autoregressive text generation method with iterative token refinement','a decoding strategy that reduces sequential steps in language generation'],
'model-efficiency':['a weight quantization method for low-memory language-model inference','a distillation method that compresses a larger model into a smaller serving model','an inference engine that schedules heterogeneous requests to improve serving throughput','a pruning method that reduces compute while preserving the original model behavior'],
'model-architecture':['a new recurrent sequence backbone evaluated across general language tasks','a foundation-model family with a new mixture-of-experts routing architecture','a hybrid attention and state-space architecture for general sequence modeling','a new attention replacement studied across language-model scales'],
'data-curation':['a method for removing duplicate documents from pretraining corpora','a synthetic training-data generation pipeline with explicit quality filters','a study of data mixture selection for building general-purpose pretraining corpora','a scalable filtering procedure that selects useful training documents'],
'interpretability-analysis':['a causal probing study of internal representations used during factual recall','a mechanistic analysis that identifies circuits behind a model behavior','an empirical scaling-law analysis explaining how performance changes with data and compute','an analysis of hallucination mechanisms using controlled interventions on model activations'],
'vlm-architectures':['a general-purpose vision-language model that aligns visual and text representations','a unified multimodal foundation-model architecture for image understanding and generation','a training method for visual representation alignment in a general-purpose multimodal LLM','a technical report introducing a general-purpose image-and-text model family'],
'image-generation':['a text-to-image model with improved control of visual composition','an image-editing model that preserves untouched regions under text instructions','a generative model that creates single images with more faithful object placement','a training method that improves fidelity and editability of image synthesis'],
'video-generation':['a video synthesis model that maintains consistent motion across generated frames','a post-training method that improves camera stability in generated video','an interactive generative world model that predicts future visual scenes','a benchmark that measures temporal consistency of video-generation systems'],
'video-understanding':['a benchmark requiring questions to be answered about existing video clips','a temporal grounding method that locates events in observed video','a model for recognizing relationships between events in long videos','an evaluation suite for reasoning about recorded video sequences'],
'spatial-3d':['a method that reconstructs a 3D scene from multiple observations','a generative model for structured 3D objects and spatial layouts','a point-cloud model for 3D object detection and perception','a 4D reconstruction method that estimates scene geometry over time'],
'document-ocr':['a document parsing model that converts scanned pages into structured text','a table extraction method for visually complex documents','an OCR model evaluated on multilingual scanned documents','a benchmark of document-to-markdown conversion with layout preservation'],
'audio-speech':['a speech recognition model robust to overlapping voices','a controllable speech synthesis model','a generative system for music conditioned on text descriptions','an omni-modal foundation model jointly handling audio, vision, and text'],
'robot-policies':['a vision-language-action policy for physical robot manipulation','a robot navigation policy trained from demonstrations','an imitation-learning method for transferring robot control across embodiments','a benchmark evaluating robot policies on embodied manipulation tasks'],
'science-medicine':['a scientific foundation model for predicting molecular properties','a clinical decision-support model evaluated on medical cases','a medical imaging model for interpreting diagnostic scans','a system that proposes and validates scientific hypotheses in a specific laboratory domain'],
'research-automation':['a system that assists researchers in drafting and reviewing papers','a pipeline that turns scientific papers into reproducible executable artifacts','a literature synthesis tool that organizes evidence for researchers','a workflow that automates preparation of scientific reports and peer-review feedback'],
'safety-alignment':['a preference-training method that improves adherence to intended behavior','a defense against prompt injection and jailbreak attacks','a refusal-training procedure evaluated on adversarial requests','a misuse evaluation framework focused on harmful model behavior'],
}
TRAIN_TEMPLATES=[
'This fictional study introduces {main}. The researchers use {method} as an implementation tool. The substantive evaluation tests {outcome}. Discussion compares against {other}, which is background rather than a proposed result.',
'The artifact released in this fictional paper is {main}. To construct it, the team incorporates {method}. Experiments center on {outcome}. The related-work section discusses {other}; that work is not extended here.',
'In this fictional abstract, the authors investigate {outcome}. They present {main}. Their implementation relies on {method}, and their ablations include components inspired by {other}. Those components support the proposed artifact.',
'The central deliverable of this fictional report is {main}. Its experiments measure {outcome}. The team employs {method} during development. Although the introduction motivates the work using {other}, the deliverable addresses the stated problem.',
'We describe {main} in a fictional research report. The implementation combines {method} with ordinary optimization. The reported gains concern {outcome}. References to {other} provide a comparison point, not a second claimed contribution.',
]
DEV_TEMPLATES=['A fictional project uses {method} to build {main}. Review of the experiments shows that success is defined by {outcome}. The project discusses {other} solely as supporting context.','The new artifact in a fictional manuscript is {main}; its evaluation concerns {outcome}. The authors borrowed {method} without proposing changes to it. They mention {other} in the survey section.']
TEST_TEMPLATES=['A fictional manuscript lists {method} among its implementation ingredients and cites {other}. The work it actually develops is {main}. The experimental claims are about {outcome}.','What this fictional study delivers is {main}. Its evidence evaluates {outcome}, while {method} is an off-the-shelf ingredient and {other} appears only in the surrounding discussion.']
METHODS=['reinforcement learning with verifiable rewards','a quantized language model','a multimodal encoder','a web-search tool','a diffusion model','a sparse mixture-of-experts backbone','a pretrained vision-language model','a repository of agent trajectories']
OUTCOMES={
'reasoning-methods':'the quality and efficiency of inference-time deliberation', 'rl-for-reasoning':'learning reasoning ability from the proposed RL objective', 'agent-training':'learning stronger interactive agents across environments','agent-benchmarks':'measurement of agent behavior on the released tasks','coding-agents':'successful code generation and repository modifications','search-agents':'finding supported answers from external sources','agent-skills':'reuse of learned procedures and past experiences','long-context':'retention of evidence in long input contexts','diffusion-lms':'the text-generation or decoding procedure','model-efficiency':'reducing deployment or training resource requirements','model-architecture':'the capabilities of the proposed general-purpose backbone','data-curation':'the quality and selection of training data','interpretability-analysis':'explaining model mechanisms and observed behavior','vlm-architectures':'general-purpose visual and language capabilities','image-generation':'the quality of generated or edited still images','video-generation':'the quality and consistency of generated video','video-understanding':'understanding content in existing video','spatial-3d':'geometric reconstruction, generation, or spatial perception','document-ocr':'extracting structured information from documents','audio-speech':'speech, music, or joint audio capabilities','robot-policies':'successful embodied control','science-medicine':'scientific or clinical task performance','research-automation':'supporting the scientific research workflow','safety-alignment':'robust adherence to intended behavioral constraints'}
SEED='paper-feedback-v21-1'
def rng(x):return random.Random(int(v.digest(SEED+str(x)),16))
def build_research(split,groups_per_topic,options,question):
 rows=[];keys=list(CONTRIBUTIONS);templates=TRAIN_TEMPLATES if split=='train' else DEV_TEMPLATES if split=='dev' else TEST_TEMPLATES
 for topic in keys:
  for i in range(groups_per_topic):
   r=rng((split,topic,i)); other=r.choice([k for k in keys if k!=topic]);group=f'research-v21:{split}:{topic}:{i}'
   template=r.choice(templates);method=r.choice(METHODS)
   # Counterfactual pairs swap contribution/background roles; order augmentation stays in one split.
   for side,(gold,distractor) in enumerate([(topic,other),(other,topic)]):
    main=r.choice(CONTRIBUTIONS[gold]);background=r.choice(CONTRIBUTIONS[distractor])
    state={'document_type':'Fictional research summary for classification practice','summary':template.format(main=main,method=method,outcome=OUTCOMES[gold],other=background)}
    for variant,n in enumerate([24,24,12,8]):
     # Variant 0 preserves the benchmark's canonical topic order; others break position memorization.
     chosen=list(options) if n==24 else [o for o in options if o['key'] in {gold,distractor}]+r.sample([o for o in options if o['key'] not in {gold,distractor}],n-2)
     if variant:r.shuffle(chosen)
     tagged=[dict(o,label=string.ascii_uppercase[j]) for j,o in enumerate(chosen)]
     answer=next(o['label'] for o in tagged if o['key']==gold)
     rows.append({'id':f'{group}:{side}:{variant}','group_id':group,'source':'research_taxonomy_v21','split':split,'kind':'choice','state':state,'question':question,'options':tagged,'answer':answer,'answer_key':gold,'provenance':{'generator':SEED,'v2_role':'synthetic_research_feedback','fictional':True,'main_contribution_key':gold,'background_key':distractor,'contribution':main,'background':background,'template':template,'method':method,'option_variant':variant,'pair_side':side,'template_split':split,'label_method':'authored contribution/background contrast; construction labels, not real-paper annotation'}})
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=PROJECT/'data/v2.1');args=ap.parse_args()
 if args.output.exists():raise SystemExit('Output exists')
 options=OPTIONS;question=QUESTION
 assert len(options)==24 and {o['key'] for o in options}==set(CONTRIBUTIONS)
 base=PROJECT/'data/v2';oldmanifest=json.loads((base/'manifest.json').read_text())
 parts={s:b.read(base/'records'/f'{s}.jsonl') for s in oldmanifest['splits']}
 extra={'train':build_research('train',20,options,question),'dev':build_research('dev',4,options,question),'research_challenge':build_research('research_challenge',4,options,question)}
 for s,rows in extra.items():parts.setdefault(s,[]);parts[s]+=rows
 # v1 integrity is generic about option counts, and validates grouped cross-split leakage.
 integrity=v.validate_partitions(parts)
 # Exact repeated prompts are allowed only as documented within-group option permutations.
 group_splits={};prompt_splits={};template_sets={s:set() for s in extra}
 for s,rows in extra.items():
  for row in rows:
   p=row['provenance'];assert row['answer_key']==p['main_contribution_key']!=p['background_key'];assert p['contribution'] in CONTRIBUTIONS[row['answer_key']]
   assert p['background'] in CONTRIBUTIONS[p['background_key']]
   assert row['state']['summary']==p['template'].format(main=p['contribution'],method=p['method'],outcome=OUTCOMES[row['answer_key']],other=p['background'])
   assert next(o['key'] for o in row['options'] if o['label']==row['answer'])==row['answer_key']
   template_sets[s].add(p['template']);group_splits.setdefault(row['group_id'],s);assert group_splits[row['group_id']]==s
   prompt=v.digest(v.messages(row)[:-1]);prompt_splits.setdefault(prompt,s);assert prompt_splits[prompt]==s
 assert not template_sets['train']&(template_sets['dev']|template_sets['research_challenge'])
 from transformers import AutoTokenizer
 pin=oldmanifest['sources']['tokenizer'];tok=AutoTokenizer.from_pretrained(pin['repo'],revision=pin['revision'],local_files_only=True)
 ids={c:tok.encode(c,add_special_tokens=False) for c in string.ascii_uppercase[:24]};assert all(len(x)==1 for x in ids.values())
 manifest={'version':'2.1','seed':SEED,'builder_sha256':v.digest(Path(__file__).read_text()),'base_manifest_sha256':v.digest((base/'manifest.json').read_text()),'sources':oldmanifest['sources'],'integrity':integrity,'label_token_ids':ids,'max_tokens':2048,'eos_token':tok.eos_token,'files':{},'splits':{},'feedback':{'benchmark_matches':579,'benchmark_n':891,'jev_matches':813,'reference':'two-model consensus, not human ground truth','taxonomy_sha256':v.digest(options),'question_sha256':v.digest(question),'rubric_source':'frozen inline copy from request-examples/sample.py, which reproduces the original paper-benchmark request format; no local benchmark checkout required','papers_or_judge_labels_used_for_training':0,'new_supervision':'authored fictional summaries; construction labels','limitations':'template augmentation, not thousands of independent papers; shared contribution vocabulary across splits; research_challenge is a controlled rendering test, not an independent real-paper benchmark'}}
 for split,rows in parts.items():
  if split in extra:rng('export'+split).shuffle(rows)
  chats=[];raw=[];lengths=[]
  for row in rows:
   chat=v.messages(row);prefix=tok.apply_chat_template(chat[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   pre=tok.encode(prefix,add_special_tokens=False);assert tok.encode(prefix+row['answer'],add_special_tokens=False)==pre+ids[row['answer']]
   completion=row['answer']+tok.eos_token;length=len(tok.encode(prefix+completion,add_special_tokens=False));assert length<=2048,(row['id'],length)
   row['token_count']=length;lengths.append(length);chats.append({'messages':chat});raw.append({'prompt':prefix,'completion':completion})
  for folder,items in [('records',rows),('sft',chats),('instruction',raw)]:
   path=args.output/folder/f'{split}.jsonl';v.write_jsonl(path,items);manifest['files'][str(path.relative_to(args.output))]=v.digest(path.read_text())
  manifest['splits'][split]={'count':len(rows),'sources':dict(Counter(x['source'] for x in rows)),'tokens':{'total':sum(lengths),'max':max(lengths)},'research_answer_letters':dict(sorted(Counter(x['answer'] for x in rows if x['source']=='research_taxonomy_v21').items()))}
  print('Exported',split,manifest['splits'][split],flush=True)
 v.write_json(args.output/'manifest.json',manifest)
 v.write_json(args.output/'research_samples.json',extra['train'][:8])
 print('COMPLETE',args.output,flush=True)
if __name__=='__main__':main()
