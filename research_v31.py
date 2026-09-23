"""New authored research scenarios: no benchmark titles, summaries, or rationales."""
import json,random,string,copy
from pathlib import Path
import build_dataset as v
# Two training cores and one separately authored development core per topic.
# Each core describes the artifact and its evidence, rather than naming its category.
CORES={
'reasoning-methods':[
'A controller revisits uncertain intermediate answers during inference. It allocates a fixed computation budget across candidate solution paths, discards branches that repeat earlier deductions, and stops when independent paths agree. Model weights remain frozen. On logic problems, the controller improves answer accuracy at the same number of generated tokens.',
'A solver selects between short and extended deliberation at inference time using agreement among partial solutions. Its stopping rule is evaluated by varying the permitted number of reasoning steps. The release contains a runtime algorithm and experiments on answer quality versus computation, without a new training objective.',
'A frozen language model maintains several partial proofs and periodically reconciles their assumptions before continuing generation. A search policy chooses which partial proof to expand. The experiments isolate expansion and stopping rules and report solved problems per unit of inference computation.'],
'rl-for-reasoning':[
'A training objective redistributes terminal correctness rewards across intermediate reasoning decisions. An advantage estimator subtracts a baseline formed from alternative derivations of the same answer. Experiments train mathematical problem solvers and measure accuracy after updating their weights. The contribution is the reward estimator and its effect on learning, not a deployed tool-use system.',
'A policy-gradient procedure controls exploration when most generated mathematical solutions fail the final verifier. It mixes successful and near-successful derivations while correcting for their sampling probabilities. Training stability and held-out reasoning accuracy are compared under equal update budgets; the method does not require interactive environments.',
'A reward schedule gradually replaces dense verifier hints with final-answer rewards while optimizing a theorem-solving language model. The analysis measures exploration collapse and gradient variance. The central comparison is between learning objectives for reasoning, with the runtime decoder fixed across trained policies.'],
'agent-training':[
'A curriculum builder generates interactive tasks with increasing numbers of tool calls, trains a policy on the resulting trajectories, and revises the curriculum using its failures. The tasks span calendars, inventory systems, and file operations. Evaluation measures how much the policy improves after each round of environment interaction and weight updates.',
'A post-training pipeline combines successful and failed tool-use trajectories across simulated applications. It assigns credit to actions that recover from mistakes and optimizes the agent policy between rounds. The released recipe improves completion rates in new interactive environments using the same inference harness.',
'A learning framework alternates between generating application states and updating an agent from its attempted solutions. It controls curriculum difficulty using measured failure rates rather than maintaining a library of reusable procedures. Experiments compare policy improvement across several families of interactive software tasks.'],
'agent-benchmarks':[
'A suite of desktop tasks measures whether agents can finish work across multiple applications after interruptions. Hidden state checks verify that forms, files, and settings were changed correctly. The release consists of tasks, resettable environments, and an evaluation harness; fixed agents are compared without retraining them.',
'A computer-use harness synchronizes screenshots, pending actions, and verification checks during long workflows. It recovers from interface changes by inspecting application state before resuming. The study evaluates existing agents on office tasks and reports completion reliability, without introducing a policy-learning algorithm.',
'A browser task collection introduces delayed effects and misleading success messages. Independent validators inspect server-side task state to determine whether work is complete. The study releases a reproducible evaluation environment and compares existing computer-use agents under the same action budgets.'],
'coding-agents':[
'A repository repair system builds a dependency graph from failing tests, identifies candidate source files, and validates proposed changes by rerunning focused tests. It maintains patch candidates rather than a general skill library. Experiments measure resolved software issues and regressions introduced by each patch.',
'A dataset and evaluation harness pair repository feature requests with executable acceptance tests. Tasks require coordinated changes across interfaces and implementation files. Existing code agents are scored on functional correctness and compatibility, with repository-level software changes as the target capability.',
'A code-generation model is trained to preserve interface contracts while editing several related modules. Its evaluation checks compilation, hidden functional tests, and unintended changes. The released model and software benchmarks focus on completing repository modifications.'],
'search-agents':[
'A research agent tracks unanswered subquestions, issues web queries, and stores evidence with source identifiers before composing an answer. It checks whether each claim is supported by the retrieved pages. Evaluation centers on finding and synthesizing information across sources, including cases where initial search results are misleading.',
'A benchmark evaluates long-chain web investigation. Each task requires finding several sources, resolving inconsistent evidence, and producing a supported answer. The release includes source-verification tools and answer checks. Although it is a benchmark of agents, its tasks and scoring concern open-web information seeking.',
'A retrieval system alternates between identifying missing evidence and querying new sources. A benchmark accompanying the system measures source coverage and factual support in research reports. Training and evaluation are specific to evidence gathering rather than general-purpose application control.'],
'agent-skills':[
'A procedural memory stores successful tool-use routines with preconditions and expected effects. During later tasks, the agent retrieves and adapts a routine before taking individual actions. The study measures skill reuse across applications and compares retrieval and storage policies while holding the base policy weights fixed.',
'A library-building method compresses repeated interaction traces into reusable procedures and removes routines that fail transfer checks. A selector decides which procedure to invoke. Reinforcement learning tunes that selector, but the new artifact is the procedure library and its demonstrated reuse across tasks.',
'An experience store records failure explanations and repair procedures from past agent runs. A retrieval mechanism selects relevant procedures for a new task and updates their reliability after use. Experiments measure transfer from remembered workflows rather than improvements from retraining the base agent.'],
'long-context':[
'A cache manager retains tokens that carry evidence needed for later queries and compresses repetitive passages in long documents. It evaluates retrieval and answer accuracy as context length grows. Its memory budget is fixed, and the contribution concerns preserving distant context during generation.',
'A sparse attention layout connects remote document sections through a small set of shared states. Training exposes the model to progressively longer inputs. Experiments test whether answers still depend on evidence near the beginning of the input and compare memory use at matched context lengths.',
'A serving method moves inactive segments of the attention cache out of accelerator memory and restores them before a query needs their evidence. Experiments vary document length and query position while tracking latency and retained information.'],
'diffusion-lms':[
'A text model begins with masked output positions and repeatedly revises uncertain tokens in parallel. A scheduling rule chooses which positions to reveal at each step. Experiments compare the number of refinement steps and final text quality against autoregressive generation.',
'A decoding algorithm proposes blocks of text with a small draft model and verifies them in parallel with a larger model. It changes the proposal length according to recent acceptance rates. The release studies exact sampling behavior and generation speed without changing the target model weights.',
'A sampler for discrete text diffusion revisits low-confidence positions while preserving stable token spans. Its experiments measure generation quality across refinement schedules and parallel update budgets. The central artifact is a decoding procedure for text.'],
'model-efficiency':[
'A runtime compresses the weights and intermediate activations of existing video generators with low-bit arithmetic. It preserves their generation objectives and architecture while replacing numerical kernels. Experiments report memory, latency, and output-quality changes; the contribution is cheaper execution of existing models.',
'A serving scheduler packs requests across accelerators and shares reusable computations between them. The study evaluates language, image, and video models with unchanged weights. Throughput and tail latency under mixed loads are the primary outcomes, rather than a new generator or application capability.',
'A distillation and quantization pipeline reduces the cost of an existing multimodal model while preserving its input-output tasks. Ablations separate compression error from runtime overhead. The release focuses on inference memory and execution speed across deployment settings.'],
'model-architecture':[
'A sequence model replaces part of its attention stack with recurrent state updates and introduces a new interaction between the two blocks. The paper studies scaling across parameter sizes on general language tasks. The contribution is the backbone design, rather than a task-specific agent or training curriculum.',
'A new foundation-model family uses a hierarchical routing mechanism to coordinate expert modules. The report describes its architecture, pretraining recipe, and broad capability evaluation. Coding and reasoning results demonstrate the model family rather than a specialized software agent or reward objective.',
'A general-purpose language backbone combines local token interactions with a persistent recurrent state. Controlled experiments vary the new block design across model sizes. The release is an architectural proposal and a family of pretrained checkpoints.'],
'data-curation':[
'A corpus-construction pipeline detects near-duplicate documents, estimates source diversity, and chooses a mixture for pretraining. Models trained with equal token budgets compare the resulting datasets. The released artifact is the filtered corpus and selection procedure.',
'A synthetic example generator uses independent consistency checks to reject unsupported answers before adding them to a training corpus. The study evaluates data quality and mixture proportions using several downstream models. It does not introduce a new model architecture.',
'A dataset filtering method identifies passages with contradictory supervision and tracks coverage of rare skills. The evaluation holds model architecture and training compute fixed while changing the selected data. The contribution is a reproducible curation procedure.'],
'interpretability-analysis':[
'A controlled study probes hidden states while language models solve multi-step problems. Interventions remove or replace selected representations to test which intermediate facts affect later answers. The findings characterize how reasoning information is represented; no new reasoning algorithm is proposed.',
'A scaling study measures how data diversity, optimization steps, and model size interact during reasoning fine-tuning. Matched experiments isolate correlations and test competing explanations. The contribution is an empirical account of generalization, not a training recipe claimed to improve the solver.',
'A causal analysis follows factual information through successive model layers and replaces selected activations with counterfactual states. The experiments explain when a model recalls a fact and when it fabricates an answer. The output is evidence about model behavior.'],
'vlm-architectures':[
'A shared image-and-text backbone supports both understanding questions and producing images. A joint representation connects visual inputs and generated outputs without separate task-specific models. The report evaluates image recognition, visual reasoning, and image synthesis to establish the general-purpose multimodal architecture.',
'A foundation-model report introduces a visual encoder and a joint training pipeline for language, images, and interactive visual tasks. Agent and image-generation benchmarks are part of a broad evaluation. The release is the multimodal model family and its representation design.',
'A unified multimodal model aligns image features and text tokens in a common sequence representation. Shared weights are evaluated on captioning, visual question answering, and image reconstruction. The contribution concerns general-purpose visual-language modeling.'],
'image-generation':[
'A text-conditioned image editor changes selected objects while preserving unrelated pixels. Its training objective separates editable regions from protected content. Evaluation measures instruction adherence and preservation on individual still images, without temporal video outputs.',
'A still-image generator controls object layout using spatial constraints embedded in its denoising process. The experiments assess placement, visual fidelity, and prompt alignment. Its output is one image per request rather than a time sequence.',
'A photograph editing model follows local text instructions while keeping identity and background consistent. The study compares edit locality and image quality on single-frame tasks.'],
'video-generation':[
'A streaming editor transforms consecutive video frames according to text instructions while carrying appearance information between chunks. Experiments measure temporal consistency and edit fidelity over sequences. Image models provide initialization, but the new behavior is coherent video editing.',
'A world simulator generates future visual sequences in response to control inputs. A temporal consistency objective reduces abrupt object changes across frames. Evaluation measures generated motion and controllability rather than understanding a supplied video.',
'A video synthesis model conditions each new segment on persistent scene information from earlier segments. The study evaluates motion continuity and visual identity during extended generation.'],
'video-understanding':[
'A model identifies when described events occur in existing videos. It combines frame features with temporal relationships and outputs event intervals. The experiments evaluate localization and question answering over supplied clips, without producing new visual sequences.',
'A benchmark presents videos whose answers require comparing distant moments. Models must identify actions and explain temporal ordering from the observed frames. The release evaluates video comprehension rather than video synthesis.',
'A video-question-answering system retrieves relevant moments from a supplied recording before answering. Its study measures grounding accuracy and recognition of event order.'],
'spatial-3d':[
'A reconstruction system converts several calibrated images into a geometric scene representation. It optimizes surfaces and camera relationships and evaluates depth and shape accuracy. Rendering is used to inspect the geometry, not as the primary generated-video task.',
'A point-cloud model detects objects and estimates their spatial relationships from range measurements. The experiments compare geometric localization across sensor conditions. The released representation supports three-dimensional perception.',
'A scene generator produces explicit meshes and object layouts from spatial constraints. Evaluation measures geometric validity, connectivity, and consistency across viewpoints.'],
'document-ocr':[
'A document parser recognizes text, table cells, and reading order in scanned pages. It reconstructs a structured representation that preserves layout relationships. Experiments measure extraction accuracy rather than open-ended image understanding.',
'A model converts page images into markup while separating headings, formulas, and table structure. Its evaluation checks both recognized content and correct document organization. The release focuses on document transcription.',
'A layout-aware recognizer links table headers to their corresponding cells in scanned reports. The evaluation measures structured extraction and reading-order errors.'],
'audio-speech':[
'A speech synthesizer controls speaker identity and timing from text plus a short audio example. Experiments measure pronunciation, intelligibility, and voice preservation. The visual components used during development are auxiliary.',
'A joint audio model transcribes speech and answers questions about environmental sounds. Its training aligns acoustic segments with language representations. Evaluation centers on audio perception and spoken interaction.',
'A music generation model maintains rhythmic structure over successive audio segments. Listening and signal-based evaluations measure continuity and acoustic quality.'],
'robot-policies':[
'A control policy maps camera observations and language instructions to robot joint actions. Demonstrations teach it to manipulate unfamiliar objects, and physical experiments measure task completion. Visual-language representations are components of the embodied policy.',
'A navigation policy learns from simulated trajectories and transfers to a wheeled robot. It must react to obstacles and recover from localization errors. The release evaluates physical action execution rather than a general-purpose multimodal model.',
'A robot learning method combines demonstrations from several manipulators and predicts actions in the control space of each robot. Experiments test transfer to a new embodiment.'],
'science-medicine':[
'A scientific model predicts chemical properties from molecular structures and laboratory measurements. Its evaluation tests accuracy on held-out compounds and physically meaningful constraints. The contribution is a domain-specific predictive capability, not automation of administrative research work.',
'A clinical imaging model combines scan regions and patient measurements to estimate diagnostic findings. The study evaluates case-level predictive performance and robustness under acquisition changes. The model serves a medical task rather than general-purpose visual-language understanding.',
'A scientific surrogate predicts material behavior under changing experimental conditions. Evaluation compares its physical predictions with measurements on unseen specimens.'],
'research-automation':[
'A workflow assistant organizes literature evidence, drafts an experimental plan, and assembles a reproducibility checklist. It uses existing scientific models as tools without changing their predictions. Evaluation measures completeness and verifiability of the produced research artifacts.',
'A system turns a methods section into an executable reproduction plan, runs the supplied analysis, and prepares a report of deviations. Its contribution is automation of the research workflow, not a new predictor for the scientific domain.',
'A research assistant compares claims across papers and builds an evidence-linked review outline. The evaluation checks coverage of the literature and traceability of written conclusions.'],
'safety-alignment':[
'A defense separates instructions from untrusted retrieved content and tests whether agents follow malicious text embedded in documents. The experiments measure attack success and retained task utility. The contribution addresses prompt-injection robustness.',
'A preference-training method reduces harmful compliance while preserving useful responses to benign requests. The evaluation separates refusal behavior from general answer quality. Its contribution concerns intended model behavior rather than ordinary task learning.',
'A red-team benchmark measures whether models reveal protected information after adversarial instruction sequences. The release includes attack categories and reproducible safety measurements.']}

METHODS=['a pretrained language model with fixed weights','a standard transformer implementation','existing visual and text representations','an off-the-shelf retrieval component','a conventional optimizer and a fixed data budget','a compressed baseline model']
CONTEXTS=[
'The implementation reuses {method}. Earlier work on {background} motivates one comparison. That earlier system is a baseline, and the experiments do not claim its contribution as new.',
'The study compares against prior work on {background}. It keeps {method} fixed when evaluating the proposed component. Results include an ablation removing the proposed component and a discussion of settings where gains disappear.',
'The experimental setup includes {method}. The introduction discusses {background} as a neighboring line of work. The authors separate their released artifact from this background and include matched-compute comparisons.',
]
DEV_CONTEXT='For comparison, the experiments include earlier work on {background} and use {method}. The evaluation reports both benefits and failure cases under the same resource limits.'

def build(split,options,question,confusions):
 rows=[];keys=list(CORES)
 edges={k:[] for k in keys}
 for c in confusions:
  if c['predicted'] not in edges[c['gold']]:edges[c['gold']].append(c['predicted'])
 focus={'rl-for-reasoning','agent-skills','search-agents','interpretability-analysis','model-efficiency','reasoning-methods','research-automation','vlm-architectures'}
 for topic in keys:
  indices=[0,1] if split=='train' else [2]
  for core_id in indices:
   core=CORES[topic][core_id];group=f'research-v31:{split}:{topic}:{core_id}'
   count=(20 if topic in focus else 12) if split=='train' else 4
   for i in range(count):
    rng=random.Random(int(v.digest(group+str(i)),16));neighbors=edges[topic] or [k for k in keys if k!=topic];other=neighbors[i%len(neighbors)]
    background=CORES[other][core_id].split('.')[0].lower()
    context=(CONTEXTS[i%3] if split=='train' else DEV_CONTEXT).format(method=METHODS[i%len(METHODS)],background=background)
    # Vary evidence location and presentation; do not expose topic key in the title.
    summary=(core+' '+context) if i%2==0 else (context+' '+core)
    title=['A controlled study of learning systems','Design and evaluation of a computational method','An empirical comparison of system components','A reproducible study of model behavior'][(i//6)%4 if split=='train' else i%4]
    state=f'Title: {title}\narXiv categories: cs.AI\n\nSummary:\n{summary}\n\nLimitations:\nThe study uses a finite evaluation suite; its reported benefits need not extend to every model or application. Reproduction requires matching the stated data and computational budget.'
    for variant in ['canonical24','shuffled24']:
     tagged=copy.deepcopy(options)
     if variant=='shuffled24':rng.shuffle(tagged)
     for j,o in enumerate(tagged):o['label']=string.ascii_uppercase[j]
     answer=next(o['label'] for o in tagged if o['key']==topic)
     rows.append({'id':f'{group}:{i}:{variant}','source':'research_taxonomy_v31','split':split,'kind':'choice','group_id':group,'state':state,'question':question,'options':tagged,'answer':answer,'answer_key':topic,'provenance':{'fictional':True,'generator':'research_v31','core_id':core_id,'topic':topic,'background_topic':other,'context_index':i,'option_variant':variant,'label_method':'authored artifact and evaluation target; no benchmark paper or judge text','core':core,'context':context}})
 return rows
