"""Resolve source revisions once and cache public data and tokenizer (no weights)."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi
from datasets import load_dataset
from transformers import AutoTokenizer

SOURCES = {
    "mnli": ("nyu-mll/multi_nli", None),
    "boolq": ("google/boolq", None),
    "banking77": ("legacy-datasets/banking77", None),
    "ag_news": ("fancyzhx/ag_news", None),
    "sst5": ("SetFit/sst5", None),
}
api = HfApi()
lockpath = ROOT / "sources.lock.json"
lock = json.loads(lockpath.read_text()) if lockpath.exists() else {"datasets": {}}
for name, (repo, config) in SOURCES.items():
    if name not in lock["datasets"]:
        info = api.dataset_info(repo)
        card = info.card_data.to_dict() if info.card_data else {}
        lock["datasets"][name] = {"repo": repo, "revision": info.sha, "config": config, "license_metadata": card.get("license", "unspecified"), "url": f"https://huggingface.co/datasets/{repo}"}
        lockpath.write_text(json.dumps(lock, indent=2) + "\n")
    source = lock["datasets"][name]
    ds = load_dataset(repo, name=config, revision=source["revision"])
    print(name, [(k, len(v)) for k,v in ds.items()], flush=True)
if "tokenizer" not in lock:
    repo = "Qwen/Qwen3.5-2B"
    lock["tokenizer"] = {"repo": repo, "revision": api.model_info(repo).sha}
    lockpath.write_text(json.dumps(lock, indent=2) + "\n")
t = AutoTokenizer.from_pretrained(lock["tokenizer"]["repo"], revision=lock["tokenizer"]["revision"])
print("LABELS", {c:t.encode(c, add_special_tokens=False) for c in "ABCDEFGH"})
print("TEMPLATE", repr(t.apply_chat_template([{"role":"user","content":"Pick A or B."}], tokenize=False, add_generation_prompt=True, enable_thinking=False)))
