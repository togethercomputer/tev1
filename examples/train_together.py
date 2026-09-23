"""Preview the tev1-4B-experimental 4B starting recipe; uploads/starts a job only with --launch."""
import argparse
import json
from pathlib import Path
import time

SETTINGS = dict(
    model="Qwen/Qwen3.5-4B",
    suffix="tev1-4B-experimental",
    training_method="sft",
    lora=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0,
    lora_trainable_modules="all-linear",
    n_epochs=1,
    batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    scheduler_num_cycles=0.5,
    min_lr_ratio=0,
    max_grad_norm=1,
    weight_decay=0,
    warmup_ratio=0.03,
    max_seq_length=2048,
    packing=True,
    n_evals=6,
    n_checkpoints=3,
    early_stopping_enabled=False,
    random_seed=42,
    train_on_inputs=False,
)


def upload_ready(client, path, timeout=600):
    uploaded = client.files.upload(file=str(path), purpose="fine-tune", check=True)
    print(f"Uploaded {path.name}: {uploaded.id}", flush=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        meta = client.files.retrieve(uploaded.id)
        status = getattr(meta, "processing_status", None)
        if status == "COMPLETED":
            return uploaded.id
        if status in {"FAILED", "INVALID_FORMAT"}:
            raise RuntimeError(f"File ingestion failed for {uploaded.id}: {status}")
        time.sleep(5)
    raise TimeoutError(f"File {uploaded.id} was not ready within {timeout}s; inspect it before retrying")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/new-v1/instruction"))
    parser.add_argument("--launch", action="store_true", help="Upload files and start billed training")
    parser.add_argument("--settings", type=Path, help="JSON object of SDK settings overriding the defaults")
    args = parser.parse_args()
    settings = dict(SETTINGS)
    if args.settings:
        overrides = json.loads(args.settings.read_text())
        if not isinstance(overrides, dict) or set(overrides) - SETTINGS.keys():
            parser.error("Settings must be an object containing only supported recipe keys")
        settings.update(overrides)
    print(json.dumps(settings, indent=2))
    if not args.launch:
        print("Preview only. Add --launch to upload train/dev and start a new billed job.")
        return
    for name in ["train.jsonl", "dev.jsonl"]:
        if not (args.data / name).is_file():
            parser.error(f"Missing {args.data / name}")
    from together import Together
    client = Together()  # Reads TOGETHER_API_KEY.
    train_id = upload_ready(client, args.data / "train.jsonl")
    dev_id = upload_ready(client, args.data / "dev.jsonl")
    job = client.fine_tuning.create(training_file=train_id, validation_file=dev_id, **settings)
    print(f"Training job: {job.id}")
    print("Save this ID and follow the job in Together. Do not rerun --launch to poll it.")


if __name__ == "__main__":
    main()
