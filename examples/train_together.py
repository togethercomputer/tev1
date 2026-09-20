"""Recreate the first open-jev training configuration; uploads/starts a job only with --launch."""
import argparse
import json
from pathlib import Path
import time

SETTINGS = dict(
    model="Qwen/Qwen3.5-2B",
    suffix="jev",
    training_method="sft",
    lora=True,
    lora_r=8,
    lora_alpha=16,
    lora_trainable_modules="all-linear",
    n_epochs=2,
    batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    scheduler_num_cycles=0.5,
    warmup_ratio=0.03,
    max_seq_length=4096,
    packing=True,
    n_evals=6,
    n_checkpoints=2,
    early_stopping_enabled=True,
    early_stopping_patience=2,
    early_stopping_warmup_evals=1,
    random_seed=42,
    train_on_inputs="auto",
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
    parser.add_argument("--data", type=Path, default=Path("data/v1/instruction"))
    parser.add_argument("--launch", action="store_true", help="Upload files and start billed training")
    args = parser.parse_args()
    print(json.dumps(SETTINGS, indent=2))
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
    job = client.fine_tuning.create(training_file=train_id, validation_file=dev_id, **SETTINGS)
    print(f"Training job: {job.id}")
    print("Save this ID and follow the job in Together. Do not rerun --launch to poll it.")


if __name__ == "__main__":
    main()
