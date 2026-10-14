"""
E3: train the ONE QLoRA SFT adapter on data/train.jsonl (validation loss on data/val.jsonl).

Adapted from Week 3's src/train_qlora.py (commit 158bc9a, see PROVENANCE.md). Differences:
domain data, torch-free tokenization helpers (src/sft_data.py), automatic max_seq_len with a hard
"no truncation" rule, the trainer log history is saved (Week 3 did not save it), no OOM demo, and
the frozen-manifest check runs first.

Run on a Colab T4:   python src/train_qlora.py
Writes: outputs/qlora_adapter/ (gitignored), logs/experiment_log_train.json,
        logs/trainer_log_history.json, logs/token_length_audit.json, logs/adapter_sha256.json
"""
import argparse
import hashlib
import inspect
import json
import os

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments,
)

import manifest
import sft_data
from appeals_common import load_jsonl
from resource_monitor import GpuMemoryMonitorCallback, StepTimingCallback, check_hardware, profile_run, resolve_precision


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def make_pad_collate(tokenizer):
    def pad_collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = tokenizer.pad_token_id
        ids, labels, attn = [], [], []
        for x in batch:
            pad = max_len - len(x["input_ids"])
            ids.append(x["input_ids"] + [pad_id] * pad)
            labels.append(x["labels"] + [-100] * pad)
            attn.append([1] * len(x["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(attn)}
    return pad_collate


def build_bnb_config(cfg):
    q = cfg["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"], bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )


def load_base(cfg):
    kw = dict(quantization_config=build_bnb_config(cfg), device_map={"": 0})
    dt = getattr(torch, cfg["quantization"]["bnb_4bit_compute_dtype"])
    try:
        return AutoModelForCausalLM.from_pretrained(cfg["model_name"], dtype=dt, **kw)
    except TypeError:  # older transformers
        return AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=dt, **kw)


def verify_target_modules(model, targets):
    names = {n.split(".")[-1] for n, m in model.named_modules()
             if m.__class__.__name__ in ("Linear4bit", "Linear8bitLt", "Linear")}
    missing = [t for t in targets if t not in names]
    if missing:
        raise SystemExit(f"LoRA target modules {missing} not found among linear layers {sorted(names)}")
    print(f"verified LoRA target modules: {targets}")


def build_training_args(cfg, out_dir, smoke=False):
    t = cfg["training"]
    params = inspect.signature(TrainingArguments.__init__).parameters
    kw = dict(
        output_dir=out_dir,
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        num_train_epochs=t["num_train_epochs"], learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"], warmup_steps=t["warmup_steps"],
        max_grad_norm=t["max_grad_norm"], optim=t["optim"], bf16=t["bf16"], fp16=t["fp16"],
        gradient_checkpointing=t["gradient_checkpointing"], logging_steps=5,
        save_strategy="no", report_to="none", seed=t["seed"],
    )
    if smoke:
        kw["max_steps"] = 2
    kw["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "epoch"
    if t["gradient_checkpointing"] and "gradient_checkpointing_kwargs" in params:
        kw["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return TrainingArguments(**kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/qlora_appeals.json")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--smoke", action="store_true",
                    help="2 optimizer steps on 32 train / 8 val examples -> outputs_smoke/ and logs_smoke/ (API check only, never evidence)")
    args = ap.parse_args()
    if args.smoke:
        args.out_dir, args.log_dir = "outputs_smoke", "logs_smoke"

    manifest.verify()
    os.chdir(manifest.root())
    with open(args.config) as f:
        cfg = json.load(f)
    os.makedirs(args.log_dir, exist_ok=True)
    check_hardware(resolve_precision(cfg["training"]))

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- tokenize, audit lengths, choose max_seq_len, refuse to truncate -----------------
    train_recs, val_recs = load_jsonl("data/train.jsonl"), load_jsonl("data/val.jsonl")
    items_train = sft_data.tokenize_records(tok, train_recs)
    items_val = sft_data.tokenize_records(tok, val_recs)
    audit = dict(train=sft_data.length_audit(items_train), val=sft_data.length_audit(items_val))
    max_seq_len = cfg["max_seq_len"]
    if max_seq_len == "auto":
        max_seq_len = sft_data.choose_max_seq_len(items_train + items_val)
    train_ex, tr_trunc = sft_data.build_examples(items_train, max_seq_len)
    val_ex, va_trunc = sft_data.build_examples(items_val, max_seq_len)
    audit.update(max_seq_len=max_seq_len, n_truncated_train=tr_trunc, n_truncated_val=va_trunc)
    with open(os.path.join(args.log_dir, "token_length_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit, indent=2))
    if tr_trunc or va_trunc:
        raise SystemExit("Some examples would be truncated; refusing to train on cut-off targets.")
    if args.smoke:
        train_ex, val_ex = train_ex[:32], val_ex[:8]
    lab = sft_data.validate_labels(train_ex + val_ex)
    print("label check:", lab)
    if lab["n_zero_supervised"]:
        raise SystemExit("examples with zero supervised tokens; aborting")

    # --- model ----------------------------------------------------------------------------
    base = load_base(cfg)
    verify_target_modules(base, cfg["lora"]["target_modules"])
    model = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg["training"]["gradient_checkpointing"] else None,
    )
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"], task_type=cfg["lora"]["task_type"], target_modules=cfg["lora"]["target_modules"],
    ))
    if cfg["training"]["gradient_checkpointing"]:
        model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    params = dict(trainable_params=trainable, total_params=total, trainable_pct=round(100 * trainable / total, 4))
    print(params)

    step_timer = StepTimingCallback()
    trainer_holder = {}

    def do_train():
        trainer = Trainer(
            model=model, args=build_training_args(cfg, os.path.join(args.out_dir, "sft_ckpt"), smoke=args.smoke),
            train_dataset=ListDataset(train_ex), eval_dataset=ListDataset(val_ex),
            data_collator=make_pad_collate(tok),
            callbacks=[step_timer, GpuMemoryMonitorCallback(check_every_n_steps=10, ceiling_gb=14.0)],
        )
        trainer.train()
        trainer_holder["t"] = trainer

    run_cfg = {**cfg["training"], "max_seq_len": max_seq_len, "n_train": len(train_ex), "n_val": len(val_ex), **params}
    profile_run("qlora_sft_appeals", do_train, config=run_cfg, step_timer=step_timer,
                log_path=os.path.join(args.log_dir, "experiment_log_train.json"))

    with open(os.path.join(args.log_dir, "trainer_log_history.json"), "w") as f:
        json.dump(trainer_holder["t"].state.log_history, f, indent=2)

    adapter_dir = os.path.join(args.out_dir, "qlora_adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    sha = {}
    for fn in sorted(os.listdir(adapter_dir)):
        if fn.endswith((".safetensors", ".bin")):
            h = hashlib.sha256()
            with open(os.path.join(adapter_dir, fn), "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            sha[fn] = h.hexdigest()
    with open(os.path.join(args.log_dir, "adapter_sha256.json"), "w") as f:
        json.dump(sha, f, indent=2)
    print(f"saved adapter to {adapter_dir}; sha256 {sha}")


if __name__ == "__main__":
    main()
