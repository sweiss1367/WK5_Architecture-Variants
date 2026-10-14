"""
Batched generation with the trained adapter for one stage of the audit (GPU).

    python src/generate_batched.py --stage task        # val + eval_h, greedy
    python src/generate_batched.py --stage probes      # 100 probes: greedy + 3 sampled seeds
    python src/generate_batched.py --stage fairness    # 600 matched prompts x 3 sampled seeds
    python src/generate_batched.py --stage probes --smoke   # 8 records only, written to a separate dir

Stages, inputs and decoding settings come from configs/decoding.json (frozen). Decoding parameters
are set explicitly on every call so the model's bundled generation_config cannot leak in; the
effective config is saved beside the outputs. Seeds change only the sampling random state of the
single adapter (generation variability, not training-seed variability).

Resumable: records are appended to <file>.partial after every batch. Sampling is seeded per batch
(seed * 100000 + batch index), so resuming after a disconnect gives the same outputs as an
uninterrupted run on the same hardware and library versions.

Output record: {id, raw_output, finished, n_new_tokens, run, decoding}. `finished` is False when no
end-of-turn token appeared within max_new_tokens (such an output is scored as invalid).
"""
import argparse
import hashlib
import json
import os
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

import manifest
from appeals_common import build_chat_prompt, load_jsonl
from resource_monitor import check_hardware

SHORT = {"data/val.jsonl": "val", "data/eval_h.jsonl": "eval_h",
         "probes/probe_suite.jsonl": "probes", "data/fairness_matched.jsonl": "fairness"}


def load_model(cfg, adapter_dir):
    q = cfg["quantization"]
    dt = getattr(torch, q["bnb_4bit_compute_dtype"])
    bnb = BitsAndBytesConfig(load_in_4bit=q["load_in_4bit"], bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
                             bnb_4bit_compute_dtype=dt, bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"])
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kw = dict(quantization_config=bnb, device_map={"": 0})
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], dtype=dt, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=dt, **kw)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tok


def gen_kwargs(dec, run):
    base = dec["greedy"] if run == "greedy" else dec["sample"]
    kw = {k: v for k, v in base.items() if v is not None}
    kw["max_new_tokens"] = dec["max_new_tokens"]
    return kw


def eos_id_set(model, tok):
    ids = {tok.eos_token_id}
    g = getattr(model, "generation_config", None)
    e = getattr(g, "eos_token_id", None)
    if isinstance(e, int):
        ids.add(e)
    elif isinstance(e, (list, tuple)):
        ids.update(e)
    return {i for i in ids if i is not None}


@torch.no_grad()
def run_file(model, tok, records, run, seed, dec, out_path, batch_size):
    part = out_path + ".partial"
    done = {}
    if os.path.exists(part):
        done = {r["id"]: r for r in load_jsonl(part)}
        print(f"  resuming {out_path}: {len(done)}/{len(records)} already done")
    kw = gen_kwargs(dec, run)
    eos = eos_id_set(model, tok)
    t0, new_tokens = time.time(), 0
    with open(part, "a", encoding="utf-8") as fh:
        for b, start in enumerate(range(0, len(records), batch_size)):
            chunk = records[start:start + batch_size]
            if all(r["id"] in done for r in chunk):
                continue
            if run != "greedy":
                set_seed(seed * 100000 + b)
            prompts = [build_chat_prompt(tok, r["notice"], r["summary"], r.get("extra_user_text")) for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            out = model.generate(**enc, pad_token_id=tok.pad_token_id, **kw)
            for r, row in zip(chunk, out[:, enc["input_ids"].shape[1]:].tolist()):
                cut = next((i for i, t in enumerate(row) if t in eos), None)
                finished = cut is not None
                ids = row[:cut] if finished else row
                n_tok = (cut + 1) if finished else len(row)
                new_tokens += n_tok
                rec = dict(id=r["id"], raw_output=tok.decode(ids, skip_special_tokens=True).strip(),
                           finished=finished, n_new_tokens=n_tok, run=run, decoding=kw)
                done[r["id"]] = rec
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  {run}: {min(start + batch_size, len(records))}/{len(records)}")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(done[r["id"]], ensure_ascii=False) + "\n")
    os.remove(part)
    return dict(file=out_path, n=len(records), wall_sec=round(time.time() - t0, 1), new_tokens=new_tokens,
                n_unfinished=sum(1 for r in done.values() if not r["finished"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["task", "probes", "fairness"])
    ap.add_argument("--adapter", default="outputs/qlora_adapter")
    ap.add_argument("--config", default="configs/qlora_appeals.json")
    ap.add_argument("--decoding", default="configs/decoding.json")
    ap.add_argument("--out-dir", default="results/generations")
    ap.add_argument("--smoke", action="store_true", help="first 8 records only; writes to <out-dir>_smoke (never evidence)")
    args = ap.parse_args()

    man = manifest.verify()
    os.chdir(manifest.root())
    cfg, dec = json.load(open(args.config)), json.load(open(args.decoding))
    out_dir = args.out_dir + ("_smoke" if args.smoke else "")
    os.makedirs(out_dir, exist_ok=True)
    check_hardware("fp16")

    model, tok = load_model(cfg, args.adapter)
    adapter_hash = {}
    for fn in sorted(os.listdir(args.adapter)):
        if fn.endswith((".safetensors", ".bin")):
            h = hashlib.sha256()
            with open(os.path.join(args.adapter, fn), "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            adapter_hash[fn] = h.hexdigest()

    stage = dec["stages"][args.stage]
    jobs = []
    for path in stage["inputs"]:
        recs = load_jsonl(path)
        if args.smoke:
            recs = recs[:8]
        for run in stage["runs"]:
            seed = int(run.replace("seed", "")) if run.startswith("seed") else 0
            jobs.append((SHORT[path], path, recs, run, seed))

    summaries = []
    for short, path, recs, run, seed in jobs:
        print(f"[{args.stage}] {short} / {run}")
        summaries.append(run_file(model, tok, recs, run, seed, dec, os.path.join(out_dir, f"{short}_{run}.jsonl"), dec["batch_size"]))

    meta = dict(
        stage=args.stage, smoke=args.smoke, manifest_sha256=manifest.manifest_sha256(),
        adapter_dir=args.adapter, adapter_sha256=adapter_hash,
        effective_model_generation_config=getattr(model, "generation_config", None).to_dict()
        if getattr(model, "generation_config", None) is not None else None,
        decoding_used={"greedy": gen_kwargs(dec, "greedy"), "sample": gen_kwargs(dec, "sample")},
        torch=torch.__version__, gpu=torch.cuda.get_device_name(0), jobs=summaries,
        note="Seeds change only the sampling random state of one adapter: generation variability, not training-seed variability.",
    )
    with open(os.path.join(out_dir, f"_run_meta_{args.stage}.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
