"""
E0 preflight (run first, on the Colab T4).

Verifies the frozen manifest, records the exact package/GPU versions the run used, confirms the
tokenizer is reachable and shows what one real prompt looks like and how many tokens it has.

Writes: results/env_manifest.json, logs/pip_freeze.txt
Exit code is non-zero if there is no GPU or the manifest does not verify.
"""
import importlib.metadata as md
import json
import os
import subprocess
import sys

import manifest

PACKAGES = ["torch", "transformers", "peft", "bitsandbytes", "accelerate", "numpy", "pandas"]


def version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def main():
    man = manifest.verify()
    base = manifest.root()
    os.chdir(base)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU: switch the Colab runtime to a T4 GPU.")
    props = torch.cuda.get_device_properties(0)
    info = dict(
        python=sys.version.split()[0],
        packages={p: version(p) for p in PACKAGES},
        gpu=props.name,
        gpu_total_memory_gib=round(props.total_memory / 2**30, 2),
        cuda_capability=f"{props.major}.{props.minor}",
        torch_cuda=torch.version.cuda,
        bf16_supported_flag=torch.cuda.is_bf16_supported(),
        manifest_sha256=manifest.manifest_sha256(),
        manifest_created_utc=man["created_utc"],
    )
    missing = [p for p, v in info["packages"].items() if v is None]
    if missing:
        raise SystemExit(f"missing packages: {missing}")

    from appeals_common import build_chat_prompt, load_jsonl
    from transformers import AutoTokenizer
    cfg = json.load(open("configs/qlora_appeals.json"))
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    rec = load_jsonl("data/train.jsonl")[0]
    prompt = build_chat_prompt(tok, rec["notice"], rec["summary"])
    info["example_prompt_tokens"] = len(tok(prompt, add_special_tokens=False)["input_ids"])
    info["example_target_tokens"] = len(tok(rec["target_json"] + tok.eos_token, add_special_tokens=False)["input_ids"])
    info["tokenizer_eos_token"] = tok.eos_token
    info["tokenizer_pad_token"] = tok.pad_token
    print("----- example prompt (train record %s) -----" % rec["id"])
    print(prompt)
    print("----- example target -----")
    print(rec["target_json"])

    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    with open("logs/pip_freeze.txt", "w") as f:
        f.write(freeze)
    with open("results/env_manifest.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
    print("PREFLIGHT OK")


if __name__ == "__main__":
    main()
