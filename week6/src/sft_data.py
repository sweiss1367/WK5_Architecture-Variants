"""
Torch-free helpers for building the SFT dataset, so label masking and length handling can be
unit-tested without a GPU.

Loss is computed only on the assistant's JSON (+ end-of-turn token). Every prompt token is
masked with -100, as in Week 3.
"""
import statistics

from appeals_common import build_chat_prompt


def tokenize_records(tokenizer, records):
    items = []
    for r in records:
        prompt = build_chat_prompt(tokenizer, r["notice"], r["summary"], r.get("extra_user_text"))
        target = r["target_json"] + tokenizer.eos_token
        items.append(dict(
            id=r["id"],
            prompt_ids=tokenizer(prompt, add_special_tokens=False)["input_ids"],
            target_ids=tokenizer(target, add_special_tokens=False)["input_ids"],
        ))
    return items


def length_audit(items):
    lens = sorted(len(i["prompt_ids"]) + len(i["target_ids"]) for i in items)
    tgt = sorted(len(i["target_ids"]) for i in items)

    def pct(xs, p):
        return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]

    return dict(
        n=len(items),
        combined_tokens=dict(min=lens[0], median=statistics.median(lens), p90=pct(lens, 0.90), p95=pct(lens, 0.95), max=lens[-1]),
        target_tokens=dict(min=tgt[0], median=statistics.median(tgt), p90=pct(tgt, 0.90), max=tgt[-1]),
    )


def choose_max_seq_len(items, multiple=64, cap=1024):
    longest = max(len(i["prompt_ids"]) + len(i["target_ids"]) for i in items)
    chosen = -(-longest // multiple) * multiple
    if chosen > cap:
        raise SystemExit(f"longest example is {longest} tokens -> {chosen} exceeds the {cap} cap; shorten the data, do not truncate")
    return chosen


def build_examples(items, max_seq_len):
    """Returns (examples, n_truncated). Aborts is the caller's job; this only counts."""
    out, truncated = [], 0
    for it in items:
        ids = it["prompt_ids"] + it["target_ids"]
        labels = [-100] * len(it["prompt_ids"]) + list(it["target_ids"])
        if len(ids) > max_seq_len:
            truncated += 1
        out.append(dict(input_ids=ids[:max_seq_len], labels=labels[:max_seq_len]))
    return out, truncated


def validate_labels(examples):
    counts = [sum(1 for l in e["labels"] if l != -100) for e in examples]
    zero = sum(1 for c in counts if c == 0)
    return dict(n=len(examples), min_supervised=min(counts), max_supervised=max(counts), n_zero_supervised=zero)
