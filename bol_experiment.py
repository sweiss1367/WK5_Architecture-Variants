# %% [markdown]
# # Bill-of-lading classifier: dataset + baseline Transformer
# Part 1 of the architecture-variants experiment. Contents: environment info, synthetic
# dataset, tokenizer, baseline model, training with seeds 0/1/2, test evaluation.
# Every number printed below comes from running this file.

# %%
import json, math, os, platform, random, re, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# %% [markdown]
# ## 0. Environment verification (GPU required)
# This notebook stops if CUDA is not available. In Colab: Runtime > Change runtime type > GPU.

# %%
# Must be set before the first CUDA call. Used with warn_only determinism below.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

print("Python version         :", sys.version.split()[0])
print("PyTorch version        :", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. This experiment must run on a GPU (Colab: Runtime > Change runtime type > "
        "T4 GPU), then restart and run all cells. It will not fall back to CPU.")

DEVICE = torch.device("cuda")
_p = torch.cuda.get_device_properties(0)
print("GPU name               :", torch.cuda.get_device_name(0))
print("CUDA version (torch)   :", torch.version.cuda)
print("GPU compute capability :", f"{_p.major}.{_p.minor}")
print("GPU memory (GB)        :", round(_p.total_memory / 1e9, 2))

ENV = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
       "device": str(DEVICE), "cuda_available": True, "gpu_name": torch.cuda.get_device_name(0),
       "cuda_version": torch.version.cuda, "compute_capability": f"{_p.major}.{_p.minor}",
       "gpu_memory_gb": round(_p.total_memory / 1e9, 2)}

# Best-effort determinism. GPU kernels are not guaranteed bitwise identical to the CPU run
# (or across GPU types), so expect small numeric differences from the CPU reference results.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Synthetic bill-of-lading dataset
# Six classes. Two are rare (hazmat, oversize/heavy), 4% each.
#
# **Input corruption (train, val AND test):** OCR character errors, missing fields,
# odd spacing, truncated values, and a conflicting instruction borrowed from another class.
# The class label is the underlying shipment type and is never changed by the corruption.
#
# **Label noise:** 3% of *training* labels are flipped to a different random class.
# Validation and test labels are clean.

# %%
CLASSES = ["general_dry", "retail_pallet", "refrigerated", "fragile_high_value", "hazmat", "oversize_heavy"]
PROPS = [0.34, 0.26, 0.20, 0.12, 0.04, 0.04]
RARE_IDS = [CLASSES.index("hazmat"), CLASSES.index("oversize_heavy")]
N_TOTAL = 15_000
SPLIT = {"train": 0.70, "val": 0.10, "test": 0.20}
DATA_SEED = 42
TRAIN_LABEL_NOISE = 0.03
MAX_LEN = 128

# Corruption settings, fixed before looking at any model results
P_DROP_FIELD = 0.25       # each class-specific field is missing with this probability
P_DROP_COMMODITY = 0.05
P_CONFLICT = 0.15         # an instruction typical of another class is appended
P_TRUNCATE = 0.06         # a field is cut off mid-value
OCR_LEVELS = [0.0, 0.01, 0.03, 0.06]     # per-character error rate, one level per document
SPACE_LEVELS = [0.0, 0.03, 0.10]

NAMES = ["ACME", "NORTHSTAR", "BLUE RIDGE", "PACIFIC", "MIDWEST", "SUMMIT", "HARBOR", "KEYSTONE", "LONE STAR", "ATLAS"]
SUFFIX = ["SUPPLY CO", "DISTRIBUTION", "INDUSTRIES", "FOODS", "LOGISTICS", "MFG", "TRADING", "WHSE"]
CITIES = ["DALLAS TX", "MEMPHIS TN", "RENO NV", "ATLANTA GA", "COLUMBUS OH", "TAMPA FL", "DENVER CO", "NEWARK NJ"]
GENERIC_INSTR = ["CALL BEFORE DELIVERY", "LIFTGATE REQUIRED", "APPT REQUIRED", "DELIVER 8AM-4PM", "DO NOT STACK", "INSIDE DELIVERY"]

CLASS_INSTR = {   # used as the conflicting instruction pool
    0: ["NO SPECIAL HANDLING", "STACKABLE", "DRY VAN"],
    1: ["STORE REPLENISHMENT", "ROUTING GUIDE COMPLIANT", "ASN SENT"],
    2: ["KEEP FROZEN", "DO NOT FREEZE", "REEFER CONTINUOUS"],
    3: ["FRAGILE", "SIGNATURE REQUIRED", "HANDLE WITH CARE", "THIS SIDE UP"],
    4: ["HAZMAT", "PLACARD REQUIRED", "FLAMMABLE"],
    5: ["OVERSIZE LOAD", "ESCORT REQUIRED", "FLATBED TARP"],
}


def _party(r):
    return f"{r.choice(NAMES)} {r.choice(SUFFIX)} {r.choice(CITIES)}"


def make_fields(c, r):
    """Returns (always_present_fields, droppable_class_fields, commodity_field)."""
    base = [f"BOL {r.randint(100000, 999999)}", f"DATE {r.randint(1, 12):02d}/{r.randint(1, 28):02d}/25",
            f"SHIPPER {_party(r)}", f"CONSIGNEE {_party(r)}", f"CARRIER {r.choice(NAMES)} FREIGHT",
            r.choice(["PREPAID", "COLLECT", "3RD PARTY"])]
    sig, com = [], None
    pcs = r.randint(1, 40)
    if c == 0:
        com = "DESC " + r.choice(["PAPER PRODUCTS", "STEEL BRACKETS", "EMPTY PLASTIC BOTTLES", "CARDBOARD CARTONS",
                                  "PVC FITTINGS", "FASTENERS", "CERAMIC TILE", "LUMBER", "TIRES"])
        sig = [f"{pcs} {r.choice(['PLT', 'CTN', 'BDL'])}", f"WT {r.randint(2000, 44000)} LB",
               f"FREIGHT CLASS {r.choice([50, 55, 60, 65, 70, 77.5, 85])}", f"NMFC {r.randint(10000, 160000)}",
               r.choice(CLASS_INSTR[0])]
    elif c == 1:
        com = "DESC " + r.choice(["APPAREL", "HOUSEWARES", "SMALL APPLIANCES", "TOYS", "HOME TEXTILES", "PET SUPPLIES"])
        sig = [f"{pcs} PLT", f"WT {r.randint(2000, 30000)} LB", f"PO {r.randint(10000000, 99999999)}",
               f"DC {r.randint(1, 99):02d}", r.choice(["SSCC LABELS APPLIED", "PALLET LABELS ATTACHED"]),
               r.choice(CLASS_INSTR[1]), "DELIVERY APPT REQUIRED"]
    elif c == 2:
        frozen = r.random() < 0.5
        com = "DESC " + r.choice(["FROZEN POULTRY", "ICE CREAM", "FROZEN SEAFOOD"] if frozen
                                 else ["FRESH PRODUCE", "DAIRY", "CHILLED MEAT", "FLORAL"])
        sig = [f"{pcs} PLT", f"WT {r.randint(8000, 42000)} LB",
               f"SET TEMP {r.choice([-20, -18, -15]) if frozen else r.choice([34, 36, 38])} {'C' if frozen else 'F'}",
               "REEFER", r.choice(CLASS_INSTR[2])]
    elif c == 3:
        com = "DESC " + r.choice(["LCD MONITORS", "GLASSWARE", "LAPTOPS", "FRAMED ARTWORK", "MEDICAL DEVICES", "JEWELRY"])
        sig = [f"{pcs} CTN", f"WT {r.randint(200, 9000)} LB", f"DECLARED VALUE ${r.randint(20, 900) * 1000}",
               r.choice(CLASS_INSTR[3]), r.choice(CLASS_INSTR[3])]
    elif c == 4:
        com = "DESC " + r.choice(["GASOLINE", "FLAMMABLE LIQUID NOS", "PAINT", "LITHIUM ION BATTERIES", "ETHANOL SOLUTION",
                                  "NITRIC ACID", "SODIUM HYDROXIDE SOLUTION", "AEROSOLS"])
        sig = [f"{pcs} DRM", f"WT {r.randint(500, 30000)} LB", f"UN{r.choice([1203, 1993, 1263, 3480, 1170, 2031, 1824, 1950])}",
               f"CLASS {r.choice(['3', '8', '9', '2.1', '4.1'])}", f"PG {r.choice(['II', 'III'])}",
               "PLACARD REQUIRED", f"24HR EMERGENCY CONTACT {r.randint(200, 999)}-{r.randint(200, 999)}-{r.randint(1000, 9999)}"]
    else:
        com = "DESC " + r.choice(["EXCAVATOR", "CNC MACHINE", "TRANSFORMER", "WIND BLADE SECTION", "INDUSTRIAL BOILER",
                                  "STEEL BEAMS", "PRECAST CONCRETE"])
        sig = ["1 PC", f"WT {r.randint(42000, 140000)} LB", f"DIMS {r.randint(240, 600)}X{r.randint(96, 160)}X{r.randint(96, 170)} IN",
               "OVERSIZE LOAD", f"PERMIT {r.randint(100000, 999999)}", "ESCORT REQUIRED", "FLATBED TARP"]
    return base, sig, com


OCR_MAP = {"O": "0", "0": "O", "I": "1", "1": "I", "S": "5", "5": "S", "B": "8", "8": "B", "E": "F", "M": "N", "L": "I"}


def ocr_corrupt(s, p, r):
    out = []
    for ch in s:
        u = r.random()
        if ch in OCR_MAP and u < p:
            out.append(OCR_MAP[ch])
        elif ch.isalnum() and u < 1.5 * p:      # dropped character
            continue
        else:
            out.append(ch)
    return "".join(out)


def space_corrupt(s, p, r):
    out = []
    for ch in s:
        if ch == " " and r.random() < p:
            out.append(r.choice(["  ", "   ", ""]))
        elif ch.isalpha() and r.random() < p / 4:
            out.append(ch + " ")
        else:
            out.append(ch)
    return "".join(out)


def make_doc(c, r):
    base, sig, com = make_fields(c, r)
    sig = [f for f in sig if r.random() > P_DROP_FIELD]
    if com is not None and r.random() > P_DROP_COMMODITY:
        sig.insert(0, com)
    extra = [r.choice(GENERIC_INSTR)] if r.random() < 0.5 else []
    if r.random() < P_CONFLICT:
        other = r.choice([k for k in range(len(CLASSES)) if k != c])
        extra.append(r.choice(CLASS_INSTR[other]))
    fields = base + sig + extra
    # keep header fields first, shuffle the rest lightly so order is not a perfect cue
    head, tail = fields[:6], fields[6:]
    r.shuffle(tail)
    fields = head + tail
    fields = [f[: max(3, int(len(f) * r.uniform(0.3, 0.9)))] if r.random() < P_TRUNCATE else f for f in fields]
    text = r.choice([" ", " | ", " ; ", "\n"]).join(fields)
    text = ocr_corrupt(text, r.choice(OCR_LEVELS), r)
    return space_corrupt(text, r.choice(SPACE_LEVELS), r)


def build_dataset(seed=DATA_SEED):
    r = random.Random(seed)
    counts = [round(p * N_TOTAL) for p in PROPS]
    splits = {k: ([], []) for k in SPLIT}
    for c, n in enumerate(counts):
        n_tr, n_va = round(n * SPLIT["train"]), round(n * SPLIT["val"])
        bounds = {"train": (0, n_tr), "val": (n_tr, n_tr + n_va), "test": (n_tr + n_va, n)}
        for k, (a, b) in bounds.items():
            for _ in range(b - a):
                splits[k][0].append(make_doc(c, r))
                splits[k][1].append(c)
    data = {}
    for k, (texts, labels) in splits.items():
        idx = list(range(len(texts)))
        r.shuffle(idx)
        data[k] = ([texts[i] for i in idx], np.array([labels[i] for i in idx]))
    # label noise on TRAIN labels only
    tr_texts, y_clean = data["train"]
    y_noisy = y_clean.copy()
    nr = np.random.RandomState(seed)
    flip = nr.rand(len(y_noisy)) < TRAIN_LABEL_NOISE
    for i in np.where(flip)[0]:
        y_noisy[i] = nr.choice([k for k in range(len(CLASSES)) if k != y_clean[i]])
    data["train"] = (tr_texts, y_noisy)
    return data, y_clean, flip


data, y_train_clean, flipped = build_dataset()

# Fingerprint of the exact dataset (texts + labels, all splits). If this differs from the reference
# below, the dataset on this machine is NOT the one used in the earlier CPU run.
import hashlib
_h = hashlib.sha256()
for _k in ("train", "val", "test"):
    _h.update(json.dumps([data[_k][0], data[_k][1].tolist()]).encode())
DATA_SHA256 = _h.hexdigest()
REFERENCE_DATA_SHA256 = "e74c4242ce78d13fec8ad56ebe0647c6598766c815068a00ccb2369c6307d610"  # from the earlier CPU run
REFERENCE_VOCAB = 5565
print("dataset sha256:", DATA_SHA256)
if DATA_SHA256 != REFERENCE_DATA_SHA256:
    raise RuntimeError("Dataset differs from the reference (data seed 42) dataset. Stopping so a different "
                       "experiment is not run by accident.")
print("dataset matches the reference dataset: OK")
print("Example documents:")
for c in range(len(CLASSES)):
    i = int(np.where(data["test"][1] == c)[0][0])
    print(f"[{CLASSES[c]}]", data["test"][0][i].replace("\n", " / ")[:230])
print()
dist = {k: np.bincount(v[1], minlength=6).tolist() for k, v in data.items()}
print("class counts per split", dist)
print(f"train label noise: {int(flipped.sum())} of {len(flipped)} labels flipped ({flipped.mean():.2%}); val/test labels are clean")

# %% [markdown]
# ## 2. Tokenizer
# Lowercase, split into alphanumeric runs and single punctuation marks. Vocabulary is built from the
# training texts only (min frequency 2). Fixed length 128, pad id 0, unknown id 1.

# %%
TOKEN_RE = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")
PAD, UNK = 0, 1


def tok(s):
    return TOKEN_RE.findall(s.lower())


counter = Counter(t for s in data["train"][0] for t in tok(s))
itos = ["<pad>", "<unk>"] + [t for t, n in counter.most_common() if n >= 2]
stoi = {t: i for i, t in enumerate(itos)}
VOCAB = len(itos)


def encode(texts):
    ids = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
    lens, trunc = [], 0
    for i, s in enumerate(texts):
        t = [stoi.get(w, UNK) for w in tok(s)]
        trunc += len(t) > MAX_LEN
        t = t[:MAX_LEN]
        ids[i, : len(t)] = t
        lens.append(len(t))
    return torch.from_numpy(ids), np.array(lens), trunc


X, Y, TOKSTATS = {}, {}, {}
for k, (texts, labels) in data.items():
    ids, lens, trunc = encode(texts)
    X[k], Y[k] = ids, torch.from_numpy(labels)
    TOKSTATS[k] = {"n_docs": len(texts), "mean_len": float(lens.mean()), "max_len_before_trunc_cap": int(lens.max()),
                   "pct_truncated": 100 * trunc / len(texts), "nonpad_tokens": int(lens.sum())}
unk_rate = {k: float((X[k] == UNK).sum() / (X[k] != PAD).sum()) for k in X}
print("vocab size:", VOCAB)
assert VOCAB == REFERENCE_VOCAB, f"vocab size {VOCAB} != reference {REFERENCE_VOCAB}"
print("token stats:", json.dumps(TOKSTATS, indent=1))
print("unk rate:", {k: round(v, 4) for k, v in unk_rate.items()})

# %% [markdown]
# ## 3. Baseline model
# Embedding + learned positions, 4 pre-LN Transformer blocks (d=128, 4 heads, FFN 256),
# padding-aware softmax attention, mean pooling over non-padding tokens, linear head.
# The attention is its own module so a later variant can swap it out.

# %%
D_MODEL, N_LAYERS, N_HEADS, D_FF = 128, 4, 4, 256


class SoftmaxSelfAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, pad_mask):          # pad_mask: (B, T) True where real token
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        key_mask = pad_mask[:, None, None, :]                       # padded keys are never attended to
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)
        return self.out(o.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.attn = attn
        self.ff = nn.Sequential(nn.Linear(D_MODEL, D_FF), nn.GELU(), nn.Linear(D_FF, D_MODEL))

    def forward(self, x, pad_mask):
        x = x + self.attn(self.ln1(x), pad_mask)
        return x + self.ff(self.ln2(x))


class Classifier(nn.Module):
    def __init__(self, vocab, n_classes, attn_factory):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, D_MODEL, padding_idx=PAD)
        self.pos_emb = nn.Embedding(MAX_LEN, D_MODEL)
        self.blocks = nn.ModuleList([Block(attn_factory()) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, n_classes)

    def forward(self, ids):
        pad_mask = ids != PAD
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(ids.shape[1], device=ids.device))
        for b in self.blocks:
            x = b(x, pad_mask)
        x = self.ln_f(x)
        m = pad_mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)               # mean over real tokens only
        return self.head(pooled)


def build_baseline():
    return Classifier(VOCAB, len(CLASSES), lambda: SoftmaxSelfAttention(D_MODEL, N_HEADS))


def param_counts(model):
    total = sum(p.numel() for p in model.parameters())
    emb = model.tok_emb.weight.numel() + model.pos_emb.weight.numel()
    return {"total": total, "embedding": emb, "non_embedding": total - emb}


# Sanity check: padding must not change the output. Logits with the full 128 positions should match
# logits when the trailing all-padding columns are cut off (so padding is masked in attention and pooling).
_m = build_baseline().eval()
_ids = X["val"][:8]
_L = int((_ids != PAD).sum(1).max())
with torch.no_grad():
    _diff = (_m(_ids) - _m(_ids[:, :_L])).abs().max().item()
print(f"padding invariance: max |logit diff| = {_diff:.2e} (cut to {_L} of {MAX_LEN} positions)")
assert _diff < 1e-4, "padding is leaking into the output"
print("param counts:", param_counts(_m))

# %% [markdown]
# ## 4. Training
# AdamW, lr 1e-3, weight decay 0.01, 100 warmup steps then linear decay to 0, batch 64, 8 epochs,
# plain cross-entropy (no class weights, no resampling), fp32. The model after the last epoch is the model
# that gets evaluated. Validation is only monitored. Test is evaluated once, after training.

# %%
EPOCHS, BATCH, LR, WD, WARMUP = 8, 64, 1e-3, 0.01, 100
SEEDS = [0, 1, 2]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, ids, bs=256):
    model.eval()
    return torch.cat([model(ids[i:i + bs].to(DEVICE)).argmax(-1).cpu() for i in range(0, len(ids), bs)]).numpy()


def metrics(y_true, y_pred):
    per_class = {}
    for c, name in enumerate(CLASSES):
        m = y_true == c
        per_class[name] = float((y_pred[m] == c).mean())      # per-class accuracy = recall of that class
    rare = [per_class[CLASSES[i]] for i in RARE_IDS]
    return {"overall_acc": float((y_true == y_pred).mean()), "per_class_acc": per_class,
            "rare_class_recall_macro": float(np.mean(rare)),
            "rare_support": {CLASSES[i]: int((y_true == i).sum()) for i in RARE_IDS}}


def train_baseline(seed):
    set_seed(seed)
    model = build_baseline().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    n = len(X["train"])
    steps_per_epoch = math.ceil(n / BATCH)
    total = EPOCHS * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / WARMUP if s < WARMUP else max(0.0, (total - s) / (total - WARMUP)))
    g = torch.Generator().manual_seed(seed)
    xtr, ytr = X["train"], Y["train"]
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    log, t0 = [], time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=g)
        run, cnt = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = xtr[idx].to(DEVICE), ytr[idx].to(DEVICE)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            run += loss.item() * len(idx); cnt += len(idx)
        val_acc = float((predict(model, X["val"]) == Y["val"].numpy()).mean())
        log.append({"epoch": ep + 1, "train_loss": run / cnt, "val_acc": val_acc})
        print(f"  seed {seed} epoch {ep + 1}/{EPOCHS} train_loss {run / cnt:.4f} val_acc {val_acc:.4f}")
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return model, log, time.time() - t0


# %%
results = {"env": ENV, "config": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "wd": WD, "warmup": WARMUP,
                                  "max_len": MAX_LEN, "d_model": D_MODEL, "layers": N_LAYERS, "heads": N_HEADS,
                                  "d_ff": D_FF, "vocab": VOCAB, "train_label_noise": TRAIN_LABEL_NOISE,
                                  "data_seed": DATA_SEED, "data_sha256": DATA_SHA256},
           "class_counts": dist, "token_stats": TOKSTATS, "baseline": {}}
y_test = Y["test"].numpy()
for seed in SEEDS:
    model, log, secs = train_baseline(seed)
    pred = predict(model, X["test"])
    r = metrics(y_test, pred)
    r.update(train_seconds=secs, epoch_log=log, params=param_counts(model),
             val_acc_final=log[-1]["val_acc"])
    results["baseline"][f"seed{seed}"] = r
    print(f"seed {seed}: test acc {r['overall_acc']:.4f} | rare macro recall {r['rare_class_recall_macro']:.4f} | {secs:.0f}s")
    torch.save(model.state_dict(), f"{OUT_DIR}/baseline_seed{seed}.pt")

# %% [markdown]
# ## 5. Baseline report (seeds 0, 1, 2)

# %%
import pandas as pd

HAZ, OVR = "hazmat", "oversize_heavy"
runs = results["baseline"]
print("GPU:", ENV["gpu_name"], "| CUDA", ENV["cuda_version"], "| cc", ENV["compute_capability"])

per_seed = pd.DataFrame({
    sd: {"train_seconds": r["train_seconds"], "final_val_acc": r["val_acc_final"],
         "overall_test_acc": r["overall_acc"],
         **{f"acc_{c}": r["per_class_acc"][c] for c in CLASSES},
         "rare_macro_recall": r["rare_class_recall_macro"]}
    for sd, r in runs.items()})
print(per_seed.round(4).to_string())

p = runs["seed0"]["params"]
print(f"\nparams: total {p['total']:,} | embedding {p['embedding']:,} | non-embedding {p['non_embedding']:,}")

agg = pd.DataFrame({"overall_test_acc": per_seed.loc["overall_test_acc"],
                    "hazmat_recall": per_seed.loc[f"acc_{HAZ}"],
                    "oversize_recall": per_seed.loc[f"acc_{OVR}"],
                    "rare_macro_recall": per_seed.loc["rare_macro_recall"]}).T
agg["mean"], agg["std"] = agg.mean(axis=1), agg.std(axis=1, ddof=1)
print("\nmean and sample std (ddof=1) over 3 seeds")
print(agg[["mean", "std"]].round(4).to_string())

results["summary"] = {"mean": agg["mean"].to_dict(), "std": agg["std"].to_dict()}
with open(f"{OUT_DIR}/baseline_results_cuda.json", "w") as f:
    json.dump(results, f, indent=1)
