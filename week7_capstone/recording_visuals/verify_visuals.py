"""Independent check of the recording visuals against the repository.

    python recording_visuals/verify_visuals.py

Recomputes the V2 numbers from the nine raw run files (not from main_summary.json), then checks that
  - every number and label printed in the two SVGs matches the source,
  - the figures embedded in the dashboard are byte-identical to figures/*.png,
  - the frozen files still verify.
Exits non-zero on any mismatch. Writes verification_report.txt next to this file.
"""
import base64
import html as htmllib
import glob
import hashlib
import json
import os
import re
import statistics as st
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "recording_visuals")
lines, bad = [], []


def check(ok, msg):
    lines.append(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        bad.append(msg)


def sgn(x, nd=4):
    return f"{x:+.{nd}f}".replace("-", "\u2212")


svg1 = open(os.path.join(OUT, "v1_changed_vs_fixed.svg")).read()
svg2 = open(os.path.join(OUT, "v2_frozen_rule_applied.svg")).read()
text1 = htmllib.unescape(" ".join(re.findall(r">([^<]+)<", svg1)))
text2 = htmllib.unescape(" ".join(re.findall(r">([^<]+)<", svg2)))

base = yaml.safe_load(open(os.path.join(ROOT, "configs/base.yaml")))
runs = {}
for f in glob.glob(os.path.join(ROOT, "results/runs/main/*.json")):
    r = json.load(open(f))
    if r.get("stage") == "main" and r.get("status") == "completed":
        runs[(r["arm"], r["seed"])] = r
check(len(runs) == 9, f"nine completed main runs found ({len(runs)})")

# ---- V1 ---------------------------------------------------------------
arms = {a: yaml.safe_load(open(os.path.join(ROOT, f"configs/arm_{a}.yaml"))) for a in "ABC"}
for a, want in (("A", "learned"), ("B", "sinusoidal"), ("C", "rope")):
    check(arms[a] == {"model": {"pos_encoding": want}}, f"arm_{a}.yaml overrides only model.pos_encoding = {want}")
    check(f"= {want}" in text1, f"V1 shows 'Arm {a} = {want}'")
check("model.pos_encoding" in text1, "V1 names the changed variable model.pos_encoding")
m, tr = base["model"], base["train"]
expect1 = {
    "model": f"{m['d_model']}-d \u00b7 {m['n_layers']} layers \u00b7 {m['n_heads']} heads",
    "model2": f"d_ff {m['d_ff']} \u00b7 dropout {m['dropout']} \u00b7 {m['precision']}",
    "optimizer": f"AdamW \u00b7 lr {tr['lr']}",
    "weight decay": f"{tr['weight_decay']}",
    "warmup": f"{tr['warmup_steps']} warmup steps",
    "schedule": "linear warmup, then linear decay" if tr["schedule"] == "linear_warmup_then_linear_decay" else "??",
    "batch": f"{tr['batch_size']}",
    "steps": f"{tr['steps']} steps",
    "early stopping": "None" if tr["early_stopping"] is False else "??",
    "data seed": f"{base['data']['data_seed']}",
    "seeds": " \u00b7 ".join(str(s) for s in base["study"]["main_seeds"]),
    "policy": base["determinism"]["policy"],
    "series/years": f"{base['data']['n_series']} series \u00b7 {base['data']['years']} years",
}
for k_, v in expect1.items():
    check(v in text1, f"V1 {k_}: '{v}' matches configs/base.yaml")
check(tr["grad_accumulation"] == 1, "no gradient accumulation: base.yaml grad_accumulation == 1")
check(base["eval"]["checkpoint_selection"] == "none", "final-step weights only: eval.checkpoint_selection == none")
dh = {r["data_sha256"] for r in runs.values()}
check(len(dh) == 1 and f"hash {next(iter(dh))[:8]}" in text1 and f"in all {len(runs)} runs" in text1,
      f"same dataset: one data_sha256 ({next(iter(dh))[:8]}...) across all {len(runs)} runs")
check({r["policy"] for r in runs.values()} == {"P1"}, "all nine run files record policy P1")
check({r["steps"] for r in runs.values()} == {8000}, "all nine run files record 8000 steps")
for s in (0, 1, 2):
    h = {runs[(a, s)]["shared_init_weights_sha256"] for a in "ABC"}
    check(len(h) == 1, f"seed {s}: shared_init_weights_sha256 identical across arms A, B, C")
check("Identical across arms" in text1, "V1 states identical shared-backbone initialisation")

# ---- V2: recompute from raw runs --------------------------------------
delta, k = base["analysis"]["delta_mape_points"], base["analysis"]["k"]
check((delta, k) == (0.10, 2.0), f"delta={delta}, k={k} from configs/base.yaml")
pre = open(os.path.join(ROOT, "PREREGISTRATION.md")).read()
check("(a) all `d_s` share a sign; (b) \\|d\u0304\\| \u2265 \u03b4; (c) \\|d\u0304\\| \u2265 k \u00d7 SD_d; (d)" in pre,
      "PREREGISTRATION.md section 5 defines conditions (a)-(d) as used")
check("No difference detectable at this budget" in pre, "PREREGISTRATION.md verdict wording present")


def sign(x):
    return (x > 0) - (x < 0)


for arm in ("B", "C"):
    d = [runs[(arm, s)]["val_mape"] - runs[("A", s)]["val_mape"] for s in (0, 1, 2)]
    mean, sd = st.mean(d), st.stdev(d)
    loo = [st.mean([d[j] for j in range(3) if j != i]) for i in range(3)]
    ksd = k * sd
    a = len({sign(v) for v in d}) == 1 and 0 not in {sign(v) for v in d}
    b = abs(mean) >= delta
    c = abs(mean) >= ksd
    dd = all(sign(v) == sign(mean) and abs(v) >= delta for v in loo)
    verdict = "supported, tentative" if (a and b and c and dd) else ("direction only" if (a and b) else "no difference detectable at this budget")
    summ = json.load(open(os.path.join(ROOT, "results/main_summary.json")))["comparisons"][f"{arm}_vs_A"]
    check(abs(summ["mean"] - mean) < 1e-12 and abs(summ["k_times_sd"] - ksd) < 1e-12,
          f"{arm}-A: recomputed mean/k*SD match main_summary.json ({mean:+.5f}, {ksd:.5f})")
    check((a, b, c, dd) == tuple(summ["conditions"].values()), f"{arm}-A: conditions (a,b,c,d) = {(a, b, c, dd)} match main_summary.json")
    check(verdict == summ["verdict"], f"{arm}-A: verdict '{verdict}' matches main_summary.json")
    for v in d:
        check(sgn(v) in text2, f"{arm}-A seed difference {sgn(v)} printed in V2")
    for v in loo:
        check(sgn(v) in text2, f"{arm}-A leave-one-out {sgn(v)} printed in V2")
    check(sgn(mean) in text2 and f"{ksd:.4f}" in text2, f"{arm}-A mean {sgn(mean)} and k*SD {ksd:.4f} printed in V2")
    if arm == "C":
        check(all(v > 0 for v in d), "C-A: all three differences positive (higher error than A)")
        check(f"{abs(mean):.3f} < {delta:.2f}" in text2 and f"{delta - abs(mean):.4f} short" in text2,
              f"C-A story: {abs(mean):.3f} < {delta:.2f}, gap {delta - abs(mean):.4f} printed")
check(text2.count("no difference detectable at this budget") == 3, "V2 shows the repository verdict wording (2 verdict bars + story chip)")
check(text2.count("\u2713 PASS") == 2 and text2.count("\u2717 FAIL") == 6, "V2 pass/fail marks: 2 PASS (C-A a, c), 6 FAIL")

# ---- dashboard figures byte-identical ---------------------------------
dash = open(os.path.join(OUT, "recording_dashboard.html")).read()
for name in ("fig1_seed_level_mape", "fig2_matched_differences", "fig3_cost"):
    raw = open(os.path.join(ROOT, "figures", name + ".png"), "rb").read()
    check(base64.b64encode(raw).decode() in dash, f"dashboard embeds figures/{name}.png unchanged")
check(dash.count('class="slide') >= 5, "dashboard has five sections")

# ---- png size, freeze, tracked files untouched ------------------------
try:
    from PIL import Image
    for n in ("v1_changed_vs_fixed", "v2_frozen_rule_applied"):
        p = os.path.join(OUT, n + ".png")
        check(os.path.exists(p) and Image.open(p).size == (1920, 1080), f"{n}.png is 1920x1080")
except ImportError:
    lines.append("SKIP  Pillow missing: PNG size not checked")
fr = subprocess.run([sys.executable, os.path.join(ROOT, "src/freeze.py"), "--verify"], capture_output=True, text=True, cwd=ROOT)
check(fr.returncode == 0 and "13 files match" in fr.stdout, "freeze.py --verify: " + fr.stdout.strip().splitlines()[-1])
gs = subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True, text=True, cwd=ROOT).stdout
changed_tracked = [l for l in gs.splitlines() if not l.startswith("??")]
check(not changed_tracked, f"no tracked file modified (git status shows {len(changed_tracked)} modified)")

open(os.path.join(OUT, "verification_report.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n{len(lines) - len(bad)} passed, {len(bad)} failed")
sys.exit(1 if bad else 0)
