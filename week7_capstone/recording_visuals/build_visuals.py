"""Builds the recording visuals from the frozen repository files. Nothing here changes an experimental file.

    python recording_visuals/build_visuals.py            # SVG + HTML dashboard
    python recording_visuals/build_visuals.py --png      # also render V1 and V2 to 1920x1080 PNG (needs Chromium)

Reads:  configs/base.yaml, configs/arm_{A,B,C}.yaml, results/main_summary.json, results/runs/main/*.json,
        figures/fig1..fig3 PNG (copied into the dashboard as embedded images, not modified).
Writes: recording_visuals/{v1_changed_vs_fixed,v2_frozen_rule_applied}.svg, recording_dashboard.html,
        and with --png the two PNG files. Every number shown is read from those files at build time.
"""
import argparse
import base64
import glob
import json
import os
import subprocess
import sys
from html import escape

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "recording_visuals")

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
PANEL, BAND = "#f4f3f0", "#e9e8e3"
ARM = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}          # same fixed order and values as the figures
FAIL_FG, FAIL_BG = "#9b1c13", "#fbe9e6"
FONT = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'SF Mono', Consolas, 'DejaVu Sans Mono', monospace"
W, H = 1920, 1080


# ---------------------------------------------------------------- data
def load():
    base = yaml.safe_load(open(os.path.join(ROOT, "configs/base.yaml")))
    arms = {a: yaml.safe_load(open(os.path.join(ROOT, f"configs/arm_{a}.yaml")))["model"]["pos_encoding"] for a in "ABC"}
    summ = json.load(open(os.path.join(ROOT, "results/main_summary.json")))
    runs = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results/runs/main/*.json"))):
        r = json.load(open(f))
        if r.get("stage") == "main" and r.get("status") == "completed":
            runs.append(r)
    return base, arms, summ, runs


def sgn(x, nd=4):
    return f"{x:+.{nd}f}".replace("-", "−")


def num(x, nd=4):
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------- svg helpers
def t(x, y, s, size=28, weight=400, fill=INK, anchor="start", family=FONT, extra=""):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" {extra}>{escape(str(s))}</text>')


def rect(x, y, w, h, fill, stroke="none", sw=0, rx=0):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def svg_doc(body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
            f'aria-label="{escape(title)}"><rect width="{W}" height="{H}" fill="{SURFACE}"/>{body}</svg>')


# ---------------------------------------------------------------- V1
def build_v1(base, arms, runs):
    m, tr, d, dt = base["model"], base["train"], base["determinism"], base["data"]
    assert tr["early_stopping"] is False and tr["steps"] == 8000 and d["policy"] == "P1"
    data_hash = {r["data_sha256"] for r in runs}
    assert len(data_hash) == 1
    dh = data_hash.pop()[:8]
    seeds = base["study"]["main_seeds"]
    init_ok = all(len({r["shared_init_weights_sha256"] for r in runs if r["seed"] == s}) == 1 for s in seeds)
    assert init_ok and {r["policy"] for r in runs} == {"P1"} and {r["steps"] for r in runs} == {8000}
    o = [t(80, 108, "ONE FACTOR CHANGED", 84, 800)]
    # changed band
    o.append(rect(80, 140, 1760, 290, PANEL, GRID, 2, 18))
    o.append(t(120, 200, "model.pos_encoding", 50, 600, INK, family=MONO))
    names = {"A": arms["A"], "B": arms["B"], "C": arms["C"]}
    # three cards inside the panel (inner width 1680 from x=120)
    cw, gap = 520, 60
    for i, a in enumerate("ABC"):
        x = 120 + i * (cw + gap)
        o.append(rect(x, 232, cw, 168, SURFACE, GRID, 2, 14))
        o.append(rect(x, 232, 14, 168, ARM[a], "none", 0, 0))
        o.append(t(x + 44, 304, f"Arm {a}", 30, 600, INK2))
        o.append(t(x + 44, 372, f"= {names[a]}", 62, 800, INK, family=MONO, extra='xml:space="preserve"'))
        o.append(f'<circle cx="{x + cw - 50}" cy="268" r="14" fill="{ARM[a]}" stroke="{SURFACE}" stroke-width="3"/>')
    o.append(t(80, 508, "EVERYTHING ELSE HELD FIXED", 60, 800))
    cells = [
        ("MODEL ARCHITECTURE", f"{m['d_model']}-d · {m['n_layers']} layers · {m['n_heads']} heads",
         f"d_ff {m['d_ff']} · dropout {m['dropout']} · {m['precision']}"),
        ("OPTIMIZER AND LEARNING RATE", f"{tr['optimizer'].capitalize().replace('Adamw', 'AdamW')} · lr {tr['lr']}", "no per-arm tuning"),
        ("WEIGHT DECAY", f"{tr['weight_decay']}", "same in every arm"),
        ("WARMUP AND SCHEDULE", f"{tr['warmup_steps']} warmup steps", "linear warmup, then linear decay"),
        ("BATCH SIZE", f"{tr['batch_size']}", f"no gradient accumulation"),
        ("TRAINING LENGTH", f"{tr['steps']} steps", "fixed step budget"),
        ("EARLY STOPPING", "None", "final-step weights only"),
        ("DATA SEED", f"{dt['data_seed']}", "model seeds never touch the data"),
        ("SAME DATASET", f"{dt['n_series']} series · {dt['years']} years", f"hash {dh}… in all {len(runs)} runs"),
        ("MATCHED SEEDS", " · ".join(str(s) for s in seeds), "same three seeds for every arm"),
        ("DETERMINISM POLICY", d["policy"], "deterministic kernels, every arm"),
        ("SHARED-BACKBONE INIT", "Identical across arms", f"hash-checked for seeds {', '.join(str(s) for s in seeds)}"),
    ]
    cw2, ch, gx, gy = 422, 150, 24, 22
    for i, (lab, val, sub) in enumerate(cells):
        r_, c_ = divmod(i, 4)
        x = 80 + c_ * (cw2 + gx)
        y = 548 + r_ * (ch + gy)
        o.append(rect(x, y, cw2, ch, PANEL, GRID, 2, 12))
        o.append(t(x + 22, y + 36, lab, 20, 700, INK2, extra='letter-spacing="1.2"'))
        o.append(t(x + 22, y + 92, val, 32, 800, INK))
        o.append(t(x + 22, y + 130, sub, 22, 400, INK2))
    return svg_doc("".join(o), "One factor changed, everything else held fixed")


# ---------------------------------------------------------------- V2
def chip(x, y, w, h, line1, line2, fill=PANEL, fg=INK, fg2=INK2, stroke=GRID, s1=34, s2=22):
    return (rect(x, y, w, h, fill, stroke, 2, 14) + t(x + w / 2, y + h / 2 - 2, line1, s1, 800, fg, "middle")
            + t(x + w / 2, y + h / 2 + 30, line2, s2, 500, fg2, "middle"))


def build_panel(px, py, pw, ph, key, c, delta, k, emph):
    arm, ref = key.split("_vs_")
    o = []
    o.append(rect(px, py, pw, ph, SURFACE, INK if emph else GRID, 5 if emph else 2, 20))
    o.append(f'<circle cx="{px + 44}" cy="{py + 44}" r="15" fill="{ARM[arm]}" stroke="{SURFACE}" stroke-width="3"/>')
    o.append(t(px + 76, py + 58, f"{arm} − {ref}", 46, 800))
    nm = {"B": "sinusoidal", "C": "RoPE", "A": "learned"}
    o.append(t(px + pw - 30, py + 56, f"{nm[arm]} − {nm[ref]}", 28, 500, INK2, "end"))
    # --- number line
    lo, hi = -0.15, 0.15
    ix0, ix1 = px + 50, px + pw - 50
    def X(v):
        return ix0 + (v - lo) / (hi - lo) * (ix1 - ix0)
    cy = py + 205
    o.append(rect(X(-delta), cy - 85, X(delta) - X(-delta), 170, BAND))
    for v in (-delta, delta):
        o.append(f'<line x1="{X(v)}" y1="{cy - 85}" x2="{X(v)}" y2="{cy + 85}" stroke="{INK2}" stroke-width="3" stroke-dasharray="2 8" stroke-linecap="round"/>')
    o.append(f'<line x1="{ix0}" y1="{cy}" x2="{ix1}" y2="{cy}" stroke="{MUTED}" stroke-width="2"/>')
    o.append(f'<line x1="{X(0)}" y1="{cy - 85}" x2="{X(0)}" y2="{cy + 85}" stroke="{INK2}" stroke-width="2"/>')
    o.append(t(X(-delta), cy + 122, f"−δ = −{delta:.2f}", 22, 600, INK2, "middle"))
    o.append(t(X(0), cy + 122, "0", 22, 600, INK2, "middle"))
    o.append(t(X(delta), cy + 122, f"+δ = +{delta:.2f}", 22, 600, INK2, "middle"))
    ds = [(int(s), v) for s, v in c["d_by_seed"].items()]
    order = sorted(range(3), key=lambda i: ds[i][1])
    for rank, i in enumerate(order):
        s, v = ds[i]
        o.append(f'<circle cx="{X(v)}" cy="{cy}" r="13" fill="{ARM[arm]}" stroke="{SURFACE}" stroke-width="4"/>')
        off = 34 + 58 * (rank % 2)
        halo = f'stroke="{SURFACE}" stroke-width="7" paint-order="stroke" stroke-linejoin="round"'
        anc, dx = ("start", 6) if (rank == 2 and v > 0.05) else ("middle", 0)   # keep the right-most label clear of the mean tick
        o.append(t(X(v) + dx, cy - off - 26, f"seed {s}", 21, 600, INK2, anc, extra=halo))
        o.append(t(X(v) + dx, cy - off, sgn(v), 25, 800, INK, anc, extra=halo))
    mean = c["mean"]
    o.append(f'<line x1="{X(mean)}" y1="{cy - 50}" x2="{X(mean)}" y2="{cy + 50}" stroke="{INK}" stroke-width="7" stroke-linecap="round"/>')
    o.append(t(X(mean) - 16, cy + 58, f"mean {sgn(mean)}", 25, 800, INK, "end"))
    if emph and abs(mean) < delta:
        gap_ = delta - abs(mean)
        a, b_ = X(mean), X(delta)
        yb = cy + 52
        o.append(f'<path d="M{a} {yb - 8} V{yb + 8} M{a} {yb} H{b_} M{b_} {yb - 8} V{yb + 8}" stroke="{FAIL_FG}" stroke-width="4" fill="none" stroke-linecap="round"/>')
        o.append(t(a - 16, cy + 82, f"{gap_:.4f} short of δ", 22, 700, FAIL_FG, "end"))
    # --- stat chips
    sy = py + 345
    cw = (pw - 2 * 28 - 2 * 16) / 3
    stats = [("mean d̄", sgn(mean)), ("δ", f"{delta:.2f}"), ("k × SD_d", num(c["k_times_sd"]))]
    for i, (lab, val) in enumerate(stats):
        x = px + 28 + i * (cw + 16)
        o.append(rect(x, sy, cw, 86, PANEL, GRID, 2, 12))
        o.append(t(x + cw / 2, sy + 30, lab, 22, 600, INK2, "middle"))
        o.append(t(x + cw / 2, sy + 70, val, 36, 800, INK, "middle"))
    # --- checks
    cond = c["conditions"]
    loo = list(c["leave_one_out_means"].values())
    rows = [
        ("Same sign", "(a)", cond["a_same_sign"],
         ("3 of 3 higher error" if all(v > 0 for v in c["d_by_seed"].values()) else
          "signs are mixed")),
        ("Practical threshold", "(b)", cond["b_abs_mean_at_least_delta"],
         f"|mean| {num(abs(mean))} {'≥' if cond['b_abs_mean_at_least_delta'] else '<'} δ {delta:.2f}"),
        ("Noise", "(c)", cond["c_abs_mean_at_least_k_sd"],
         f"|mean| {num(abs(mean))} {'≥' if cond['c_abs_mean_at_least_k_sd'] else '<'} k×SD {num(c['k_times_sd'])}"),
        ("Leave-one-out", "(d)", cond["d_leave_one_out_consistent"],
         " · ".join(sgn(v) for v in loo)),
    ]
    ry = py + 462
    for i, (lab, key_, ok, detail) in enumerate(rows):
        y = ry + i * 60
        o.append(rect(px + 28, y, pw - 56, 52, PANEL if ok else FAIL_BG, GRID if ok else "none", 1.5, 10))
        o.append(t(px + 46, y + 35, f"{lab} {key_}", 25, 700, INK if ok else FAIL_FG))
        o.append(t(px + 330, y + 35, detail, 23, 500, INK if ok else FAIL_FG))
        o.append(t(px + pw - 46, y + 36, "✓ PASS" if ok else "✗ FAIL", 24, 800, INK if ok else FAIL_FG, "end"))
    # --- verdict
    vy = ry + 4 * 60 + 12
    o.append(rect(px + 28, vy, pw - 56, 68, INK, "none", 0, 12))
    o.append(t(px + pw / 2, vy + 44, c["verdict"], 31, 800, SURFACE, "middle"))
    return "".join(o)


def build_v2(base, summ):
    delta, k = base["analysis"]["delta_mape_points"], base["analysis"]["k"]
    assert summ["delta"] == delta and summ["k"] == k
    cmp_ = summ["comparisons"]
    o = [t(60, 78, "Frozen Rule Applied", 58, 800)]
    o.append(t(1860, 70, f"δ = {delta:.2f} MAPE points  ·  k = {k:g}  ·  exploratory screening criteria, not significance tests",
               24, 500, INK2, "end"))
    py, ph, pw = 108, 790, 900
    o.append(build_panel(60, py, pw, ph, "B_vs_A", cmp_["B_vs_A"], delta, k, False))
    o.append(build_panel(960, py, pw, ph, "C_vs_A", cmp_["C_vs_A"], delta, k, True))
    # story strip for C - A
    c = cmp_["C_vs_A"]
    mean = c["mean"]
    n_pos = sum(v > 0 for v in c["d_by_seed"].values())
    chips = [
        (f"{n_pos} of 3 seeds", "higher error than A", 310, PANEL, INK, INK2, GRID),
        (f"mean {sgn(mean, 3)}", "MAPE points", 270, PANEL, INK, INK2, GRID),
        (f"δ = {delta:.2f}", "practical threshold", 270, PANEL, INK, INK2, GRID),
        (f"{abs(mean):.3f} < {delta:.2f}", "threshold not met", 300, FAIL_BG, FAIL_FG, FAIL_FG, FAIL_BG),
        ("no difference detectable", "at this budget", 420, INK, SURFACE, SURFACE, INK),
    ]
    total = sum(ch[2] for ch in chips) + 4 * 40
    x = (W - total) / 2
    cy = 934
    o.append(t(x, cy - 12, "C − A", 24, 800, INK2))
    for i, (l1, l2, w, fill, fg, fg2, stroke) in enumerate(chips):
        o.append(chip(x, cy, w, 108, l1, l2, fill, fg, fg2, stroke, 33, 22))
        x += w
        if i < len(chips) - 1:
            o.append(f'<path d="M{x + 8} {cy + 56} H{x + 30} M{x + 22} {cy + 46} L{x + 32} {cy + 56} L{x + 22} {cy + 66}" stroke="{INK2}" stroke-width="4" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
        x += 40
    return svg_doc("".join(o), "Frozen decision rule applied to B minus A and C minus A")


# ---------------------------------------------------------------- dashboard
def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()


def build_html(v1, v2):
    figs = {n: b64(os.path.join(ROOT, "figures", f)) for n, f in
            (("fig1", "fig1_seed_level_mape.png"), ("fig2", "fig2_matched_differences.png"), ("fig3", "fig3_cost.png"))}
    sections = [
        ("Changed vs Fixed", f'<div class="svgwrap">{v1}</div>'),
        ("Frozen Rule Applied", f'<div class="svgwrap">{v2}</div>'),
        ("Seed-Level Validation MAPE", f'<div class="ttl">Seed-Level Validation MAPE</div><div class="figwrap"><img alt="Seed-level validation MAPE by arm" src="data:image/png;base64,{figs["fig1"]}"></div>'),
        ("Matched Differences", f'<div class="ttl">Matched Differences</div><div class="figwrap"><img alt="Matched differences against delta" src="data:image/png;base64,{figs["fig2"]}"></div>'),
        ("Cost Comparison", f'<div class="ttl">Cost Comparison</div><div class="figwrap"><img alt="Training time, latency and memory by arm" src="data:image/png;base64,{figs["fig3"]}"></div>'),
    ]
    tabs = "".join(f'<button class="tab" data-i="{i}">{i + 1}&nbsp;&nbsp;{escape(n)}</button>' for i, (n, _) in enumerate(sections))
    slides = "".join(f'<section class="slide" data-i="{i}">{body}</section>' for i, (_, body) in enumerate(sections))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Week 7 Recording Dashboard</title>
<style>
:root{{--surface:{SURFACE};--ink:{INK};--ink2:{INK2};--grid:{GRID};--panel:{PANEL}}}
*{{box-sizing:border-box}} html,body{{margin:0;height:100%;background:#1b1b1a;overflow:hidden;font-family:{FONT}}}
#nav{{position:fixed;top:0;left:0;right:0;height:48px;display:flex;gap:6px;align-items:center;padding:0 12px;background:#2a2a28;z-index:5}}
#nav.hide{{display:none}}
#nav button{{font:600 15px {FONT};color:#e9e8e3;background:#3a3a37;border:0;border-radius:8px;padding:9px 14px;cursor:pointer}}
#nav button:hover{{background:#4a4a46}} #nav button:focus-visible{{outline:2px solid #fff;outline-offset:2px}}
#nav .tab.on{{background:#fcfcfb;color:#0b0b0b}} #nav .sp{{flex:1}} #nav .hint{{color:#9d9c97;font:500 13px {FONT}}}
#stage{{position:absolute;left:0;top:0;width:{W}px;height:{H}px;transform-origin:0 0;background:var(--surface);overflow:hidden}}
.slide{{display:none;position:absolute;inset:0}} .slide.on{{display:block}}
.svgwrap,.svgwrap svg{{width:{W}px;height:{H}px;display:block}}
.ttl{{position:absolute;left:60px;top:34px;font:800 56px {FONT};color:var(--ink)}}
.figwrap{{position:absolute;left:0;right:0;top:130px;bottom:30px;display:flex;align-items:center;justify-content:center}}
.figwrap img{{max-width:1840px;max-height:920px;width:auto;height:auto}}
</style></head><body>
<div id="nav"><button id="prev" aria-label="Previous section">&#8249; Prev</button>{tabs}<button id="next" aria-label="Next section">Next &#8250;</button>
<span class="sp"></span><span class="hint">&larr; &rarr; or 1&ndash;5 &middot; H hides this bar</span></div>
<div id="stage">{slides}</div>
<script>
(function(){{
var slides=[].slice.call(document.querySelectorAll('.slide')),tabs=[].slice.call(document.querySelectorAll('.tab')),
nav=document.getElementById('nav'),stage=document.getElementById('stage'),cur=0;
function fit(){{var top=nav.classList.contains('hide')?0:48,w=innerWidth,h=innerHeight-top,s=Math.min(w/{W},h/{H});
stage.style.transform='scale('+s+')';stage.style.left=((w-{W}*s)/2)+'px';stage.style.top=(top+(h-{H}*s)/2)+'px';}}
function show(i){{cur=Math.max(0,Math.min(slides.length-1,i));slides.forEach(function(s,j){{s.classList.toggle('on',j===cur)}});
tabs.forEach(function(t,j){{t.classList.toggle('on',j===cur)}});try{{history.replaceState(null,'','#'+(cur+1))}}catch(e){{}}}}
tabs.forEach(function(t){{t.addEventListener('click',function(){{show(+t.dataset.i)}})}});
document.getElementById('prev').onclick=function(){{show(cur-1)}};document.getElementById('next').onclick=function(){{show(cur+1)}};
addEventListener('keydown',function(e){{if(e.key==='ArrowRight'||e.key==='PageDown')show(cur+1);else if(e.key==='ArrowLeft'||e.key==='PageUp')show(cur-1);
else if(e.key>='1'&&e.key<='5')show(+e.key-1);else if(e.key==='h'||e.key==='H'){{nav.classList.toggle('hide');fit()}}}});
addEventListener('resize',fit);var n=parseInt((location.hash||'#1').slice(1),10)||1;show(n-1);fit();
}})();
</script></body></html>'''


# ---------------------------------------------------------------- png
def chromium():
    for p in ("/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
              "/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell"):
        if os.path.exists(p):
            return p
    sys.exit("Chromium not found; open the SVG files in a browser and export manually.")


def render_png(svg_path, png_path):
    tmp = png_path + ".tmp.png"
    subprocess.run([chromium(), "--headless", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                    f"--window-size={W},{H + 200}", f"--screenshot={tmp}", "file://" + os.path.abspath(svg_path)],
                   check=True, capture_output=True, timeout=120)
    from PIL import Image
    Image.open(tmp).convert("RGB").crop((0, 0, W, H)).save(png_path)   # headless window chrome eats ~90px; crop to exact size
    os.remove(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    base, arms, summ, runs = load()
    assert len(runs) == 9 and summ["complete"] is True and arms == {"A": "learned", "B": "sinusoidal", "C": "rope"}
    v1, v2 = build_v1(base, arms, runs), build_v2(base, summ)
    files = {"v1_changed_vs_fixed": v1, "v2_frozen_rule_applied": v2}
    for name, svg in files.items():
        open(os.path.join(OUT, name + ".svg"), "w").write(svg)
    open(os.path.join(OUT, "recording_dashboard.html"), "w").write(build_html(v1, v2))
    if args.png:
        for name in files:
            render_png(os.path.join(OUT, name + ".svg"), os.path.join(OUT, name + ".png"))
    print("built:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
