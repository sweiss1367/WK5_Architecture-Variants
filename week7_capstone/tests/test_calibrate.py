import json
import os

import calibrate as C


def write(d, arm, steps, train_loss, val_mape, status="completed", stage="calibration", seed=900):
    os.makedirs(d, exist_ok=True)
    json.dump({"arm": arm, "seed": seed, "steps": steps, "stage": stage, "status": status, "attempt": 1,
               "final_train_loss_100": train_loss, "val_mape": val_mape, "val_mae_kw": val_mape, "test_mape": val_mape,
               "baselines": {"x": val_mape}}, open(os.path.join(d, f"{arm}_{steps}_{stage}_{seed}_{status}.json"), "w"))


def build(d, losses, val=5.0):
    for (arm, steps), l in losses.items():
        write(d, arm, steps, l, val)


A = {("A", 1000): 0.200, ("A", 2000): 0.150, ("A", 4000): 0.148}
OK_ALL = {**A, ("B", 2000): 0.151, ("B", 4000): 0.149, ("C", 2000): 0.152, ("C", 4000): 0.150}


def test_stage1_picks_smallest_plateaued_step(tmp_path):
    build(str(tmp_path), A)
    s, d, st = C.stage1(C.load_training_losses(str(tmp_path)))
    assert (s, st) == (2000, "ok") and d[0]["qualifies"] is False and d[1]["qualifies"] is True


def test_stage1_reports_no_qualifying_value_when_still_improving(tmp_path):
    build(str(tmp_path), {("A", 1000): 0.30, ("A", 2000): 0.20, ("A", 4000): 0.15})
    assert C.stage1(C.load_training_losses(str(tmp_path)))[0::2] == (None, "no_grid_value_qualifies")


def test_stage2_passes_and_fails_per_arm(tmp_path):
    build(str(tmp_path), OK_ALL)
    per, st = C.stage2(C.load_training_losses(str(tmp_path)), 2000)
    assert st == "ok" and all(v["passes"] for v in per.values())
    build(str(tmp_path), {("C", 2000): 0.20, ("C", 4000): 0.15})            # arm C still improving
    per, st = C.stage2(C.load_training_losses(str(tmp_path)), 2000)
    assert st == "fails_on_some_arm" and per["C"]["passes"] is False and per["B"]["passes"] is True


def test_stage2_flags_missing_runs(tmp_path):
    build(str(tmp_path), A)
    assert C.stage2(C.load_training_losses(str(tmp_path)), 2000)[1] == "missing_runs"


def test_validation_values_cannot_influence_the_decision(tmp_path):
    """Same training losses, wildly different validation numbers (including NaN and negative): identical decision."""
    a, b = tmp_path / "a", tmp_path / "b"
    build(str(a), OK_ALL, val=5.0)
    build(str(b), OK_ALL, val=float("nan"))
    for f in os.listdir(b):                                                 # make validation adversarial per arm
        r = json.load(open(b / f)); r["val_mape"] = -1e9 if r["arm"] == "A" else 1e9
        json.dump(r, open(b / f, "w"))
    la, lb = C.load_training_losses(str(a)), C.load_training_losses(str(b))
    assert la == lb
    assert C.stage1(la) == C.stage1(lb) and C.stage2(la, 2000) == C.stage2(lb, 2000)


def test_loader_never_returns_validation_fields(tmp_path):
    build(str(tmp_path), A)
    for v in C.load_training_losses(str(tmp_path)).values():
        assert isinstance(v, float)
    assert not any("val" in k or "test" in k for k in C.LOADED_KEYS)


def test_other_stages_seeds_and_failed_runs_are_ignored(tmp_path):
    build(str(tmp_path), A)
    write(str(tmp_path), "A", 1000, 0.001, 1.0, stage="pilot")
    write(str(tmp_path), "A", 2000, 0.001, 1.0, seed=901)
    write(str(tmp_path), "A", 4000, 0.001, 1.0, status="failed")
    assert C.load_training_losses(str(tmp_path)) == {("A", 1000): 0.200, ("A", 2000): 0.150, ("A", 4000): 0.148}
