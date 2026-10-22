import json
import os

import calibrate as C


def write(d, arm, steps, train_loss, val_mape, status="completed", stage="calibration", seed=900):
    os.makedirs(d, exist_ok=True)
    json.dump({"arm": arm, "seed": seed, "steps": steps, "stage": stage, "status": status, "attempt": 1,
               "final_train_loss_100": train_loss, "train_seconds": 10.0, "val_mape": val_mape, "val_mae_kw": val_mape, "test_mape": val_mape,
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


def test_next_runs_asks_for_arm_a_first_then_only_what_the_rule_needs(tmp_path):
    assert C.next_runs({}) == [("A", 1000), ("A", 2000), ("A", 4000)]
    la = {("A", 1000): 0.200, ("A", 2000): 0.150, ("A", 4000): 0.148}
    assert C.next_runs(la) == [("B", 2000), ("B", 4000), ("C", 2000), ("C", 4000)]       # S* = 2000
    part = {**la, ("B", 2000): 0.151, ("B", 4000): 0.149}
    assert C.next_runs(part) == [("C", 2000), ("C", 4000)]
    assert C.next_runs(OK_ALL) == []                                                      # everything passes: done


def test_next_runs_moves_up_when_an_arm_has_not_plateaued():
    la = {("A", 1000): 0.1500, ("A", 2000): 0.1490, ("A", 4000): 0.1485}                  # S* = 1000
    assert C.next_runs(la) == [("B", 1000), ("B", 2000), ("C", 1000), ("C", 2000)]
    loss = {**la, ("B", 1000): 0.1510, ("B", 2000): 0.1500, ("C", 1000): 0.20, ("C", 2000): 0.15}   # C still improving
    assert C.next_runs(loss) == [("B", 4000), ("C", 4000)]                                # now check S = 2000 -> 4000
    full = {**loss, ("B", 4000): 0.1495, ("C", 4000): 0.1495}
    d = C.decide(full)
    assert d["status"] == "ok" and d["recommended_S"] == 2000 and "moved up" in d["note"]
    assert C.next_runs(full) == []


def test_decide_reports_grid_exhausted_and_expansion_trigger():
    la = {("A", 1000): 0.30, ("A", 2000): 0.20, ("A", 4000): 0.15}
    d = C.decide(la)
    assert d["status"] == "no_grid_value_qualifies" and "8000" in d["next_action"] and C.next_runs(la) == []
    lb = {("A", 1000): 0.200, ("A", 2000): 0.150, ("A", 4000): 0.148, ("B", 2000): 0.20, ("B", 4000): 0.15,
          ("C", 2000): 0.151, ("C", 4000): 0.149}
    assert C.decide(lb)["status"] == "grid_exhausted"


def test_decision_ignores_validation_even_through_the_full_decide_path(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    build(str(a), OK_ALL, val=5.0); build(str(b), OK_ALL, val=-1e9)
    assert C.decide(C.load_training_losses(str(a))) == C.decide(C.load_training_losses(str(b)))


G8 = [1000, 2000, 4000, 8000]
A8 = {("A", 1000): 0.05330, ("A", 2000): 0.04839, ("A", 4000): 0.04531}


def test_outcome_improvement_below_threshold_selects_4000_after_stage_2():
    """4000 -> 8000 improves by less than 2%: the plateau is at 4000. 8000 is NOT the selected value."""
    h = {**A8, ("A", 8000): 0.04500}                                   # 0.68% improvement
    assert C.stage1(h, G8)[0] == 4000
    assert C.next_runs(h, G8) == [("B", 4000), ("B", 8000), ("C", 4000), ("C", 8000)]
    assert C.decide(h, G8)["status"] == "missing_runs"
    full = {**h, ("B", 4000): 0.04600, ("B", 8000): 0.04580, ("C", 4000): 0.04700, ("C", 8000): 0.04690}
    d = C.decide(full, G8)
    assert d["status"] == "ok" and d["selected_S"] == 4000 and "limitation" not in d
    assert C.next_runs(full, G8) == []


def test_outcome_improvement_at_or_above_threshold_falls_back_to_8000_with_limitation():
    """4000 -> 8000 still improves by 2% or more: the pre-registered fallback is the largest value, plus a limitation."""
    h = {**A8, ("A", 8000): 0.04300}                                   # 5.1% improvement
    assert C.next_runs(h, G8) == []                                    # stage 2 is not called for, nothing more is run
    d = C.decide(h, G8)
    assert d["status"] == "no_grid_value_qualifies" and d["selected_S"] == 8000
    assert "pre-registered fallback" in d["selection_basis"] and "not converged" in d["limitation"]
    assert "without approval" in d["next_action"]                      # no 16000 or other new value is added silently


def test_exactly_two_percent_is_not_below_the_threshold():
    h = {**A8, ("A", 8000): 0.04531 * 0.98}                            # exactly 2.000% improvement
    assert C.stage1(h, G8)[0] is None


def test_stage_2_failure_at_4000_exhausts_the_grid_and_falls_back_to_8000():
    h = {**A8, ("A", 8000): 0.04500, ("B", 4000): 0.0480, ("B", 8000): 0.0450, ("C", 4000): 0.0470, ("C", 8000): 0.0468}
    d = C.decide(h, G8)                                                 # arm B still improves 6.3% from 4000 to 8000
    assert d["status"] == "grid_exhausted" and d["selected_S"] == 8000 and "not converged" in d["limitation"]


def test_original_grid_still_reports_the_expansion_trigger():
    d = C.decide(A8)
    assert d["status"] == "no_grid_value_qualifies" and "selected_S" not in d and "8000" in d["next_action"]


def test_arm_a_improvements_are_always_reported():
    d = C.decide({**A8, ("A", 8000): 0.04300}, G8)
    assert set(d["arm_A_relative_improvements"]) == {"1000->2000", "2000->4000", "4000->8000"}
