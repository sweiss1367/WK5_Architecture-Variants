import json
import os
import subprocess
import sys

import pytest

import run_ablation as RA
import smoke_checks as SC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_p1_sets_cublas_workspace_and_p2_removes_it(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")        # an inherited setting must not leak into P2
    RA.configure_policy("P2")
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    RA.configure_policy("P1")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    RA.configure_policy("P2")                                      # restore the default process state for other tests
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_run_records_the_workspace_setting_for_each_policy(tmp_path, base_cfg):
    import copy, yaml
    cfg = copy.deepcopy(base_cfg); cfg["data"]["n_series"] = 2
    base = tmp_path / "b.yaml"; base.write_text(yaml.safe_dump(cfg))
    seen = {}
    for pol in ("P1", "P2"):
        p = subprocess.run([sys.executable, os.path.join(ROOT, "src", "run_ablation.py"), "--config", os.path.join(ROOT, "configs", "arm_B.yaml"),
                            "--base", str(base), "--seed", "0", "--stage", "smoke", "--steps", "10", "--policy", pol, "--timing-only",
                            "--out", str(tmp_path / pol), "--manifest", str(tmp_path / "m.json"), "--data-dir", str(tmp_path / "d")],
                           capture_output=True, text=True, cwd=ROOT, env={k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"})
        assert p.returncode == 0, p.stderr
        seen[pol] = json.load(open(tmp_path / pol / "smoke" / "B_seed0_attempt1.json"))["cublas_workspace_config"]
    assert seen == {"P1": ":4096:8", "P2": None}


def test_default_session_id_is_shared_by_all_scripts(tmp_path, base_cfg):
    """No script overrides the session id any more, so the manifest counts VM sessions, not scripts."""
    for f in ("choose_policy.py", "smoke_checks.py"):
        assert "--session-id" not in open(os.path.join(ROOT, "src", f)).read().split('"""', 2)[2]


@pytest.mark.parametrize("run,probe,ok", [(65.627, 65.627, True), (65.627, 65.0, True), (79.9, 65.6, False), (65.6, 67.0, False)])
def test_v16_verdict_flags_leftover_memory(run, probe, ok):
    assert SC.v16_verdict(run, probe)[0] is ok


def test_v16_would_have_failed_a_real_leak():
    """If the 12.2 MiB of on-GPU datasets were not freed, the run would hold that much more than the probe."""
    assert SC.v16_verdict(65.6 + 12.24, 65.6)[0] is False


def test_arm_order_rotates_across_repetitions():
    import choose_policy as CP
    arms = ["A", "B", "C"]
    assert [CP.rotated(arms, r) for r in range(4)] == [["A", "B", "C"], ["B", "C", "A"], ["C", "A", "B"], ["A", "B", "C"]]
