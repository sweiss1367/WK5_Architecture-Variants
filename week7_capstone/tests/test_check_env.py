import json
import os
import subprocess
import sys

import pytest

import check_env as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cli(tmp_path, *args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "src", "check_env.py"), "--lock", str(tmp_path / "lock.txt"),
                           "--report", str(tmp_path / "rep" / "env_report.json"), *args], capture_output=True, text=True)


def test_capture_pins_packages_and_never_overwrites(tmp_path):
    lock = str(tmp_path / "lock.txt")
    E.capture(lock)
    pins, py = E.read_lock(lock)
    assert set(pins) == set(E.PINNED) and all(v == E.installed(n) for n, v in pins.items()) and py
    with pytest.raises(FileExistsError):
        E.capture(lock)


def test_verify_detects_a_changed_version_and_a_missing_lock(tmp_path):
    lock = str(tmp_path / "lock.txt")
    assert E.verify(lock)["lock_present"] is False
    E.capture(lock)
    assert E.verify(lock)["matches"] is True
    text = open(lock).read().replace(f"numpy=={E.installed('numpy')}", "numpy==0.0.1")
    open(lock, "w").write(text)
    rep = E.verify(lock)
    assert rep["matches"] is False and rep["mismatches"][0]["package"] == "numpy"


def test_auto_captures_once_then_verifies_and_flags_a_mismatch(tmp_path):
    p = cli(tmp_path, "--auto")
    assert p.returncode == 0 and "lock written" in p.stdout
    rep = json.load(open(tmp_path / "rep" / "env_report.json"))
    assert rep["captured_this_run"] is True and rep["matches"] is True and os.path.exists(rep["full_freeze_file"])
    p = cli(tmp_path, "--auto")
    assert p.returncode == 0 and "matches the lock" in p.stdout
    text = open(tmp_path / "lock.txt").read().replace(f"numpy=={E.installed('numpy')}", "numpy==0.0.1")
    open(tmp_path / "lock.txt", "w").write(text)
    p = cli(tmp_path, "--auto")
    assert p.returncode == 4 and "DIFFERS" in p.stdout
