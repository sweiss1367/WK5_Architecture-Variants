"""Record the software environment and check it against the pinned lock. Never installs or upgrades anything.

    python src/check_env.py --auto      # no lock yet: capture one from this runtime; lock present: verify it
    python src/check_env.py --capture   # write requirements.lock.txt from the current runtime (refuses to overwrite)
    python src/check_env.py --verify    # compare this runtime with the lock; exit 4 on a mismatch

The lock pins the packages the experiment imports (PINNED) plus the Python minor version.
Colab preinstalls torch, so the lock records what Colab provides on the day of the run and is
checked in later sessions. The full `pip freeze` equivalent is saved next to the report, with a hash.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

PINNED = ["torch", "numpy", "PyYAML"]
LOCK_DEFAULT = "requirements.lock.txt"
REPORT_DEFAULT = "results/env_report.json"


def installed(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def full_freeze():
    return sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())


def read_lock(path):
    pins, py = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith("# python"):
            py = line.split()[-1]
        elif line and not line.startswith("#") and "==" in line:
            n, v = line.split("==", 1)
            pins[n.strip()] = v.strip()
    return pins, py


def capture(path):
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists; the lock is never overwritten silently")
    py = ".".join(platform.python_version_tuple()[:2])
    lines = [f"# Captured {datetime.now(timezone.utc).isoformat()} on {platform.platform()}",
             f"# python {py}"] + [f"{n}=={installed(n)}" for n in PINNED]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def verify(path=LOCK_DEFAULT):
    """Return a report dict. matches is True only if a lock exists and every pinned version agrees."""
    rep = {"lock_path": path, "lock_present": os.path.exists(path), "matches": None, "mismatches": [],
           "python": platform.python_version()}
    if not rep["lock_present"]:
        return rep
    pins, py = read_lock(path)
    mism = [{"package": n, "pinned": v, "installed": installed(n)} for n, v in pins.items() if installed(n) != v]
    if py and ".".join(platform.python_version_tuple()[:2]) != py:
        mism.append({"package": "python", "pinned": py, "installed": platform.python_version()})
    rep["mismatches"], rep["matches"] = mism, not mism
    return rep


def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out or None
    except Exception:
        return None


def write_report(rep, report_path):
    freeze = full_freeze()
    digest = hashlib.sha256("\n".join(freeze).encode()).hexdigest()
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    freeze_path = os.path.join(os.path.dirname(report_path) or ".", f"pip_freeze_{digest[:10]}.txt")
    with open(freeze_path, "w") as f:
        f.write("\n".join(freeze) + "\n")
    rep.update(created_utc=datetime.now(timezone.utc).isoformat(), full_freeze_sha256=digest, full_freeze_file=freeze_path,
               pinned_installed={n: installed(n) for n in PINNED}, platform=platform.platform(), gpu=gpu_info())
    with open(report_path, "w") as f:
        json.dump(rep, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--auto", action="store_true")
    g.add_argument("--capture", action="store_true")
    g.add_argument("--verify", action="store_true")
    ap.add_argument("--lock", default=LOCK_DEFAULT)
    ap.add_argument("--report", default=REPORT_DEFAULT)
    args = ap.parse_args()

    captured = False
    if args.capture or (args.auto and not os.path.exists(args.lock)):
        capture(args.lock)
        captured = True
    rep = verify(args.lock)
    rep["captured_this_run"] = captured
    write_report(rep, args.report)

    print("pinned packages:", {n: installed(n) for n in PINNED}, "| python", rep["python"])
    if captured:
        print(f"lock written to {args.lock}. Commit it. Later sessions are checked against it.")
    elif rep["matches"]:
        print("environment matches the lock")
    elif rep["lock_present"]:
        print("ENVIRONMENT DIFFERS FROM THE LOCK:", rep["mismatches"])
        sys.exit(4)
    else:
        print("no lock present")
        sys.exit(4)


if __name__ == "__main__":
    main()
