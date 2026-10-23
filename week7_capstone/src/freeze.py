"""Make the pre-registration freeze checkable.

    python src/freeze.py --write     # run once, right after the commit that contains the final frozen files
    python src/freeze.py --verify    # exit 5 if anything frozen has changed

The freeze record (results/freeze_record.json) stores a SHA-256 for every file that defines the experiment or its decision
rule (FROZEN_FILES), the commit those files were committed in, and a UTC timestamp. A file cannot contain its own commit
hash, so the record is a separate file, committed afterwards. main and extension runs are refused unless every frozen file
still matches its recorded hash and the recorded commit is an ancestor of the code being run. Notebooks, tests and
documentation are deliberately not frozen, so they can be fixed without invalidating the freeze; changing a frozen file
requires a new, logged freeze.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_FILES = ["PREREGISTRATION.md", "configs/base.yaml", "configs/arm_A.yaml", "configs/arm_B.yaml", "configs/arm_C.yaml",
                "requirements.lock.txt", "src/run_ablation.py", "src/model.py", "src/make_data.py", "src/analyze_main.py",
                "src/run_main.py", "src/check_env.py", "src/freeze.py"]
DEFAULT_RECORD = os.path.join(ROOT, "results", "freeze_record.json")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def git(*args, root=ROOT):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=root)


def build_record(root=ROOT, frozen_commit=None, files=FROZEN_FILES):
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_commit": frozen_commit or git("rev-parse", "HEAD", root=root).stdout.strip(),
            "files": {f: sha256(os.path.join(root, f)) for f in files},
            "note": "SHA-256 of every file that defines the experiment or its decision rule. A change to any of them blocks main "
                    "and extension runs. The commit hash of the commit that adds this record is not stored here (a record cannot "
                    "contain its own commit)."}


def verify(root=ROOT, record_path=None):
    """Return a list of problems (empty means the freeze holds)."""
    record_path = record_path or os.path.join(root, "results", "freeze_record.json")
    if not os.path.exists(record_path):
        return [f"freeze record missing: {record_path}"]
    rec = json.load(open(record_path))
    problems = []
    for f, h in rec["files"].items():
        p = os.path.join(root, f)
        if not os.path.exists(p):
            problems.append(f"frozen file missing: {f}")
        elif sha256(p) != h:
            problems.append(f"frozen file changed since the freeze: {f}")
    c = rec.get("frozen_commit")
    if not c:
        problems.append("freeze record has no frozen_commit")
    elif git("cat-file", "-e", f"{c}^{{commit}}", root=root).returncode != 0:
        problems.append(f"frozen commit {c[:10]} is not in this checkout")
    elif git("merge-base", "--is-ancestor", c, "HEAD", root=root).returncode != 0:
        problems.append(f"frozen commit {c[:10]} is not an ancestor of the code being run")
    return problems


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--verify", action="store_true")
    ap.add_argument("--record", default=DEFAULT_RECORD)
    args = ap.parse_args()
    if args.write:
        dirty = git("status", "--porcelain", "--untracked-files=no", "--", *FROZEN_FILES).stdout.strip()
        if dirty:
            sys.exit(f"refusing to freeze: frozen files differ from HEAD (commit them first):\n{dirty}")
        if os.path.exists(args.record):
            sys.exit(f"{args.record} exists; a freeze is never overwritten silently")
        rec = build_record()
        os.makedirs(os.path.dirname(args.record), exist_ok=True)
        json.dump(rec, open(args.record, "w"), indent=1)
        print(f"freeze record written: commit {rec['frozen_commit']}, {len(rec['files'])} files, {rec['created_utc']}")
    else:
        problems = verify(record_path=args.record)
        if problems:
            print("FREEZE BROKEN:\n  " + "\n  ".join(problems))
            sys.exit(5)
        rec = json.load(open(args.record))
        print(f"freeze holds: {len(rec['files'])} files match; frozen commit {rec['frozen_commit'][:10]} (created {rec['created_utc'][:19]} UTC)")


if __name__ == "__main__":
    main()
