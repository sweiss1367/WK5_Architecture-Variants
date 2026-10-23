import json
import os
import subprocess

import pytest

import freeze as F


def git(cwd, *a):
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=cwd, check=True, capture_output=True)


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "r"
    for f, body in (("PREREGISTRATION.md", "frozen text\n"), ("src/a.py", "x = 1\n"), ("configs/base.yaml", "a: 1\n"), ("docs/notes.md", "n\\n")):
        (r / f).parent.mkdir(parents=True, exist_ok=True); (r / f).write_text(body)
    git(r, "init", "-q"); git(r, "add", "-A"); git(r, "commit", "-q", "-m", "frozen content")
    files = ["PREREGISTRATION.md", "src/a.py", "configs/base.yaml"]
    rec = F.build_record(str(r), files=files)
    path = r / "results" / "freeze_record.json"
    path.parent.mkdir(); path.write_text(json.dumps(rec))
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "freeze record")          # the record is its own, later commit
    return r, files, str(path)


def test_a_fresh_freeze_holds_and_records_the_commit_of_the_frozen_content(repo):
    r, files, path = repo
    assert F.verify(str(r), path) == []
    rec = json.load(open(path))
    assert set(rec["files"]) == set(files) and rec["frozen_commit"] != subprocess.run(["git", "rev-parse", "HEAD"], cwd=r, capture_output=True, text=True).stdout.strip()


@pytest.mark.parametrize("name", ["PREREGISTRATION.md", "src/a.py", "configs/base.yaml"])
def test_changing_any_frozen_file_breaks_the_freeze(repo, name):
    r, _, path = repo
    (r / name).write_text("edited after the freeze\n")
    problems = F.verify(str(r), path)
    assert problems and any(name in p for p in problems)


def test_unfrozen_files_may_change_without_breaking_the_freeze(repo):
    r, _, path = repo
    (r / "docs" / "notes.md").write_text("fixed a typo\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "docs only")
    assert F.verify(str(r), path) == []                                          # a later commit that touches nothing frozen is fine


def test_a_missing_record_a_missing_file_and_a_foreign_commit_are_reported(repo, tmp_path):
    r, _, path = repo
    assert "missing" in F.verify(str(r), str(tmp_path / "nope.json"))[0]
    os.remove(r / "src" / "a.py")
    assert any("missing: src/a.py" in p for p in F.verify(str(r), path))
    rec = json.load(open(path)); rec["frozen_commit"] = "1" * 40
    open(path, "w").write(json.dumps(rec))
    assert any("not in this checkout" in p for p in F.verify(str(r), path))


def test_the_frozen_commit_must_be_an_ancestor_of_the_code_being_run(repo):
    r, _, path = repo
    home = subprocess.run(["git", "branch", "--show-current"], cwd=r, capture_output=True, text=True).stdout.strip()
    git(r, "checkout", "-q", "--orphan", "unrelated"); git(r, "rm", "-rqf", ".")
    (r / "z.txt").write_text("z"); git(r, "add", "-A"); git(r, "commit", "-q", "-m", "unrelated history")
    unrelated = subprocess.run(["git", "rev-parse", "HEAD"], cwd=r, capture_output=True, text=True).stdout.strip()
    git(r, "checkout", "-q", home)                                             # back to the real history
    rec = json.load(open(path)); rec["frozen_commit"] = unrelated
    open(path, "w").write(json.dumps(rec))
    assert any("not an ancestor" in p for p in F.verify(str(r), path))


def test_the_frozen_file_list_covers_what_defines_the_experiment_and_the_rule():
    for f in ("PREREGISTRATION.md", "configs/base.yaml", "src/analyze_main.py", "src/run_ablation.py", "src/model.py", "src/make_data.py",
              "requirements.lock.txt", "configs/arm_A.yaml", "configs/arm_B.yaml", "configs/arm_C.yaml"):
        assert f in F.FROZEN_FILES
    assert "colab/week7_step6_main.ipynb" not in F.FROZEN_FILES and not any(f.startswith(("tests/", "docs/", "colab/")) for f in F.FROZEN_FILES)
