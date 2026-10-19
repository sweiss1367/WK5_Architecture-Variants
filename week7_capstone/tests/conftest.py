import copy
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


@pytest.fixture(scope="session")
def base_cfg():
    return yaml.safe_load(open(os.path.join(ROOT, "configs", "base.yaml")))


@pytest.fixture(scope="session")
def small_data_cfg(base_cfg):
    d = copy.deepcopy(base_cfg["data"])
    d["n_series"] = 3
    return d
