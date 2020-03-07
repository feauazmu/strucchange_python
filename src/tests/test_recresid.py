import json

import numpy as np
import pandas as pd
import pytest
from patsy import dmatrices

from bld.project_paths import project_paths_join as ppj
from src.model_code.recresid import recresid

DB = ["Grossarl", "GermanM1"]


@pytest.fixture
def setup_predict():
    out = {}
    formulas = {
        "Grossarl": "marriages ~ 1",
        "GermanM1": "dm ~ dy2 + dR + dR1 + dp + m1 + y1 + R1",
    }
    for i in DB:
        data = pd.read_csv(ppj("OUT_DATA", f"{i}.csv"))
        y, X = dmatrices(formulas[f"{i}"], data, return_type="matrix")
        out[f"{i}"] = {"X": X, "y": y}
    return out


@pytest.fixture
def expected_predict():
    out = {}
    for i in DB:
        with open(ppj("OUT_MODEL_SPECS", f"{i}_rid.json")) as f:
            data = json.load(f)
        out[f"{i}"] = np.array(data)
    return out


def test_recresid_only_intercept(setup_predict, expected_predict):
    db = DB[0]
    calc_recresid = recresid(setup_predict[db]["X"], setup_predict[db]["y"])
    np.testing.assert_almost_equal(
        calc_recresid, expected_predict[db], decimal=4, verbose=True
    )


def test_recresid_more_regressors(setup_predict, expected_predict):
    db = DB[1]
    calc_recresid = recresid(setup_predict[db]["X"], setup_predict[db]["y"])
    calc_recresid
    np.testing.assert_almost_equal(
        calc_recresid, expected_predict[db], decimal=4, verbose=True
    )


pytest.main()
