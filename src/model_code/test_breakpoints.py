import json
import numbers

import numpy as np
import pandas as pd
import pytest

from bld.project_paths import project_paths_join as ppj
from src.model_code.breakpoints import breakpoints as brkpts

DB = ["Grossarl", "GermanM1"]


@pytest.fixture
def setup_predict():
    out = {
        "Grossarl": {"formula": "marriages ~ 1", "h": 0.1},
        "GermanM1": {"formula": "dm ~ dy2 + dR + dR1 + dp + m1 + y1 + R1", "h": 0.15},
    }

    for i in DB:
        data = pd.read_csv(ppj("OUT_DATA", f"{i}.csv"))
        out[f"{i}"]["data"] = data
    return out


@pytest.fixture
def expected_predict():
    out = {}
    for i in DB:
        with open(ppj("OUT_MODEL_SPECS", f"{i}_bp.json")) as f:
            info = json.load(f)
        breakpoints_list = []
        for j in range(0, len(info["breakpoints"])):
            breakpoints_list.append(
                [x for x in info["breakpoints"][j] if isinstance(x, numbers.Number)]
            )

        rss = info["RSS"][0]

        bic = info["RSS"][1]
        out[f"{i}"] = {"breakpoints": breakpoints_list, "rss": rss, "bic": bic}

    return out


def test_breakpoints_breakpoints_only_intercept(setup_predict, expected_predict):
    db = DB[0]
    calc_breakpoints = brkpts(
        formula=setup_predict[db]["formula"],
        data=setup_predict[db]["data"],
        h=setup_predict[db]["h"],
    )[2]
    np.testing.assert_equal(calc_breakpoints, expected_predict[db]["breakpoints"])


def test_rss_breakpoints_only_intercept(setup_predict, expected_predict):
    db = DB[0]
    calc_rss = brkpts(
        formula=setup_predict[db]["formula"],
        data=setup_predict[db]["data"],
        h=setup_predict[db]["h"],
    )[3]
    np.testing.assert_almost_equal(
        calc_rss, expected_predict[db]["rss"], decimal=4, verbose=True
    )


def test_bic_breakpoints_only_intercept(setup_predict, expected_predict):
    db = DB[0]
    calc_bic = brkpts(
        formula=setup_predict[db]["formula"],
        data=setup_predict[db]["data"],
        h=setup_predict[db]["h"],
    )[4]
    np.testing.assert_almost_equal(
        calc_bic, expected_predict[db]["bic"], decimal=4, verbose=True
    )


def test_breakpoints_breakpoints_more_regressors(setup_predict, expected_predict):
    db = DB[1]
    calc_breakpoints = brkpts(
        formula=setup_predict[db]["formula"],
        data=setup_predict[db]["data"],
        h=setup_predict[db]["h"],
    )[2]
    np.testing.assert_equal(calc_breakpoints, expected_predict[db]["breakpoints"])


def test_rss_breakpoints_more_regressors(setup_predict, expected_predict):
    db = DB[1]
    calc_rss = brkpts(
        formula=setup_predict[db]["formula"],
        data=setup_predict[db]["data"],
        h=setup_predict[db]["h"],
    )[3]
    np.testing.assert_almost_equal(
        calc_rss, expected_predict[db]["rss"], decimal=4, verbose=True
    )


def test_bic_breakpoints_more_regressors(setup_predict, expected_predict):
    db = DB[1]
    calc_bic = brkpts(
        formula=setup_predict[db]["formula"],
        data=setup_predict[db]["data"],
        h=setup_predict[db]["h"],
    )[4]
    np.testing.assert_almost_equal(
        calc_bic, expected_predict[db]["bic"], decimal=4, verbose=True
    )
