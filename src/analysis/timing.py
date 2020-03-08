import pickle
from time import time

import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.model_code.breakpoints import breakpoints


def rand_seed(nreg, nobs, breaks):
    """Randomly seeds a dataset with structural breaks of number 'breaks'.
    Returns a string representing the formula of the regression and a
    panda.DataFrame with the data.

    """
    breaks = breaks + 1
    X = np.random.rand(nreg, nobs)
    betas = np.random.rand(breaks, nreg)
    h = int(nobs / breaks)

    y = []
    for _brk in np.arange(0, breaks):
        betas = (10 + _brk) * np.random.rand(nreg)
        y_mean = np.dot(betas, X[:, _brk * h : (_brk + 1) * h])
        y += (
            y_mean - np.multiply(y_mean, np.random.normal(0, 0.01, y_mean.shape[0]))
        ).tolist()

    y = np.array(y)

    c_names = ["y"]
    for i in np.arange(1, nreg + 1):
        c_name = f"x{i}"
        c_names.append(c_name)

    data_np = np.append([[np.array(y)]], [X], axis=1)[0]

    data = pd.DataFrame(data_np.T, columns=c_names)

    # get column names.
    c_names = list(data.columns.values)

    # generate formula.
    formula = f"{c_names[0]} ~"

    for _j in c_names[1:]:
        if _j == c_names[-1]:
            formula += f" {_j}"
        else:
            formula += f" {_j} +"

    return formula, data


# timing of the function.
def timing_of_function(nreg, nobs, breaks, nruns):
    """Calculate the times of running breakpoints function

    """
    formula, data = rand_seed(nreg=nreg, nobs=nobs, breaks=breaks)
    runtimes = []

    for _j in range(nruns):
        start = time()
        breakpoints(formula, data)
        stop = time()
        runtimes.append(stop - start)
    return runtimes


nreg = 3
breaks = 4
nruns = 6

x = np.arange(50, 550, 25)

y = []
for _i in range(50, 550, 25):
    runtimes = timing_of_function(nreg, _i, breaks, nruns)
    runtime_mean = np.mean(runtimes[1:])
    y.append(runtime_mean)

axis = np.concatenate(([x], [np.array(y)]))
# store the information of the timing
with open(ppj("OUT_ANALYSIS", "timing_info.pickle"), "wb") as out_file:
    pickle.dump(axis, out_file)
