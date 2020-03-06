import math
import warnings

import numpy as np
from patsy import dmatrices

from src.model_code.recresid import recresid


def breakpoints(formula, h, breaks, data):
    y, X = dmatrices(formula, data, return_type="matrix")

    n = X.shape[0]
    k = X.shape[1]
    intercept_only = np.array_equal(np.ones(n), np.asarray(X))
    if h is None:
        h = k + 1
    if h < 1:
        h = math.floor(n * h)
    if h <= k:
        raise ValueError(
            "minimum segment size must be greater than the number of regressors"
        )
    if h > (math.floor(n / 2)):
        raise ValueError(
            "minimum segment size must be smaller than half the number of observations"
        )
    if breaks is None:
        breaks = math.ceil((n / h) - 2)
    else:
        if breaks < 1:
            breaks = 1
            warnings.warn("number of breaks must be at least 1", stacklevel=2)
        if breaks > math.ceil((n / h) - 2):
            breaks_zero = breaks
            breaks = math.ceil((n / h) - 2)
            warnings.warn(
                f"requested number of breaks =  {breaks_zero} too large, changed to  {breaks}",
                stacklevel=2,
            )

    rss_triang = []
    for i in np.arange(0, n - h + 1):
        if intercept_only:
            ssr = (y[i:n].T - np.cumsum(y[i:n]) / (np.arange(1, n - i + 1)))[0][
                1:
            ] * np.sqrt(1 + 1 / (np.arange(1, n - i)))
        else:
            ssr = recresid(X[i:n], y[i:n])
        rssi = np.concatenate((np.repeat(np.nan, k), np.cumsum(ssr ** 2)), axis=None)
        rss_triang.append(rssi)

    break_rss = []
    for i in range(h, n - h + 1):
        rss = rss_triang[0][i - 1]
        break_rss.append(rss)

    index_np = np.arange(h, n - h + 1)
    rss_table_np = np.append([index_np], [np.array(break_rss).T], axis=0)
    nan_array = np.full(rss_table_np.shape, np.nan)

    if (breaks * 2) > rss_table_np.shape[0]:
        for m in np.arange(rss_table_np.shape[0] / 2 + 1, breaks + 1):
            my_index = np.arange(m * h, n - h + 1)
            my_rss_table = np.append(
                [rss_table_np[int((m - 1) * 2 - 2)]],
                [rss_table_np[int((m - 1) * 2 - 1)]],
                axis=0,
            )
            my_rss_table = np.concatenate((my_rss_table, nan_array), axis=0)

            for i in my_index:
                pot_index = np.arange((m - 1) * h, i - h + 1)
                break_rss = []
                for j in pot_index:
                    j_index = np.where(rss_table_np[0].astype(int) == int(j))[0]
                    rss = my_rss_table[1][j_index] + rss_triang[int(j)][int(i - j - 1)]
                    break_rss.append(rss)
                opt = np.argmin(break_rss)
                i_index = np.where(rss_table_np[0].astype(int) == i)[0]
                my_rss_table[2][i_index] = pot_index[opt]
                my_rss_table[3][i_index] = break_rss[opt]

            rss_table_np = np.concatenate((rss_table_np, my_rss_table[2:]))
    if (breaks * 2) > rss_table_np.shape[0]:
        raise ValueError("compute RSS.table with enough breaks before")

    break_rss = []
    for ind in index_np:
        ind_index_np = np.where(rss_table_np[0].astype(int) == ind)[0]
        rss = (
            rss_table_np[breaks * 2 - 1][ind_index_np]
            + rss_triang[int(ind)][n - int(ind) - 1]
        )
        break_rss.append(rss[0])
    opt = []
    opt.append(index_np[np.nanargmin(break_rss)])

    for j in np.flip(np.arange(2, breaks + 1)) * 2 - 2:
        opt_index = np.where(rss_table_np[0].astype(int) == opt[0])[0]
        opt.insert(0, int(rss_table_np[j][opt_index][0]))

    return opt
