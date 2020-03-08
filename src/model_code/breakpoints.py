import math
import warnings

import numpy as np
from patsy import dmatrices

from src.model_code.recresid import recresid


def _bic(nobs, nreg, rss, df):
    """ Calculate Bayes' information criteria.

    """
    n = nobs
    log_l = -0.5 * n * (math.log(rss) + 1 - math.log(n) + math.log(2 * math.pi))
    bic = -2 * log_l + math.log(n) * df
    return bic


def breakpoints(formula, data, h=0.15, breaks=None):
    """Computes breakpoints in a regression.

     Given a maximum number of breaks calculates the optimal breakpoints. Based
     on strucchange package of R.

     Args:
        formula (str of generic Formula object): The formula specifying the model.
        data (array_like): The data from the model.
        h (float, optional): Minimal segment size. Can be given as a fraction
            of the sample size or as the minimal number of observations in each
            segment. Default is 0.15.
        breaks (int, optional): Maximal number of breaks to be calculated. Must
            be positive. If it's not given the maximal number allowed by h is
            used. Default is None.

    Returns:
        A tuple containing:
        opt_break (int): The optimal number of breaks.
        opt_obs (list of int or nan): The optimal breakpoints. If the optimal
            solution is one segment, returns NaN.
        opt_s (list of list of int ): The optimal breakpoints for each
            breakpoint from 1 to the maximal number of breaks specified.
        rss_s (list of float): The corresponding RSS.
        bic_s (list of float): The corresponding BIC.
        rss_triang (list of numpy.array of float): A list with the RSS triangular
            matrix

    Raises:
        ValueError: If 'h' is smaller than the number of the regressors.
        ValueError: If 'h' is greater than half of the number of observations.

    """
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
    # compute ith row of the RSS diagonal matrix
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

    rss_s = [rss_triang[0][n - 1]]
    bic_s = [
        n * (math.log(rss_s[0]) + 1 - math.log(n) + math.log(2 * math.pi))
        + math.log(n) * (k + 1)
    ]
    opt_s = []
    # compute optimal breaks from 1 to the number of maximum breaks
    for num_breaks in np.arange(1, breaks + 1):
        # breaks = 1
        break_rss = []
        for i in range(h, n - h + 1):
            rss = rss_triang[0][i - 1]
            break_rss.append(rss)

        index_np = np.arange(h, n - h + 1)
        rss_table_np = np.append([index_np], [np.array(break_rss).T], axis=0)
        nan_array = np.full(rss_table_np.shape, np.nan)
        # breaks >= 1
        if (num_breaks * 2) > rss_table_np.shape[0]:
            for m in np.arange(rss_table_np.shape[0] / 2 + 1, num_breaks + 1):
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
                        rss = (
                            my_rss_table[1][j_index]
                            + rss_triang[int(j)][int(i - j - 1)]
                        )
                        break_rss.append(rss)
                    opt = np.argmin(break_rss)
                    i_index = np.where(rss_table_np[0].astype(int) == i)[0]
                    my_rss_table[2][i_index] = pot_index[opt]
                    my_rss_table[3][i_index] = break_rss[opt]

                rss_table_np = np.concatenate((rss_table_np, my_rss_table[2:]))
        # extract optimal breaks.
        if (num_breaks * 2) > rss_table_np.shape[0]:
            raise ValueError("compute RSS.table with enough breaks before")

        break_rss = []
        for ind in index_np:
            ind_index_np = np.where(rss_table_np[0].astype(int) == ind)[0]
            rss = (
                rss_table_np[num_breaks * 2 - 1][ind_index_np]
                + rss_triang[int(ind)][n - int(ind) - 1]
            )
            break_rss.append(rss[0])
        opt = []
        opt.append(index_np[np.nanargmin(break_rss)])

        for j in np.flip(np.arange(2, num_breaks + 1)) * 2 - 2:
            opt_index = np.where(rss_table_np[0].astype(int) == opt[0])[0]
            opt.insert(0, int(rss_table_np[j][opt_index][0]))
        # calculate the RSS and the BIC for the break
        bp = [0] + opt + [n]
        rss_bp = 0
        for brp in range(0, len(bp) - 1):
            rss_bp += rss_triang[bp[brp]][bp[brp + 1] - bp[brp] - 1]
        df = (k + 1) * (len(opt) + 1)
        bic_bp = _bic(n, k, rss_bp, df)

        opt_s.append(opt)
        rss_s.append(rss_bp)
        bic_s.append(bic_bp)
    # choose optimal break and corresponding observations
    opt_break = np.argmin(bic_s)
    if opt_break == 0:
        opt_obs = np.nan
    else:
        opt_obs = opt_s[opt_break - 1]

    return (opt_break, opt_obs, opt_s, rss_s, bic_s, rss_triang)
