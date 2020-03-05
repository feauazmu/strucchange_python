import numpy as np


def recresid(x, y):
    n = x.shape[0]
    q = x.shape[1]
    rval = np.zeros(n - q)
    y1 = y[0:q]
    Q, R = np.linalg.qr(x[0:q])
    X1 = np.linalg.inv(np.dot(R.T, R))
    betar = np.nan_to_num(np.dot(np.dot(np.linalg.inv(R), Q.T), y1))
    xr = x[q, :]
    fr = 1 + (np.dot(np.dot(xr.T, X1), xr))
    rval[0] = (y[q] - np.dot(xr.T, betar)) / np.sqrt(fr)

    if q < n:
        for r in range(q + 1, n):
            X1 = X1 - (np.dot(np.dot(X1, np.outer(xr, xr)), X1)) / fr
            betar = (betar.T + np.dot(X1, xr) * rval[r - q - 1] * np.sqrt(fr)).T
            xr = x[r, :]
            fr = 1 + np.dot(np.dot(xr.T, X1), xr)
            rval[r - q] = (y[r] - np.sum(xr * betar.T)) / np.sqrt(fr)

    return rval
