import numpy as np

def recresid(x, y):
    n = x.shape[0]
    q = x.shape[1]
    rval = np.zeros(n - q)
    y1 = y[0:q]
    Q,R = np.linalg.qr(x[0:q])
    X1 = np.linalg.inv(R.T.dot(R))
    betar = np.nan_to_num(np.linalg.inv(R).dot(Q.T).dot(y1))
    xr = x[q,:]
    fr = 1 + (xr.T.dot(X1).dot(xr))
    rval[0] = (y[q] - xr.T.dot(betar))/np.sqrt(fr)

    if q < n:
        for r in range(q+1, n):
            X1 = X1 - (X1.dot(np.outer(xr, xr)).dot(X1))/fr
            betar = (betar.T + X1.dot(xr) * rval[r-q-1] * np.sqrt(fr)).T
            xr = x[r,:]
            fr = 1 + xr.T.dot(X1).dot(xr)
            rval[r-q] = (y[r] - np.sum(xr * betar.T))/np.sqrt(fr)

    return rval
