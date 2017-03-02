import numpy as np

def gauss_eli(X):
    assert X.ndim == 2, "input must be a matrix"
    for i in range(0,X.shape[0] - 1):
        assert abs(X[i, i]) > 1e-5, "fail at i == %d" % (i,)
        for j in range(i+1, X.shape[0]):
            X[j] -= X[j,i] * X[i] / X[i,i]
    return X

def col_major_gauss_eli(X):
    assert X.ndim == 2, "input must be a matrix"
    row, col = X.shape
    for i in range(0, row - 1):
        m = i + np.argmax(np.abs(X[i:, i]))
        temp = X[m].copy()
        X[m] = X[i]
        X[i] = temp
        assert abs(X[i, i]) > 1e-5, "the matrix is nearly singular"
        for j in range(i+1, row):
            X[j] -= X[j,i] * X[i] / X[i,i]
    return X

def back_iter(X, style='U'):
    row, col= X.shape
    col -= 1
    assert X.ndim == 2, "input must be a matrix"
    assert row == col, "bad matrix shape"
    ret = np.zeros((row,),dtype = np.float32)
    if style == 'U':
        for i in range(row - 1, -1, -1):
            med = (X[i, i:col] * ret[i:]).sum()
            ret[i] = (X[i, col] - med) / X[i,i]
    elif style == 'L':
        for i in range(row):
            med = np.sum(X[i,:i] * ret[:i])
            ret[i] = (X[i, col] - med) / X[i,i]

    return ret

def solve(A,b):
    Ap = np.column_stack((A, b))
    re = col_major_gauss_eli(Ap.copy())
    return back_iter(re).T