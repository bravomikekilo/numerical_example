import numpy as np
from gauss_eli import back_iter

def doolittle(X, inplace=True):
    """
       杜利特分解 doolittle decomposition
       ==================================
       params:
            X: 
                numpy.ndarrray which ndim == 2 as a matrix
                shape requirement: row == col
            inplace: bool 
                whether do the decomposition inplace
        returns:
            (L, U):
                tuple(numpy.ndarray, numpy.ndarray)
                which ndim == 2 as a matrix
                L,U will be X itself if inplace is True

    """
    assert X.ndim == 2 , "input must be a matrix"
    row, col = X.shape
    assert row == col , "bad matrix shape"
    if inplace:
        U = X
        L = X
    else:
        L = np.eye(row, dtype=np.float32)
        U = np.zeros_like(X, dtype=np.float32)

    for i in range(row):
        for j in range(i, col):
            U[i,j] = X[i,j] - np.sum(U[:i, j] * L[i, :i])
        major = U[i,i]
        assert abs(U[i,i]) > 1e-5 , "fail at k = %d" % (i,)
        for j in range(i+1, col):
            L[j, i] = ( X[j, i] - np.sum(U[:i, i] * L[j, :i]) ) / major
    return L, U

def column_major_doolittle(X):
    """
        列主元杜利特分解 column major doolittle decompostion
        ====================================================
        params:
            X:
                numpy.ndarray which ndim == 2 as a matrix
                shape requirement: row == col
        returns:
            (L, U):
                tuple(numpy.ndarray, numpy.ndarray)
                which ndim == 2 as a matrix
         
    """
    assert X.ndim == 2, "input must be a matrix"
    row, col = X.shape
    assert row == col, "bad matrix shape"
    L = np.eye(row, dtype=np.float32)
    U = np.zeros_like(X, dtype=np.float32)
    for i in range(row):
        for j in range(i, col):
            U[i,j] = X[i,j] - np.sum(U[:i, j] * L[i, :i])
        major = U[i,i]
        for j in range(i+1, col):
            L[j, i] = ( X[j, i] - np.sum(U[:i, i] * L[j, :i]) ) / major
    return L, U

def crout(X, inplace=True):
    """
       克劳特分解 crout decomposition
       ==================================
       params:
            X: 
                numpy.ndarrray which ndim == 2 as a matrix
                shape requirement: row == col
            inplace: bool 
                whether do the decomposition inplace
        returns:
            (L, U):
                tuple(numpy.ndarray, numpy.ndarray)
                which ndim == 2 as a matrix
                L,U will be X itself if inplace is True

    """
    assert X.ndim == 2 , "input must be a matrix"
    row, col = X.shape
    assert row == col, "bad matrix shape"
    if inplace:
        U = X
        L = X
    else:
        U = np.eye(row, dtype=np.float32)
        L = np.zeros_like(X, dtype=np.float32)

    for k in range(row):
        for i in range(k, row):
            L[i, k] = X[i, k] - np.sum(L[i, :k] * U[:k, k])
        major = L[k, k]
        assert abs(L[k, k]) > 1e-5, "fail at k = %d" % (k,)
        for j in range(k+1, col):
            U[k, j] = ( X[k, j] - np.sum(L[k, :k] * U[:k, j]) ) / major
    return L, U


def inplace_back_iter(X, style='crout'):
    """
        原址反向迭代 inplace back iteration
        ===================================
        params:
            X:
                numpy.ndarrary which ndim == 2 as a Augmented matrix
            style:
                str back_iter style "crout" or "doolittle"
        returns:
            numpy.ndarray which ndim == 1 as a vector
            result of the back iteration
                
    """
    assert X.ndim == 2, "input must be a matrix"
    row, col = X.shape
    col -= 1
    assert row == col, "bad matrix shape"
    ret = np.zeros((row,),dtype = np.float32)
    if style == 'crout':
        t = back_iter(X, style='L')
        X[:, -1] = t
        for i in range(row - 1, -1, -1):
            med = (X[i, i:col] * ret[i:]).sum()
            ret[i] = (X[i, col] - med)
        return ret
    elif style == 'doolittle':
        for i in range(row):
            med = np.sum(X[i,:i] * ret[:i])
            ret[i] = (X[i, col] - med)
        X[:, -1] = ret
        return back_iter(X, style='U')
    else:
        raise ValueError('Unknown iter style')


def LU_solve(A, b, inplace=True):
    """
        LU 分解求解线性方程组 solve linear equations by LU decomposition
        ================================================================
        params:
            A:
                numpy.ndarray which ndim == 2 as a matrix
                shape requirement: row == col
            b:
                numpy.ndarray which ndim == 1 as a vector
                shape requirement len(b) == row
            inplace:
                bool whether do LU decomposition inplace
        returns:
            numpy.ndarray:
                numpy.ndarray which ndim == 1 as a vector
                result of input linear equations
    """
    if inplace:
        doolittle(A)
        return inplace_back_iter(np.column_stack([A,b]), style='doolittle')
    else:
        b = np.expand_dims(b, 1)
        L, U = doolittle(A, inplace=False)
        return back_iter(np.column_stack([U, back_iter(np.column_stack([L,b]),style='L')]) ,style='U')

