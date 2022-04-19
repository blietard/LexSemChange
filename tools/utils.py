import scipy
import numpy as np

def standardize(arr):
    return ( arr-arr.mean(0)  )/(arr.std(0) )

def unitcenter(arr):
    # the unit-center normalisation from
    # https://github.com/Garrafao/LSCDetection/blob/master/modules/embeddings.py
    arr_c = arr.copy()
    norms = np.sqrt(np.sum(arr_c**2, axis=1))
    norms[norms == 0] = 1
    arr_c /= norms[:, np.newaxis]
    avg = np.mean(arr_c, axis=0)
    arr_c -= avg
    return arr_c

def centerunit(arr):
    # same as unitcenter but operation order is reversed
    arr_c = arr.copy()
    avg = np.mean(arr_c, axis=0)
    arr_c -= avg
    norms = np.sqrt(np.sum(arr_c**2, axis=1))
    norms[norms == 0] = 1
    arr_c /= norms[:, np.newaxis]
    return arr_c


def OrthogProcrustAlign(arr1,arr2, standard=False, backward=False, std_func=standardize):
    '''
    Return Orthogonal Procrustes alignment matrix of arr1 and arr2.
    `standard` set to True if arr1 and arr2 are already standardized. Default is False.
    '''
    if standard:
        A = arr1
        B = arr2
    else:
        A = std_func(arr1)
        B = std_func(arr2)

    temp = B.T @ A
    U, e, Vt = np.linalg.svd(temp,)
    if backward:
        W = Vt.T @ U.T
    else:
        W = U @ Vt
    return W