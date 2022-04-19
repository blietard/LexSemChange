import scipy
import numpy as np

def standardize(arr):
    return ( arr-arr.mean(0)  )/(arr.std(0) )

def OrthogProcrustAlign(arr1,arr2, standard=False, backward=False):
    '''
    Return Orthogonal Procrustes alignment matrix of arr1 and arr2.
    `standard` set to True if arr1 and arr2 are already standardized. Default is False.
    '''
    if standard:
        A = arr1
        B = arr2
    else:
        A = standardize(arr1)
        B = standardize(arr2)

    temp = B.T @ A
    U, e, Vt = np.linalg.svd(temp,)
    if backward:
        W = Vt.T @ U.T
    else:
        W = U @ Vt
    return W