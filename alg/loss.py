import numpy as np


def loss(arr1, arr2, col_id):
    d1 = distance_importance(arr1[col_id], arr2[col_id]) 
    d2 = distance_ranking((-arr1).argsort().argsort(), (-arr2).argsort().argsort())
    return d1, d2

def loss_pop(arr1, arr2, col_id):
    d1 = distance_importance_pop(arr1[col_id], arr2[:, col_id])
    d2 = distance_ranking_pop((-arr1).argsort().argsort(), (-arr2).argsort(axis=1).argsort(axis=1))
    return d1, d2


def distance_importance(arr1, arr2):
    return np.absolute(arr1 - arr2).sum() / arr1.sum()

def distance_importance_pop(arr1, arr2):
    arr0 = np.tile(arr1, (arr2.shape[0], 1))
    return np.absolute(arr0 - arr2).sum(axis=1) / arr1.sum()


def distance_ranking(arr1, arr2):
    n = arr1.shape[0]
    arr0 = np.arange(n)
    w = (1/2)**arr0
    s = (w * np.absolute(arr0 - arr2[arr1.argsort()])).sum()
    m = (w * np.absolute(arr0 - arr0[::-1])).sum() # heuristic max
    return s / m

def distance_ranking_pop(arr1, arr2):
    n = arr1.shape[0]
    arr0 = np.tile(np.arange(n), (arr2.shape[0], 1))
    w = (1/2)**arr0
    s = (w * np.absolute(arr0 - arr2[:, arr1.argsort()])).sum(axis=1)
    m = (w * np.absolute(arr0 - arr0[:, ::-1])).sum(axis=1) # heuristic max
    return s / m