import numpy as np
from operator import itemgetter


def shuffle_indices(size,shuffle,random_seed = 0):
    indices = list(range(size))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    return indices


def get_list_items_with_indices(items_all, indices):
    getter = itemgetter(*indices)
    items = list(getter(items_all))
    return items

