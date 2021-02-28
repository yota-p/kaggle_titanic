# https://www.kaggle.com/gogo827jz/jane-street-ffill-transformer-baseline
# https://www.kaggle.com/gogo827jz/optimise-speed-of-filling-nan-function
import numpy as np
from numba import njit


@njit
def fast_fillna(array, values):
    if np.isnan(array.sum()):
        array = np.where(np.isnan(array), values, array)
    return array

# The first run of numba decorated function requires compiling, which takes longer time than the later runs.
# So, we compile it before submission.
# train.loc[0, features[1:]] = fast_fillna(train.loc[0, features[1:]].values, 0)
