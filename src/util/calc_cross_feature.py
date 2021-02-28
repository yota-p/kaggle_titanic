import numpy as np


def calc_cross_feature(x_tt: np.ndarray) -> np.ndarray:
    '''
    Function to caclulate cross feature.
    Input: x_tt: (1,N) ndarray. Column order should follow the dataframe created by janestreet api
    Output: y_tt: (1, N+k) ndarray. Right k rows contain calculated cross features.
    '''
    cross_41_42_43 = x_tt[:, 41] + x_tt[:, 42] + x_tt[:, 43]
    cross_1_2 = x_tt[:, 1] / (x_tt[:, 2] + 1e-5)
    y_tt = np.concatenate((
        x_tt,
        np.array(cross_41_42_43).reshape(x_tt.shape[0], 1),
        np.array(cross_1_2).reshape(x_tt.shape[0], 1),
    ), axis=1)
    return y_tt
