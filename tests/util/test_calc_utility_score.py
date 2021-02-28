import pandas as pd
from src.util.calc_utility_score import utility_score_numba


class TestUtilityScoreNumba:
    def test_calc(self):
        train = pd.DataFrame({
                    'date': [0, 1, 2, 2],
                    'weight': [1, 2, 3, 4],
                    'resp': [10, 20, 30, 40],
                    'action': [0, 0, 0, 1]
                    })
        date = train['date'].values
        weight = train['weight'].values
        resp = train['resp'].values
        action = train['action'].values

        score_ac = utility_score_numba(date, weight, resp, action)
        # Pi = [  0.   0. 160.]
        # t = 9.128709291752768
        # u = 960.0
        score_ex = 960.0
        assert(score_ac == score_ex)
