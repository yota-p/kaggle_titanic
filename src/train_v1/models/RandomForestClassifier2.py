from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss


class RandomForestClassifier2(RandomForestClassifier):
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def __getattr__(self, name):
        # this returns attributes that does not exist in this class
        return getattr(self.model, name)

    def get_evals_result(self):
        return self.evals_result_

    def fit(self, X, y, eval_set=None, eval_metrics=None, sample_weight=None):
        '''
        Fit Random forest with learning curve.
        eval_set = [(X_val1, y_val1), (X_val2, y_val2), ...]
        Note: Once after fitted, add attribute ends with a underscore.
        This is because sklearn.utils.validation.check_is_fitted()
        checks if model is fitted by this criterion.
        '''
        n_estimators = self.model.get_params()['n_estimators']
        to_validate = True if (eval_set is not None and eval_metrics is not None) else False

        # Validate by calculating metrics for various n_estimators
        if to_validate:
            # initialize evals_result
            self.evals_result_ = {}
            for i in range(0, len(eval_set)):
                # Example: evals_result = {'valid_0': {'logloss': []}, 'valid-1': {'AUC': []}}
                self.evals_result_.update({f'valid{i}': {f'{eval_metrics}': []}})

            # train through different n_estimators
            for i in range(1, n_estimators+1):
                msg = f'[{i}]'
                self.model.set_params(n_estimators=i)
                self.model.fit(X, y, sample_weight)
                for j, (X_val, y_val) in enumerate(eval_set):
                    pred_val = self.model.predict(X_val)
                    if eval_metrics == 'logloss':
                        metric = log_loss(y_val, pred_val)
                    else:
                        raise ValueError(f'Invalid eval_metrics: {eval_metrics}')
                    self.evals_result_[f'valid{j}'][f'{eval_metrics}'].append(metric)
                    msg = msg + f'\tvalid{j}-{eval_metrics}: {metric}'
                print(msg)
        else:
            self.model.fit(X, y, sample_weight)
            self.evals_result_ = None
