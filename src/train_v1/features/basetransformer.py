from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        raise NotImplementedError()

    def transform(self, X):
        raise NotImplementedError()
