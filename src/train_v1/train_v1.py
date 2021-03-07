from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost
import mlflow
import hydra
import pickle
import pprint
import warnings
from typing import List, Any, Dict  # Tuple
from omegaconf.dictconfig import DictConfig
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from src.train_v1.util.get_environment import get_datadir, is_gpu, get_exec_env, has_changes_to_commit, get_head_commit
warnings.filterwarnings("ignore")


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


def get_model(model_name: str, model_param: DictConfig) -> Any:
    if model_name == 'XGBClassifier':
        if is_gpu():  # check if you're utilizing gpu if present
            assert model_param.tree_method == 'gpu_hist'
        return xgb.XGBClassifier(**model_param)
    elif model_name == 'LGBMClassifier':
        return lgb.LGBMClassifier(**model_param)
    elif model_name == 'CatBoostClassifier':
        return catboost.CatBoostClassifier(**model_param)
    elif model_name == 'RandomForestClassifier2':
        return RandomForestClassifier2(**model_param)
    else:
        raise ValueError(f'Invalid model_name: {model_name}')


def train_full(
        train: pd.DataFrame,
        features: List[str],
        target: str,
        model_name: str,
        model_param: DictConfig,
        train_param: DictConfig,
        OUT_DIR: str
        ) -> None:

    print('Start training')

    X_train = train.loc[:, features].values
    y_train = train.loc[:, target].values
    model = get_model(model_name, model_param)
    model.fit(X_train, y_train, **train_param)

    file = f'{OUT_DIR}/model_0.pkl'
    pickle.dump(model, open(file, 'wb'))
    mlflow.log_artifact(file)
    print('End training')

    return None


def get_normalized_metricname(metricname: str) -> str:
    '''
    Function to absorve the difference in metric names
    '''
    metricname_aggs = {
        'auc': ['areaundercurve'],
        'f1': ['fmeasure', 'f1measure', 'f1value'],
        'acc': ['accuracy'],
        'logloss': ['binarylogloss']
    }
    metricname_candidates = [to_name for to_name in metricname_aggs.keys()]

    # normalize
    normalized_name = metricname.lower().replace('-', '').replace('_', '').replace(' ', '')

    # aggregation
    for to_name, from_names in metricname_aggs.items():
        if normalized_name in from_names:
            normalized_name = to_name

    if normalized_name in metricname_candidates:
        return normalized_name
    else:
        raise ValueError(f'Unexpected metricname: {metricname}')


def log_learning_curve(model_name: str, model: Any, fold=0):
    '''
    Function to log learning curve.
    For GBDT models, the schema of evals_result is uniform like below:
    evals_result = {
        'validation_0': {'logloss': ['0.604835', '0.531479']},
        'validation_1': {'logloss': ['0.41965', '0.17686']}
        }
    example key for output: fold0_valid0-logloss
    '''
    if model_name == 'XGBClassifier':
        evals_result = model.evals_result()
        for eval_idx in range(len(evals_result)):
            validation_X_raw = f'validation_{eval_idx}'  # this is the raw expression from model
            metricdict = evals_result[validation_X_raw]
            for metricname, scorelist in metricdict.items():  # this loops only once
                metricname = get_normalized_metricname(metricname)
                for i, score in enumerate(scorelist):
                    mlflow.log_metric(f'fold{fold}_valid{eval_idx}-{metricname}', score, i)

    elif model_name == 'LGBMClassifier':
        evals_result = model.evals_result_
        for eval_idx in range(len(evals_result)):
            validation_X_raw = 'training' if eval_idx == 0 else f'valid_{eval_idx}'  # this is the raw expression from model
            metricdict = evals_result[validation_X_raw]
            for metricname, scorelist in metricdict.items():  # this loops only once
                metricname = get_normalized_metricname(metricname)
                for i, score in enumerate(scorelist):
                    mlflow.log_metric(f'fold{fold}_valid{eval_idx}-{metricname}', score, i)

    elif model_name == 'CatBoostClassifier':
        evals_result = model.get_evals_result()
        for eval_idx in range(len(evals_result)-1):  # skip key 'learn', which contains same value as validation_0
            validation_X_raw = f'validation_{eval_idx}'
            metricdict = evals_result[validation_X_raw]
            for metricname, scorelist in metricdict.items():  # this loops only once
                metricname = get_normalized_metricname(metricname)
                for i, score in enumerate(scorelist):
                    mlflow.log_metric(f'fold{fold}_valid{eval_idx}-{metricname}', score, i)

    elif model_name == 'RandomForestClassifier2':
        evals_result = model.get_evals_result()
        for eval_idx in range(len(evals_result)):
            validation_X_raw = f'valid{eval_idx}'
            metricdict = evals_result[validation_X_raw]
            for metricname, scorelist in metricdict.items():  # this loops only once
                metricname = get_normalized_metricname(metricname)
                for i, score in enumerate(scorelist):
                    mlflow.log_metric(f'fold{fold}_valid{eval_idx}-{metricname}', score, i)
    else:
        raise ValueError(f'Invalid model_name: {model_name}')


def train_KFold(
        train: pd.DataFrame,
        features: List[str],
        target: str,
        model_name: str,
        model_param: DictConfig,
        train_param: DictConfig,
        cv_param: DictConfig,
        OUT_DIR: str
        ) -> Dict:
    '''
    1. Create model
    2. Split training data into folds
    3. Train model
    4. Calculate validation metrics
    5. Calculate average validation metrics
    '''
    # store scores in schema: {'tr_acc': {'fold': [0.1, 0.8, ...], 'avg': 0.005}, ...}
    metrics = ['tr_acc', 'val_acc', 'tr_auc', 'val_auc']
    scores: dict = {}
    for metric in metrics:
        scores[metric] = {'fold': [], 'avg': None}

    kf = KFold(**cv_param)
    for fold, (tr, te) in enumerate(kf.split(train[target].values, train[target].values)):
        print(f'Starting fold: {fold}, train size: {len(tr)}, validation size: {len(te)}')
        X_tr, X_val = train.loc[tr, features].values, train.loc[te, features].values
        y_tr, y_val = train.loc[tr, target].values, train.loc[te, target].values
        model = get_model(model_name, model_param)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_tr, y_tr), (X_val, y_val)],
                  **train_param)

        pred_tr, pred_val = model.predict(X_tr), model.predict(X_val)

        # log learning curve
        log_learning_curve(model_name, model, fold)

        # log summarized metrics for this fold
        score = {
            metrics[0]: accuracy_score(y_tr, pred_tr),
            metrics[1]: accuracy_score(y_val, pred_val),
            metrics[2]: roc_auc_score(y_tr, pred_tr),
            metrics[3]: roc_auc_score(y_val, pred_val)
            }
        for metric in metrics:
            scores[metric]['fold'].append(score[metric])
            mlflow.log_metric(f'fold_{metric}', score[metric], step=fold)

        # log model
        file = f'{OUT_DIR}/model_{fold}.pkl'
        pickle.dump(model, open(file, 'wb'))
        mlflow.log_artifact(file)

    # calculate fold-average scores
    for metric in metrics:
        scores[metric]['avg'] = np.array(scores[metric]['fold']).mean()
        mlflow.log_metric(f'avg_{metric}', scores[metric]['avg'])

    return scores


def predict(
        models: List[Any],
        test: pd.DataFrame,
        feature_col: List[str],
        target_col: str) -> pd.DataFrame:

    print('Start predicting')
    y_pred = np.zeros(len(test))
    for model in models:
        y_pred += model.predict(test[feature_col].values) / len(models)
    pred_df = pd.DataFrame(data={'PassengerId': test['PassengerId'].values, 'Survived': y_pred})

    print('End predicting')
    return pred_df


@hydra.main(config_path="./config", config_name="config")
def main(cfg: DictConfig) -> None:
    commit = get_head_commit()
    # Check for changes not commited
    if get_exec_env() == 'local':
        if cfg.experiment.tags.exec == 'prd' and has_changes_to_commit():  # check for changes not commited
            raise Exception(f'Changes must be commited before running production!')

    pprint.pprint(dict(cfg))
    DATA_DIR = get_datadir()
    OUT_DIR = f'{DATA_DIR}/{cfg.experiment.name}/{cfg.experiment.tags.exec}{cfg.runno}'
    Path(OUT_DIR).mkdir(exist_ok=True, parents=True)

    # follow these sequences: uri > experiment > run > others
    tracking_uri = 'http://mlflow-tracking-server:5000'
    mlflow.set_tracking_uri(tracking_uri)  # uri must be set before set_experiment. artifact_uri is defined at tracking server
    mlflow.set_experiment(cfg.experiment.name)
    mlflow.start_run()
    mlflow.set_tags(cfg.experiment.tags)
    mlflow.set_tag('commit', commit) if commit is not None else print('No commit hash')
    if get_exec_env() == 'local':
        mlflow.log_artifacts('.hydra/')
    else:
        print('Note: configuration yaml is not logged in ipykernel environment')

    mlflow.set_tag('cv', cfg.cv.name)
    mlflow.set_tag('model', cfg.model.name)

    mlflow.log_param('feature_engineering', cfg.feature_engineering)
    mlflow.log_param('feature.name', [f.name for f in cfg.features])
    mlflow.log_params(cfg.cv.param)
    mlflow.log_params(cfg.model.model_param)
    mlflow.log_params(cfg.model.train_param)

    # FE
    train = pd.DataFrame()

    # load feature, info
    features = []
    for f in cfg.features:
        df = pd.read_pickle(f'{DATA_DIR}/{f.path}').loc[:, f.cols]
        train = pd.concat([train, df], axis=1)
        features += f.cols
        print(f'Feature: {f.name}, shape: {df.shape}')

    # load info
    if cfg.info.path is not None:
        df = pd.read_pickle(f'{DATA_DIR}/{cfg.info.path}').loc[:, cfg.info.cols]
        train = pd.concat([train, df], axis=1)

    # load target
    df = pd.read_pickle(f'{DATA_DIR}/{cfg.target.path}').loc[:, cfg.target.col]
    train = pd.concat([train, df], axis=1)

    print(f'Input feature shape: {train.shape}')

    # Feature engineering
    # Fill missing values
    if cfg.feature_engineering.method_fillna == '-999':
        train.loc[:, features] = train.loc[:, features].fillna(-999)
    elif cfg.feature_engineering.method_fillna == 'forward':
        train.loc[:, features] = train.loc[:, features].fillna(method='ffill').fillna(0)
    elif cfg.feature_engineering.method_fillna is None:
        pass
    else:
        raise ValueError(f'Invalid method_fillna: {cfg.feature_engineering.method_fillna}')

    # Train
    if cfg.option.train:
        if cfg.cv.name == 'nocv':
            train_full(train, features, cfg.target.col, cfg.model.name, cfg.model.model_param, cfg.model.train_param, OUT_DIR)
        elif cfg.cv.name == 'KFold':
            train_KFold(train, features, cfg.target.col, cfg.model.name, cfg.model.model_param,
                        cfg.model.train_param, cfg.cv.param, OUT_DIR)
        else:
            raise ValueError(f'Invalid cv: {cfg.cv.name}')

    # Predict
    if cfg.option.predict:
        models = []
        for i in range(cfg.cv.param.n_splits):
            model = pd.read_pickle(open(f'{OUT_DIR}/model_{i}.pkl', 'rb'))
            models.append(model)

        test = pd.read_pickle(f'{DATA_DIR}/{cfg.test.path}')
        # Fill missing values
        if cfg.feature_engineering.method_fillna == '-999':
            test.loc[:, features] = test.loc[:, features].fillna(-999)
        elif cfg.feature_engineering.method_fillna == 'forward':
            test.loc[:, features] = test.loc[:, features].fillna(method='ffill').fillna(0)
        elif cfg.feature_engineering.method_fillna is None:
            pass
        else:
            raise ValueError(f'Invalid method_fillna: {cfg.feature_engineering.method_fillna}')

        sample_submission = pd.read_csv(f'{DATA_DIR}/raw/gender_submission.csv')

        pred_df = predict(models, test, features, cfg.target.col)
        if not pred_df.shape == sample_submission.shape:
            raise Exception(f'Incorrect pred_df.shape: {pred_df.shape}')
        pred_df.to_csv(f'{OUT_DIR}/submission.csv')

    return None


if __name__ == '__main__':
    main()
