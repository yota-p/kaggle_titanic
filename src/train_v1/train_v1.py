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
from typing import List, Any  # Tuple
from omegaconf.dictconfig import DictConfig
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from src.train_v1.util.get_environment import get_datadir, is_gpu, is_ipykernel
warnings.filterwarnings("ignore")


def get_model(model_name: str, model_param: DictConfig) -> Any:
    if model_name == 'XGBClassifier':
        if is_gpu():  # check if you're utilizing gpu if present
            assert model_param.tree_method == 'gpu_hist'
        return xgb.XGBClassifier(**model_param)
    elif model_name == 'LGBMClassifier':
        return lgb.LGBMClassifier(**model_param)
    elif model_name == 'CatBoostClassifier':
        return catboost.CatBoostClassifier(**model_param)
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


def train_KFold(
        train: pd.DataFrame,
        features: List[str],
        target: str,
        model_name: str,
        model_param: DictConfig,
        train_param: DictConfig,
        cv_param: DictConfig,
        OUT_DIR: str
        ) -> None:

    kf = KFold(**cv_param)
    scores = []
    for fold, (tr, te) in enumerate(kf.split(train[target].values, train[target].values)):
        print(f'Starting fold: {fold}, train size: {len(tr)}, validation size: {len(te)}')
        X_tr, X_val = train.loc[tr, features].values, train.loc[te, features].values
        y_tr, y_val = train.loc[tr, target].values, train.loc[te, target].values
        model = get_model(model_name, model_param)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_tr, y_tr), (X_val, y_val)],
                  **train_param)

        pred_tr, pred_val = model.predict(X_tr), model.predict(X_val)
        tr_acc = accuracy_score(y_tr, pred_tr)
        val_acc = accuracy_score(y_val, pred_val)

        score = {'fold': fold, 'tr_acc': tr_acc, 'val_acc': val_acc}

        mlflow.log_metrics(score)
        scores.append(score)
        pprint.pprint(score)

        file = f'{OUT_DIR}/model_{fold}.pkl'
        pickle.dump(model, open(file, 'wb'))
        mlflow.log_artifact(file)
        del model, X_tr, X_val, y_tr, y_val

    ave_tr_acc, ave_val_acc = 0.0, 0.0
    for score in scores:
        ave_tr_acc += score['tr_acc'] / len(scores)
        ave_val_acc += score['val_acc'] / len(scores)

    print(f'ave_tr_acc: {ave_tr_acc}, ave_val_acc: {ave_val_acc}')

    return None


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
    pprint.pprint(dict(cfg))
    DATA_DIR = get_datadir()
    OUT_DIR = f'{DATA_DIR}/{cfg.experiment.name}/{cfg.experiment.tags.exec}{cfg.runno}'
    Path(OUT_DIR).mkdir(exist_ok=True, parents=True)

    # follow these sequences: uri > experiment > run > others
    tracking_uri = f'{DATA_DIR}/mlruns'
    mlflow.set_tracking_uri(tracking_uri)  # uri must be set before set_experiment
    mlflow.set_experiment(cfg.experiment.name)
    mlflow.start_run()
    mlflow.set_tags(cfg.experiment.tags)
    if not is_ipykernel():
        mlflow.log_artifacts('.hydra/')

    mlflow.log_param('feature_engineering', cfg.feature_engineering)
    mlflow.log_param('model.name', cfg.model.name)
    mlflow.log_params(cfg.model.model_param)
    mlflow.log_params(cfg.model.train_param)
    mlflow.log_param('cv.name', cfg.cv.name)
    mlflow.log_param('feature', cfg.features)

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
        sample_submission = pd.read_csv(f'{DATA_DIR}/raw/gender_submission.csv')

        pred_df = predict(models, test, features, cfg.target.col)
        if not pred_df.shape == sample_submission.shape:
            raise Exception(f'Incorrect pred_df.shape: {pred_df.shape}')
        pred_df.to_csv(f'{OUT_DIR}/submission.csv')

    return None


if __name__ == '__main__':
    main()
