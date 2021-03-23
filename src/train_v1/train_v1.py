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
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from src.train_v1.models.RandomForestClassifier2 import RandomForestClassifier2
from src.train_v1.models.torchnn import ModelV1, LitModelV1
from src.train_v1.util.get_environment import get_datadir, is_gpu, get_exec_env, has_changes_to_commit, get_head_commit, get_device
from src.train_v1.util.seeder import seed_everything
from src.train_v1.features.basetransformer import BaseTransformer
warnings.filterwarnings("ignore")


class NaFiller(BaseTransformer):
    def __init__(self, method: str, feat_cols: List[str]) -> None:
        self.method_ = method
        self.feat_cols = feat_cols
        self.mean_: pd.DataFrame = None

    def fit(self, X):
        if self.method_ == 'mean':
            self.mean_ = X[self.feat_cols].mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.method_ == '-999':
            X.fillna(-999, inplace=True)
        elif self.method_ == 'mean':
            X.fillna(self.mean_, inplace=True)
        elif self.method_ == 'forward':
            X.fillna(method='ffill').fillna(0, inplace=True)
        elif self.method is None:
            pass
        else:
            raise ValueError(f'Invalid method: {self.method}')
        return X


def get_model(
        model_name: str,
        model_param: DictConfig,
        optimizercfg: DictConfig = None,
        schedulercfg: DictConfig = None,
        lossfncfg: DictConfig = None,
        feat_cols: List[str] = None,
        target_cols: List[str] = None,
        fold: int = 0,
        device: torch.device = None) -> Any:
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
    elif model_name == 'torch_v1':
        return ModelV1(
            feat_cols,
            target_cols,
            model_param.dropout_rate,
            model_param.hidden_size
            ).to_device()
    elif model_name == 'LitModelV1':
        return LitModelV1(
            feat_cols,
            target_cols,
            model_param.dropout_rate,
            model_param.hidden_size,
            optimizercfg,
            schedulercfg,
            lossfncfg,
            fold
            )
    else:
        raise ValueError(f'Invalid model_name: {model_name}')


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    avg_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        features = data['features'].to(device)
        label = data['label'].to(device)
        outputs = model(features)
        loss = loss_fn(outputs, label)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        avg_loss += loss.item() / len(dataloader)

    return avg_loss


def inference_fn(model, dataloader, device, target_cols):
    model.eval()
    preds = []

    for data in dataloader:
        features = data['features'].to(device)

        with torch.no_grad():
            outputs = model(features)

        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds).reshape(-1, len(target_cols))

    return preds


class TitanicDataset(Dataset):
    def __init__(self, df, feat_cols: List[str], target_cols: List[str]):
        self.features = df[feat_cols].values
        self.label = df[target_cols].values.reshape(-1, len(target_cols))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'label': torch.tensor(self.label[idx], dtype=torch.float)
        }


def train_torch_KFold(
        train_df: pd.DataFrame,
        feat_cols: List[str],
        target_cols: List[str],
        model_name: str,
        model_param: DictConfig,
        train_param: DictConfig,
        cv_param: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        loss_function: DictConfig,
        OUT_DIR: str
        ) -> None:
    '''
    1. Create model
    2. Split training data into folds
    3. Train model
    4. Calculate validation metrics
    5. Calculate average validation metrics
    '''
    # store scores in schema: {'train-acc': {'fold': [0.1, 0.8, ...], 'avg': 0.005}, ...}
    metrics = ['train-acc', 'valid-acc', 'train-auc', 'valid-auc']
    scores: dict = {}
    for metric in metrics:
        scores[metric] = {'vsfold': [], 'avg': None}

    device = get_device()
    target = target_cols[0]

    kf = KFold(**cv_param)
    for fold, (tr, te) in enumerate(kf.split(train_df[target].values, train_df[target].values)):
        print(f'Starting fold: {fold}, train size: {len(tr)}, validation size: {len(te)}')

        # split data
        train = train_df.loc[tr, :]
        valid = train_df.loc[te, :]
        y_tr = train_df.loc[tr, target]
        y_val = train_df.loc[te, target]

        # create dataset
        train_set = TitanicDataset(train, feat_cols, target_cols)
        train_loader = DataLoader(train_set, batch_size=train_param.batch_size, shuffle=True, num_workers=4)
        valid_set = TitanicDataset(valid, feat_cols, target_cols)
        valid_loader = DataLoader(valid_set, batch_size=train_param.batch_size, shuffle=False, num_workers=4)

        torch.cuda.empty_cache()
        model = get_model(
                    model_name,
                    model_param,
                    optimizer,
                    scheduler,
                    loss_function,
                    feat_cols=feat_cols,
                    target_cols=target_cols,
                    fold=fold,
                    device=device)

        early_stop_callback = EarlyStopping(
            'val-loss_vsepoch',
            patience=train_param.early_stopping_rounds,
            mode='max'
            )

        trainer = pl.Trainer(
            max_epochs=train_param.epochs,
            fast_dev_run=False,  # TODO: pass from param!
            callbacks=[early_stop_callback],
            )
        trainer.fit(model, train_loader, valid_loader)

        # log model
        torch.save(model.state_dict(), f'{OUT_DIR}/model_{fold}.pth')

        # log metrics per fold
        pred_tr = inference_fn(model, train_loader, device, target_cols)
        pred_val = inference_fn(model, valid_loader, device, target_cols)
        pred_tr = np.where(pred_tr >= 0.5, 1, 0).astype(int)
        pred_val = np.where(pred_val >= 0.5, 1, 0).astype(int)
        score = {
            metrics[0]: accuracy_score(y_tr, pred_tr),
            metrics[1]: accuracy_score(y_val, pred_val),
            metrics[2]: roc_auc_score(y_tr, pred_tr),
            metrics[3]: roc_auc_score(y_val, pred_val)
            }
        for metric in metrics:
            scores[metric]['vsfold'].append(score[metric])
            mlflow.log_metric(f'{metric}_vsfold', score[metric], step=fold)

    # log metrics averaged over folds
    for metric in metrics:
        scores[metric]['avg'] = np.array(scores[metric]['vsfold']).mean()
        mlflow.log_metric(f'{metric}_foldavg', scores[metric]['avg'])

    print('End training')
    return None


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
                    mlflow.log_metric(f'fold{fold}_valid{eval_idx}-{metricname}_vsstep', score, i)

    elif model_name == 'LGBMClassifier':
        evals_result = model.evals_result_
        for eval_idx in range(len(evals_result)):
            validation_X_raw = 'training' if eval_idx == 0 else f'valid_{eval_idx}'  # this is the raw expression from model
            metricdict = evals_result[validation_X_raw]
            for metricname, scorelist in metricdict.items():  # this loops only once
                metricname = get_normalized_metricname(metricname)
                for i, score in enumerate(scorelist):
                    mlflow.log_metric(f'fold{fold}_valid{eval_idx}-{metricname}_vsstep', score, i)

    elif model_name == 'CatBoostClassifier':
        evals_result = model.get_evals_result()
        for eval_idx in range(len(evals_result)-1):  # skip key 'learn', which contains same value as validation_0
            validation_X_raw = f'validation_{eval_idx}'
            metricdict = evals_result[validation_X_raw]
            for metricname, scorelist in metricdict.items():  # this loops only once
                metricname = get_normalized_metricname(metricname)
                for i, score in enumerate(scorelist):
                    mlflow.log_metric(f'fold{fold}_valid{eval_idx}-{metricname}_vsstep', score, i)

    elif model_name == 'RandomForestClassifier2':
        evals_result = model.get_evals_result()
        for eval_idx in range(len(evals_result)):
            validation_X_raw = f'valid{eval_idx}'
            metricdict = evals_result[validation_X_raw]
            for metricname, scorelist in metricdict.items():  # this loops only once
                metricname = get_normalized_metricname(metricname)
                for i, score in enumerate(scorelist):
                    mlflow.log_metric(f'fold{fold}_valid{eval_idx}-{metricname}_vsstep', score, i)
    else:
        raise ValueError(f'Invalid model_name: {model_name}')


def train_gbdt_KFold(
        train: pd.DataFrame,
        feat_cols: List[str],
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
    # store scores in schema: {'train-acc': {'fold': [0.1, 0.8, ...], 'avg': 0.005}, ...}
    metrics = ['train-acc', 'valid-acc', 'train-auc', 'valid-auc']
    scores: dict = {}
    for metric in metrics:
        scores[metric] = {'vsfold': [], 'avg': None}

    kf = KFold(**cv_param)
    for fold, (tr, te) in enumerate(kf.split(train[target].values, train[target].values)):
        print(f'Starting fold: {fold}, train size: {len(tr)}, validation size: {len(te)}')
        X_tr, X_val = train.loc[tr, feat_cols].values, train.loc[te, feat_cols].values
        y_tr, y_val = train.loc[tr, target].values, train.loc[te, target].values
        model = get_model(model_name, model_param)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_tr, y_tr), (X_val, y_val)],
                  **train_param)

        pred_tr, pred_val = model.predict(X_tr), model.predict(X_val)

        # log metrics per step
        log_learning_curve(model_name, model, fold)

        # log metrics per fold
        score = {
            metrics[0]: accuracy_score(y_tr, pred_tr),
            metrics[1]: accuracy_score(y_val, pred_val),
            metrics[2]: roc_auc_score(y_tr, pred_tr),
            metrics[3]: roc_auc_score(y_val, pred_val)
            }
        for metric in metrics:
            scores[metric]['vsfold'].append(score[metric])
            mlflow.log_metric(f'{metric}_vsfold', score[metric], step=fold)

        # log model
        pickle.dump(model, open(f'{OUT_DIR}/model_{fold}.pkl', 'wb'))

    # log metrics averaged over folds
    for metric in metrics:
        scores[metric]['avg'] = np.array(scores[metric]['vsfold']).mean()
        mlflow.log_metric(f'{metric}_foldavg', scores[metric]['avg'])

    return scores


@hydra.main(config_path="./config", config_name="config")
def main(cfg: DictConfig) -> None:
    pprint.pprint(dict(cfg))

    # set random seed
    seed_everything(**cfg.random_seed)

    commit = get_head_commit()
    # check for changes not commited
    if get_exec_env() == 'local':
        if cfg.experiment.tags.exec == 'prd' and has_changes_to_commit():  # check for changes not commited
            raise Exception(f'Changes must be commited before running production!')

    DATA_DIR = get_datadir()
    OUT_DIR = f'{DATA_DIR}/{cfg.experiment.name}/{cfg.experiment.tags.exec}{cfg.runno}'
    Path(OUT_DIR).mkdir(exist_ok=True, parents=True)

    device = get_device()

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

    train = pd.DataFrame()

    # load feature
    feat_cols = []
    for f in cfg.features:
        df = pd.read_pickle(f'{DATA_DIR}/{f.path}').loc[:, f.cols]
        train = pd.concat([train, df], axis=1)
        feat_cols += f.cols
        print(f'Feature: {f.name}, shape: {df.shape}')

    # load info
    if cfg.info.path is not None:
        df = pd.read_pickle(f'{DATA_DIR}/{cfg.info.path}').loc[:, cfg.info.cols]
        train = pd.concat([train, df], axis=1)

    # load target
    df = pd.read_pickle(f'{DATA_DIR}/{cfg.target.path}').loc[:, cfg.target.col]
    train = pd.concat([train, df], axis=1)

    print(f'Input feature shape: {train.shape}')

    # feature engineering
    nfl = NaFiller(cfg.feature_engineering.method_fillna, feat_cols)  # fill missing values
    pipe = [nfl]
    for p in pipe:
        train = p.fit_transform(train)

    if nfl.mean_ is not None:
        np.save(f'{OUT_DIR}/nafiller_mean.npy', nfl.mean_.values)

    # Train
    if cfg.option.train:
        if cfg.cv.name == 'nocv':
            train_full(
                train, feat_cols, cfg.target.col, cfg.model.name, cfg.model.model_param,
                cfg.model.train_param, OUT_DIR)
        elif cfg.cv.name == 'KFold':
            if cfg.model.type == 'pytorch-lightning':
                train_torch_KFold(
                    train, feat_cols, [cfg.target.col], cfg.model.name, cfg.model.model_param, cfg.model.train_param,
                    cfg.cv.param, cfg.optimizer, cfg.scheduler, cfg.loss_function, OUT_DIR)
            else:
                train_gbdt_KFold(
                    train, feat_cols, cfg.target.col, cfg.model.name, cfg.model.model_param, cfg.model.train_param,
                    cfg.cv.param, OUT_DIR)
        else:
            raise ValueError(f'Invalid cv: {cfg.cv.name}')

    # Predict
    if cfg.option.predict:
        print('Start predicting')
        # load data
        test = pd.read_pickle(f'{DATA_DIR}/{cfg.test.path}')
        sample_submission = pd.read_csv(f'{DATA_DIR}/raw/gender_submission.csv')
        y_pred = np.zeros(len(test))

        # feature engineering
        for p in pipe:
            test = p.transform(test)

        # load models
        models = []
        if cfg.model.type == 'pytorch-lightning':
            model_paths = [f'{OUT_DIR}/model_{i}.pth' for i in range(cfg.cv.param.n_splits)]
            for model_path in model_paths:
                torch.cuda.empty_cache()
                model = get_model(cfg.model.name, cfg.model.model_param, feat_cols=feat_cols, target_cols=[cfg.target.col], device=device)
                model.to(device)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model.eval()
                models.append(model)
        elif cfg.model.type == 'pytorch':
            raise NotImplementedError()
        elif cfg.model.type == 'sklearn':
            model_paths = [f'{OUT_DIR}/model_{i}.pkl' for i in range(cfg.cv.param.n_splits)]
            for model_path in model_paths:
                model = pd.read_pickle(open(model_path, 'rb'))
                models.append(model)
        else:
            raise ValueError(f'Invalid model.type: {cfg.model.type}')

        # ensemble models
        for model in models:
            if cfg.model.type in ['pytorch', 'pytorch-lightning']:
                # 1. create prediction as torch.tensor
                # 2. convert torch.tensor(418, 1) -> np.ndarray(418, 1) -> np.ndarray(418,)
                # 3. divide by len(model)
                y_pred += model(torch.tensor(test[feat_cols].values, dtype=torch.float).to(device)) \
                    .sigmoid().detach().cpu() \
                    .numpy()[:, 0] \
                    / len(models)
            elif cfg.model.type == 'sklearn':
                y_pred += model.predict(test[feat_cols].values) / len(models)
            else:
                raise ValueError(f'Invalid model.type: {cfg.model.type}')

        y_pred = np.where(y_pred >= 0.5, 1, 0).astype(int)
        pred_df = pd.DataFrame(data={'PassengerId': test['PassengerId'].values, 'Survived': y_pred})

        if not pred_df.shape == sample_submission.shape:
            raise Exception(f'Incorrect pred_df.shape: {pred_df.shape}')

        submission_path = f'{OUT_DIR}/submission.csv'
        pred_df.to_csv(submission_path, index=False)
        print('End predicting')

    mlflow.log_artifacts(OUT_DIR)
    return None


if __name__ == '__main__':
    main()
