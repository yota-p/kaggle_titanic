from pathlib import Path
import time
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
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
from src.train_v1.models.torchnn import ModelV1, EarlyStopping
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


class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets: torch.Tensor, n_labels: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1), self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


class MarketDataset(Dataset):
    def __init__(self, df, all_feat_cols: List[str], target_cols: List[str]):
        self.features = df[all_feat_cols].values
        self.label = df[target_cols].values.reshape(-1, len(target_cols))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'label': torch.tensor(self.label[idx], dtype=torch.float)
        }


def get_optimizer(
        optimizer_name: str,
        param: DictConfig,
        model_param) -> torch.optim.Optimizer:
    if optimizer_name == 'Adam':
        return torch.optim.Adam(model_param, lr=param.lr, weight_decay=param.weight_decay)
        # optimizer = Nadam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # optimizer = Lookahead(optimizer=optimizer, k=10, alpha=0.5)
    else:
        raise ValueError(f'Invalid optimizer: {optimizer_name}')


def get_scheduler(
        scheduler_name: str,
        param: DictConfig,
        steps_per_epoch: int,
        optimizer: torch.optim.Optimizer) -> Any:
    if scheduler_name is None:
        return None
    elif scheduler_name == 'OneCycleLR':
        return OneCycleLR(
                    optimizer=param.optimizer,
                    pct_start=param.pct_start,
                    div_factor=param.dev_factor,
                    max_lr=param.max_lr,
                    epochs=param.epochs,
                    steps_per_epoch=steps_per_epoch)
    else:
        raise ValueError(f'Invalid scheduler: {scheduler_name}')


def get_loss_function(
        loss_function_name: str,
        param: DictConfig) -> torch.nn.modules.loss._Loss:  # _WeightedLoss or BCEWithLogitsLoss

    if loss_function_name == 'SmoothBCEwLogits':
        return SmoothBCEwLogits(smoothing=param.smoothing)
    elif loss_function_name == 'BCEWithLogitsLoss':
        return BCEWithLogitsLoss()
    else:
        raise ValueError(f'Invalid loss functin: {loss_function_name}')


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

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

        final_loss += loss.item()

    final_loss /= len(dataloader)

    return final_loss


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


def train_cv_nn(
        df: pd.DataFrame,
        features: List[str],
        target_cols: List[str],
        model_name: str,
        model_param: DictConfig,
        train_param: DictConfig,
        cv: DictConfig,
        model_save_paths: List[str],
        cfg
        ) -> None:

    # TODO: implement KFold CV split
    '''
    if cv.name == 'nocv':
        print('Training on full data. Note that validation data overlaps train, which will overfit!')
        train = df
        valid = df
    else:
        train = df
        valid = df
    '''
    train = df
    valid = df

    train_set = MarketDataset(train, features, target_cols)
    train_loader = DataLoader(train_set, batch_size=train_param.batch_size, shuffle=True, num_workers=4)
    valid_set = MarketDataset(valid, features, target_cols)
    valid_loader = DataLoader(valid_set, batch_size=train_param.batch_size, shuffle=False, num_workers=4)

    start_time = time.time()
    torch.cuda.empty_cache()
    device = get_device()
    model = get_model(
                model_name,
                model_param,
                feat_cols=features,
                target_cols=target_cols,
                device=device)

    optimizer = get_optimizer(
                    optimizer_name=cfg.optimizer.name,
                    param=cfg.optimizer.param,
                    model_param=model.parameters())

    scheduler = get_scheduler(
                    scheduler_name=cfg.scheduler.name,
                    param=cfg.scheduler.param,
                    steps_per_epoch=len(train_loader),
                    optimizer=optimizer)

    loss_fn = get_loss_function(
                loss_function_name=cfg.loss_function.name,
                param=cfg.loss_function.param)

    es = EarlyStopping(patience=train_param.early_stopping_rounds, mode='max')

    for epoch in range(train_param.epochs):
        train_loss = train_fn(model, optimizer, scheduler, loss_fn, train_loader, device)

        valid_pred = inference_fn(model, valid_loader, device, target_cols)
        valid_auc = roc_auc_score(valid[target_cols].values, valid_pred)
        # valid_logloss = log_loss(valid[target_cols].values, valid_pred)
        valid_pred = np.median(valid_pred, axis=1)
        valid_pred = np.where(valid_pred >= 0.5, 1, 0).astype(int)
        '''
        valid_u_score = utility_score_bincount(
                            date=valid.date.values, weight=valid.weight.values,
                            resp=valid.resp.values, action=valid_pred)
        score = {
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_u_score': valid_u_score,
            'valid_auc': valid_auc,
            'time': (time.time() - start_time) / 60}
        pprint.pprint(score)
        '''
        es(valid_auc, model, model_path=model_save_paths[0])
        if es.early_stop:
            print('Early stopping')
            break
    # torch.save(model.state_dict(), model_weights)

    return None


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


def get_model(
        model_name: str,
        model_param: DictConfig,
        feat_cols: List[str] = None,
        target_cols: List[str] = None,
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
        model = ModelV1(feat_cols, target_cols, model_param.dropout_rate, model_param.hidden_size)
        model.to(device)
        return model
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
    metrics = ['train-acc', 'valid-acc', 'train-auc', 'valid-auc']
    scores: dict = {}
    for metric in metrics:
        scores[metric] = {'vsfold': [], 'avg': None}

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
        file = f'{OUT_DIR}/model_{fold}.pkl'
        pickle.dump(model, open(file, 'wb'))
        mlflow.log_artifact(file)

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
            train_full(train, feat_cols, cfg.target.col, cfg.model.name, cfg.model.model_param, cfg.model.train_param, OUT_DIR)
        elif cfg.cv.name == 'KFold':
            # TODO: this is temporal implementation. Integrate nn and gbdt later!
            if cfg.model.name == 'torch_v1':
                model_paths = [f'{OUT_DIR}/model_0.pth']
                train_cv_nn(train, feat_cols, [cfg.target.col], cfg.model.name, cfg.model.model_param, cfg.model.train_param, cfg.cv.param, model_paths, cfg)
            else:
                train_KFold(train, feat_cols, cfg.target.col, cfg.model.name, cfg.model.model_param,
                            cfg.model.train_param, cfg.cv.param, OUT_DIR)
        else:
            raise ValueError(f'Invalid cv: {cfg.cv.name}')

    # Predict
    if cfg.option.predict:
        # load data
        test = pd.read_pickle(f'{DATA_DIR}/{cfg.test.path}')
        sample_submission = pd.read_csv(f'{DATA_DIR}/raw/gender_submission.csv')
        y_pred = np.zeros(len(test))

        # load model
        models = []
        if cfg.model.name == 'torch_v1':
            model_paths = [f'{OUT_DIR}/model_{i}.pth' for i in range(cfg.cv.param.n_splits)]
            for model_path in model_paths:
                torch.cuda.empty_cache()
                model = get_model(cfg.model.name, cfg.model.model_param, feat_cols=feat_cols, target_cols=[cfg.target.col], device=device)
                model.to(device)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model.eval()
                models.append(model)
        else:
            model_paths = [f'{OUT_DIR}/model_{i}.pkl' for i in range(cfg.cv.param.n_splits)]
            for model_path in model_paths:
                model = pd.read_pickle(open(model_path, 'rb'))
                models.append(model)

        # feature engineering
        for p in pipe:
            test = p.transform(test)

        print('Start predicting')
        for model in models:  # ensemble models
            if cfg.model.name == 'torch_v1':
                # 1. create prediction as torch.tensor
                # 2. convert torch.tensor(418, 1) -> np.ndarray(418, 1) -> np.ndarray(418,)
                # 3. divide by len(model)
                y_pred += model(torch.tensor(test[feat_cols].values, dtype=torch.float).to(device)) \
                    .sigmoid().detach().cpu() \
                    .numpy()[:, 0] \
                    / len(models)
            else:
                y_pred += model.predict(test[feat_cols].values) / len(models)

        y_pred = np.where(y_pred >= 0.5, 1, 0).astype(int)
        pred_df = pd.DataFrame(data={'PassengerId': test['PassengerId'].values, 'Survived': y_pred})

        if not pred_df.shape == sample_submission.shape:
            raise Exception(f'Incorrect pred_df.shape: {pred_df.shape}')

        pred_df.to_csv(f'{OUT_DIR}/submission.csv', index=False)
        print('End predicting')

    return None


if __name__ == '__main__':
    main()
