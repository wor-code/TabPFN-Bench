import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
import torch
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcEnrichment, CalcRIE
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVR, SVC, SVR
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.post_hoc_ensembles.pfn_phe import PresetType, TaskType
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
    AutoPostHocEnsemblePredictor,
    AutoTabPFNClassifier,
    AutoTabPFNRegressor,
)
from xgboost import XGBClassifier, XGBRegressor

device_num = 0


def run_hyperopt(
        dataset_name,
        model_method,
        X, Y,
        smiles_list,
        model_type,
        test_size=0.2,
        fine_tune_model_path=None,
        hyper_model_path="result",
        mol_rep="moe",
):
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
        X,
        Y,
        smiles_list,
        test_size=test_size,
        random_state=42,
    )
    best_model = hyper_model(
        dataset_name,
        X_train, y_train, model_type, model_method,
        fine_tune_model_path,
        hyper_model_path,
        mol_rep,
    )
    return best_model


def tabpfn_phe(model_type):
    if model_type in ['CLS', 'biophysics']:
        model = TabPFNClassifier(
            device=f"cuda:{device_num}",
        )
    else:
        model = AutoPostHocEnsemblePredictor(
            preset=PresetType.DEFAULT,  # 选择默认参数
            task_type=TaskType.REGRESSION,  # 任务类型为回归
            ges_scoring_string="mse",  # 评估指标使用均方误差（MSE）
            device=f"cuda:{device_num}",  # 使用 GPU 加速计算（如果有 GPU）
            bm_random_state=42,  # 随机种子
            ges_random_state=42,  # 随机种子
            max_time=60,  # 训练时间最大 60 秒
            validation_method="cv",  # 使用交叉验证
            n_folds=5,  # 5 折交叉验证
            ges_n_iterations=50  # 进行 50 次贪心搜索迭代
        )
    return model

    pass


def tabpfn_auto(model_type):
    device_num = 0
    max_time = 30
    if model_type in ['CLS', 'biophysics']:
        model = AutoTabPFNClassifier(
            device=f"cuda:{device_num}",
            max_time=max_time,  # 120 seconds tuning time
        )
    else:
        model = AutoTabPFNRegressor(
            device=f"cuda:{device_num}",
            max_time=max_time,  # 120 seconds tuning time
        )
    return model


def tabpfn(model_type):
    if model_type in ['CLS', 'biophysics']:
        model = TabPFNClassifier(
            device=f"cuda:{device_num}",
            ignore_pretraining_limits=True,
        )
    else:
        model = TabPFNRegressor(
            ignore_pretraining_limits=True,
            device=f"cuda:{device_num}",
        )
    return model


def svm(
        model_type,
        best_params_dict=None,
        normal_flag=False,
):
    if model_type in ['CLS', 'biophysics']:
        base_model = SVC(
            probability=True,
            random_state=42
        )
    else:
        base_model = SVR()
        base_model = LinearSVR()

    if best_params_dict:
        # 判断参数是否是 pipeline 格式（带 svm__ 前缀）
        if any(k.startswith("svm__") for k in best_params_dict):
            stripped_params = {k.replace("svm__", ""): v for k, v in best_params_dict.items()}
        else:
            stripped_params = best_params_dict
    else:
        stripped_params = {}

    # 设置参数到 base model
    base_model.set_params(**stripped_params)

    if normal_flag:
        model = Pipeline([
            ('scaler', MinMaxScaler()),
            ('svm', base_model)
        ])
    else:
        model = base_model

    return model


def random_forest(model_type, best_params_dict=None):
    if model_type in ['CLS', 'biophysics']:
        if best_params_dict:
            model = RandomForestClassifier(**best_params_dict)
        else:
            model = RandomForestClassifier(n_estimators=500, random_state=42)
    else:
        if best_params_dict:
            model = RandomForestRegressor(**best_params_dict)
        else:
            model = RandomForestRegressor(n_estimators=500, random_state=42)
    return model


def xgboost(model_type, best_params_dict=None):
    if model_type in ['CLS', 'biophysics']:
        if best_params_dict:
            model = XGBClassifier(
                device='cpu',
                **best_params_dict
            )
        else:
            model = XGBClassifier(
                random_state=42,
                tree_method='hist'
            )
    else:
        if best_params_dict:
            model = XGBRegressor(
                device='cpu',
                **best_params_dict
            )
        else:
            model = XGBRegressor(random_state=42)
    return model


def hyper_rf(X, y, hyper_path, model_type):
    if model_type in ['CLS', 'biophysics']:
        model = RandomForestClassifier(
            n_jobs=-1,
            random_state=42
        )
        scoring = 'accuracy'
    else:
        model = RandomForestRegressor(
            n_jobs=-1,
            random_state=42
        )
        scoring = 'neg_mean_squared_error'

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    grid_search = GridSearchCV(
        model, param_grid,
        cv=3, n_jobs=-1,
        scoring=scoring
    )

    start_time = datetime.now()
    grid_search.fit(X, y)
    best_rf_parmas = grid_search.best_params_

    # save best_params_dict
    with open(hyper_path, "w") as f:
        json.dump(best_rf_parmas, f)
    return best_rf_parmas


def hyper_svm(
        X, y, hyper_path,
        model_type,
        normal_flag=False,
):
    start_time = datetime.now()
    param_grid = {
        'C': [0.1, 1, 10, 100],  # 正则化参数
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # 核函数
        # 'kernel': ['linear', 'rbf', 'sigmoid'],  # 核函数
        'gamma': ['scale', 'auto'],  # 核函数参数
        'degree': [2, 3, 4]  # 仅用于 poly 核
    }

    if model_type in ['CLS', 'biophysics']:
        model = SVC(
            random_state=42,
            probability=True,
        )
        scoring = make_scorer(accuracy_score)  # 分类任务使用准确率
    else:
        model = SVR()
        scoring = make_scorer(r2_score)  # 回归任务使用 R² 评分

    if normal_flag:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', model)
        ])
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],  # 正则化参数
            'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # 核函数
            'svm__gamma': ['scale', 'auto'],  # 核函数参数
            'svm__degree': [2, 3, 4]  # 仅用于 poly 核
        }
    else:
        param_grid = {
            'C': [0.1, 1, 10, 100],  # 正则化参数
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # 核函数
            'gamma': ['scale', 'auto'],  # 核函数参数
            'degree': [2, 3, 4]  # 仅用于 poly 核
        }

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring=scoring,
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X, y)

    best_svm_params = grid_search.best_params_

    # 保存最优参数
    with open(hyper_path, "w") as f:
        json.dump(best_svm_params, f)

    end_time = datetime.now()
    return best_svm_params


def hyper_xgb(X, y, hyper_path, model_type):
    param_grid = {
        'n_estimators': [100, 200, 300],  # 树的数量
        'learning_rate': [0.01, 0.05, 0.1],  # 学习率
        'max_depth': [3, 5, 7],  # 树的深度
        'subsample': [0.7, 0.8, 0.9],  # 采样率
        'min_child_weight': [1, 3, 5],  # 叶子节点最小权重
        'gamma': [0, 0.1, 0.2],  # 剪枝参数
        'colsample_bytree': [0.7, 0.8, 0.9],  # 列采样
    }
    start_time = datetime.now()

    if model_type in ['CLS', 'biophysics']:
        model = XGBClassifier(
            objective='binary:logistic' if len(np.unique(y)) == 2 else 'multi:softmax',
            n_jobs=-1,
            random_state=42,
            tree_method='hist',
            device='cpu',
        )
        scoring = make_scorer(accuracy_score)  # 使用 R2 评分
    else:
        model = XGBRegressor(
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=42,
            tree_method='hist',
            device='cpu',
        )
        scoring = make_scorer(r2_score)  # 使用准确率评分

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring=scoring,
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X, y)

    # save best_params_dict
    # create hyper_path parent_dir if not exists
    hyper_dir = os.path.dirname(hyper_path)
    if not os.path.exists(hyper_dir):
        os.makedirs(hyper_dir)

    with open(hyper_path, "w") as f:
        json.dump(grid_search.best_params_, f)
    return grid_search.best_params_


def hyper_model(
        dataset_name,
        X, y,
        model_type,
        model_method,
        fine_tune_model_path=None,
        hyper_model_path="result",
        mol_rep="rdkit2D",
):
    hyper_path = f"{hyper_model_path}/{model_type}/{dataset_name}/{model_method}.params"
    if model_type == "qm" and model_method == "SVM":
        hyper_path = f"{hyper_model_path}/{model_type}/{dataset_name}/{model_method}_new.params"
    print(f"----------- hyper_path: {hyper_path}")

    try:
        with open(hyper_path, "r") as f:
            best_params_dict = json.load(f)
    except FileNotFoundError:
        best_params_dict = None

    if model_method == "XGBoost":
        best_params_dict = hyper_xgb(X, y, hyper_path, model_type) if best_params_dict is None else best_params_dict
        model = xgboost(model_type, best_params_dict)
    elif model_method == "TabPFN":
        model = tabpfn(model_type)
    elif model_method == "TabPFN_PHE":
        model = tabpfn_phe(model_type)
    elif model_method == "TabPFN_Auto":
        model = tabpfn_auto(model_type)
    elif model_method == "RF":
        best_params_dict = hyper_rf(X, y, hyper_path, model_type) if best_params_dict is None else best_params_dict
        model = random_forest(model_type, best_params_dict)
    elif model_method == "SVM":
        if mol_rep == "moe" or (model_type == "qm" and model_method == "svm"):
            normal_flag = True
        else:
            normal_flag = False

        best_params_dict = hyper_svm(
            X, y,
            hyper_path, model_type,
            normal_flag
        ) if best_params_dict is None else best_params_dict

        model = svm(model_type, best_params_dict, normal_flag)
    else:
        model = None
        raise ValueError(f"Invalid model_method: {model_method}")
    return model


def metric_calc(labels, preds, metric='RMSE', proba_threshold='optimal'):
    """A function for metrics calculation"""
    # convert the labels and preds to list
    labels, preds = list(labels), list(preds)

    # get proba_cutoff for classification metrics calculation
    if metric not in ['RMSE', 'MAE', 'R2', 'Pearson_R', 'MAPE']:
        if proba_threshold == 'default':
            proba_cutoff = 0.5
        elif proba_threshold == 'optimal':
            # get the roc curve points
            false_pos_rate, true_pos_rate, proba = metrics.roc_curve(labels, preds)
            # calculate the optimal probability cutoff using Youden's J statistic with equal weight to FP and FN
            proba_cutoff = \
                sorted(list(zip(np.abs(true_pos_rate - false_pos_rate), proba)), key=lambda i: i[0], reverse=True)[0][1]
            # get hard_preds
            hard_preds = [1 if p > proba_cutoff else 0 for p in preds]

    ### for REG tasks
    if metric == 'RMSE':
        score = metrics.mean_squared_error(labels, preds)
        # an alternative way
    #         score = np.sqrt(np.nanmean((np.array(labels)-np.array(preds))**2))
    elif metric == 'MAE':
        score = metrics.mean_absolute_error(labels, preds)
    elif metric == 'R2':
        score = metrics.r2_score(labels, preds)
    elif metric == 'Pearson_R':
        score = scipy.stats.pearsonr(labels, preds)[0]
        # an alternative way
    #         score = np.corrcoef(labels, preds)[0,1]**2
    elif metric == 'MAPE':
        score = metrics.mean_absolute_error(labels, preds)
    ### for CLS tasks
    elif metric == 'AUROC':
        try:
            score = metrics.roc_auc_score(labels, preds)
        except ValueError:
            score = -1
    elif metric == 'AUPRC':
        # auc based on precision_recall curve
        precision, recall, _ = metrics.precision_recall_curve(labels, preds)
        score = metrics.auc(recall, precision)
        # an alternative way
    #         score = metrics.average_precision_score(labels, preds)
    elif metric == 'ACC':
        score = metrics.accuracy_score(labels, hard_preds)
    elif metric == 'Precision_PPV':
        # calculate precision based on hard_preds
        score = metrics.precision_score(labels, hard_preds, pos_label=1)
    elif metric == 'Precision_NPV':
        # calculate precision based on hard_preds
        score = metrics.precision_score(labels, hard_preds, pos_label=0)
    elif metric == 'MCC':
        # calculate precision based on hard_preds
        score = metrics.matthews_corrcoef(labels, hard_preds)
    elif metric == 'Cohen_Kappa':
        # calculate precision based on hard_preds
        score = metrics.cohen_kappa_score(labels, hard_preds)
    elif metric == 'BEDROC':
        # reference: deepchem - https://github.com/deepchem/deepchem/blob/master/deepchem/metrics/score_function.py#L132-L183
        scores = list(zip(labels, hard_preds))
        scores = sorted(scores, key=lambda pair: pair[1], reverse=True)
        # calculate BEDROC
        score = CalcBEDROC(scores, 0, alpha=20.0)  # alpha: 0-20
    elif metric == 'RIE':
        # reference: deepchem - https://github.com/deepchem/deepchem/blob/master/deepchem/metrics/score_function.py#L132-L183
        scores = list(zip(labels, hard_preds))
        scores = sorted(scores, key=lambda pair: pair[1], reverse=True)
        # calculate RIE
        score = CalcRIE(scores, 0, alpha=20.0)  # alpha: 0-20
    elif metric == 'EF':
        # reference: deepchem - https://github.com/deepchem/deepchem/blob/master/deepchem/metrics/score_function.py#L132-L183
        scores = list(zip(labels, hard_preds))
        scores = sorted(scores, key=lambda pair: pair[1], reverse=True)
        # calculate enrichment factor
        score = CalcEnrichment(scores, 0, fractions=[0.1])[0]
    return score


model_dict = {
    "RF": random_forest,
    "SVM": svm,
    "XGBoost": xgboost,
    "TabPFN": tabpfn,
    "TabPFN_Auto": tabpfn_auto,
}


def main():
    parmas = {
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_samples_leaf': 3,
        'min_samples_split': 2,
        'n_estimators': 100,
        'subsample': 0.8
    }
    file_path = f"result/REG/Bioconcentration Factor/XGBoost.params"
    with open(file_path, "w") as f:
        json.dump(parmas, f)
    pass


if __name__ == '__main__':
    main()
