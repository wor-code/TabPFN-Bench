import ast
import sys
sys.path.append('/public/home/tianyao/TabPFN_test/finetune_tabpfn_v2')
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from finetuning_scripts.finetune_tabpfn_main import fine_tune_tabpfn
from constant import *
from common import *

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

def safe_eval(x):
    try:
        if isinstance(x, (list, np.ndarray)):
            return np.array(x)
        if isinstance(x, str):
            cleaned_str = x.strip('[]').replace('\n', ' ')
            num_strings = [s for s in cleaned_str.split(' ') if s]
            return np.array([float(val) for val in num_strings])
        return x
    except Exception as e:
        print(f"Error parsing: {x}")
        print(f"Error details: {e}")
        return np.nan

def random_train_ablation(X, Y, smiles_list, ablation_ratio=0.1, random_seed=42):
    """在训练集上随机采样，消融ablation_ratio比例数据"""
    np.random.seed(random_seed)
    n_samples = X.shape[0]
    n_keep = int(n_samples * (1 - ablation_ratio))
    n_keep = max(n_keep, 1)
    indices = np.random.choice(n_samples, size=n_keep, replace=False)
    return X[indices], Y[indices], smiles_list.iloc[indices] if hasattr(smiles_list, "iloc") else smiles_list[indices]

def get_reg_metrics(y_true, y_pred, smiles_test, tag):
    result_df = pd.DataFrame({
        'preds': y_pred,
        'labels': y_true,
        'SMILES': smiles_test,
    })
    rmse = metric_calc(y_true, y_pred, 'RMSE')
    mae = metric_calc(y_true, y_pred, 'MAE')
    r2 = metric_calc(y_true, y_pred, 'R2')
    pearson_r = metric_calc(y_true, y_pred, 'Pearson_R')
    print(f"tag: {tag}, RMSE: {rmse}, MAE: {mae}, R2: {r2}, Pearson_R: {pearson_r}")
    return rmse, mae, r2, pearson_r, result_df

def get_performance(dataset_name, best_model, X, Y, random_seed, smiles_list, train_ablation_ratio=1.0, test_size=0.2):
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
        X, Y, smiles_list, test_size=test_size, random_state=random_seed
    )
    # 对train做消融
    X_train, y_train, smiles_train = random_train_ablation(X_train, y_train, smiles_train, train_ablation_ratio, random_seed)
    best_model.fit(X_train, y_train)
    Y_pred = best_model.predict(X_test)
    if model_type == "REG":
        test_rmse, test_mae, test_r2, test_pearson_r, _ = get_reg_metrics(y_test, Y_pred, smiles_test, 'test')
        return test_rmse, test_mae, test_r2, test_pearson_r
    else:
        Y_scores = best_model.predict_proba(X_test)[:, 1]
        auroc = metric_calc(y_test, Y_scores, 'AUROC')
        auprc = metric_calc(y_test, Y_scores, 'AUPRC')
        precision_ppv = metric_calc(y_test, Y_scores, 'Precision_PPV')
        precision_npv = metric_calc(y_test, Y_scores, 'Precision_NPV')
        mcc = metric_calc(y_test, Y_scores, 'MCC')
        return auroc, auprc, precision_ppv, precision_npv, mcc

def compare_method_kfold(
        dataset_name,
        mol_rep="maccs",
        test_size=0.2,
        train_ablation_ratios=None
):
    print(f"dataset_name: {dataset_name}, mol_rep: {mol_rep}, test_size: {test_size}")

    file_path = f"./data/reg_features/{dataset_name}_{mol_rep}_fp.csv" if model_type == "REG" \
        else f"./data/clf_features/{dataset_name}_{mol_rep}_fp.csv"
    df = pd.read_csv(file_path)
    df['fingerprint'] = df['fingerprint'].apply(safe_eval)
    failed_df = df[df['fingerprint'].isna()]
    print(f"Failed to parse {len(failed_df)} rows")
    df = df.dropna(subset=['fingerprint'])

    X_all = np.array(df['fingerprint'].tolist())
    Y_all = df['label'].values
    smiles_list_all = df['smiles']

    if train_ablation_ratios is None:
        train_ablation_ratios = [1.0]

    model_list = ["TabPFN"] #"XGBoost", "RF", "SVM"
    total_result = []

    for ablation_ratio in train_ablation_ratios:
        print(f"\nCurrent training data ablation ratio: {ablation_ratio}")

        for model_method in model_list:
            print(f"\n---- dataset: {dataset_name}, model: {model_method}, ablation: {ablation_ratio}")
            result = {}

            best_model = run_hyperopt(
                dataset_name, model_method, X_all, Y_all, smiles_list_all, model_type, test_size
            )

            for random_seed in range(20):
                if model_type == 'REG':
                    rmse, mae, r2, pearson_r = get_performance(
                        dataset_name, best_model, X_all, Y_all, random_seed, smiles_list_all, ablation_ratio, test_size
                    )
                    result[random_seed] = [rmse, mae, r2, pearson_r]
                else:
                    auroc, auprc, precision_ppv, precision_npv, mcc = get_performance(
                        dataset_name, best_model, X_all, Y_all, random_seed, smiles_list_all, ablation_ratio, test_size
                    )
                    result[random_seed] = [auroc, auprc, precision_ppv, precision_npv, mcc]

            result_df = pd.DataFrame(result).T
            if model_type == 'REG':
                result_df.columns = ['RMSE', 'MAE', 'R2', 'Pearson_R']
            else:
                result_df.columns = ['AUROC', 'AUPRC', 'Precision_PPV', 'Precision_NPV', 'MCC']

            mean_values = result_df.mean()
            std_values = result_df.std()
            formatted_results = {col: f"{mean_values[col]:.4f} ({std_values[col]:.4f})" for col in result_df.columns}
            formatted_df = pd.DataFrame([formatted_results], index=["Performance"])

            summary_df = pd.concat([result_df, formatted_df], axis=0)
            save_dir = f"result_dataab/{model_type}/{dataset_name}/train_ablation"
            os.makedirs(save_dir, exist_ok=True)
            result_path = f"{save_dir}/{model_method}_ablation_{int(ablation_ratio*100)}.csv"
            summary_df.to_csv(result_path, index=True)

            total_result.append(summary_df)

    # 合并并总存
    final_result_df = pd.concat(total_result)
    data_dir = f"result_dataab/{model_type}/{dataset_name}"
    os.makedirs(data_dir, exist_ok=True)
    csv_path = f"{data_dir}/total_results.csv"
    file_exists = os.path.exists(csv_path)
    final_result_df.to_csv(
        csv_path,
        index=True,
        mode='a',
        header=not file_exists,
    )

def reg_result():
    train_ablation_ratios = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    mol_rep = 'rdkit2d'
    for dataset_name in reg_dataset_list:
        compare_method_kfold(dataset_name, mol_rep, test_size, train_ablation_ratios=train_ablation_ratios)

def cls_result():
    train_ablation_ratios = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    mol_rep = 'rdkit2d'
    for dataset_name in clf_dataset_list:
        compare_method_kfold(dataset_name, mol_rep, test_size, train_ablation_ratios=train_ablation_ratios)

def main():
    # 根据需要选择回归或分类
    # reg_result()
    cls_result()

if __name__ == '__main__':
    model_type = "CLS"   # 或 "REG"
    test_size = 0.2
    split_method = "Random_cv_5"
    main()