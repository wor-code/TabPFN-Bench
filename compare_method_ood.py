import pandas as pd
import numpy as np
import os
import ast
from constant import reg_dataset_list, clf_dataset_list
from common import *


def get_performace(
        model,
        X_train, y_train, smiles_train,
        X_test, y_test, smiles_test,
):
    model.fit(X_train, y_train)
    Y_pred = model.predict(X_test)

    if model_type == 'REG':
        test_result_df = pd.DataFrame({
            'preds': Y_pred,
            'labels': y_test,
            'SMILES': smiles_test,
        }, columns=['preds', 'labels', 'SMILES'])

        rmse = metric_calc(y_test, Y_pred, 'RMSE')
        mae = metric_calc(y_test, Y_pred, 'MAE')
        r2 = metric_calc(y_test, Y_pred, 'R2')
        pearson_r = metric_calc(y_test, Y_pred, 'Pearson_R')

        return rmse, mae, r2, pearson_r
    else:
        Y_scores = model.predict_proba(X_test)[:, 1]
        # assemble the test_result_df by collecting prediction probability for each molecule
        test_result_df = pd.DataFrame({
            'preds': Y_scores,
            'labels': y_test,
            'SMILES': smiles_test,
        }, columns=['preds', 'labels', 'SMILES'])


def safe_eval(x):
    try:
        return np.array(ast.literal_eval(x))
    except Exception as e:
        return np.nan


def compare_method(
        dataset_name,
        random_seed,
        model_method,
        X_train, y_train, smiles_train,
        X_target, y_target, smiles_target,
        mol_rep="rdkit2d",
        test_size=0.2
):
    model = model_dict[model_method](model_type)
    result = {}

    rmse, mae, r2, pearson_r = get_performace(
        model,
        X_train, y_train, smiles_train,
        X_target, y_target, smiles_target,
    )
    result[random_seed] = [rmse, mae, r2, pearson_r]

    # get the average performance
    result_df = pd.DataFrame(result).T
    result_df.columns = ['RMSE', 'MAE', 'R2', 'Pearson_R']
    result_df.insert(0, 'Model', model_method)
    return result_df


def clean_data(df):
    df['fingerprint'] = df['fingerprint'].apply(safe_eval)
    df = df.dropna(subset=['fingerprint'])

    X = np.array(df['fingerprint'].tolist())
    y = df['label'].values
    smiles = df["smiles"]
    return df, X, y, smiles


def compare_method_repeat(
        dataset_name,
):
    total_result = []
    for random_seed in range(repeat_num):
        data_dir = f"./split_data/{model_type}/{dataset_name}/{random_seed}"
        train_path = f"{data_dir}/train.csv"
        train_df = pd.read_csv(train_path)
        train_df, X_train, y_train, smiles_train = clean_data(train_df)

        target_path = f"{data_dir}/target.csv"
        target_df = pd.read_csv(target_path)
        target_df, X_target, y_target, smiles_target = clean_data(target_df)

        model_list = [
            "XGBoost",
            "RF",
            "SVM",
            "TabPFN",
        ]
        for model_method in model_list:
            result_df = compare_method(
                dataset_name,
                random_seed,
                model_method,
                X_train, y_train, smiles_train,
                X_target, y_target, smiles_target,
                mol_rep="rdkit2d",
                test_size=0.2,
            )
            total_result.append(result_df)

    total_result_df = pd.concat(total_result)
    # total_result_df  column is Model,RMSE,MAE,R2,Pearson_R
    # get the average performance for each model, and save the results in total_result_df
    mean = total_result_df.groupby('Model').mean().reset_index()
    std = total_result_df.groupby('Model').std().reset_index()
    agg_df = mean.copy()
    for col in ["RMSE", "MAE", "R2", "Pearson_R"]:
        agg_df[col] = mean[col].round(4).astype(str) + " (" + std[col].round(4).astype(str) + ")"
    agg_df["Model"] = mean["Model"]

    final_result_df = pd.concat([total_result_df, agg_df], ignore_index=True)

    data_dir = f"result/{model_type}/{dataset_name}"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    csv_path = f"{data_dir}/{split_method}_total_results.csv"
    file_exists = os.path.exists(csv_path)
    final_result_df.to_csv(
        f"{data_dir}/{split_method}_total_results.csv",
        index=False,
        mode='a',
        header=not file_exists,
    )


def cls_result():
    global test_size
    global model_type

    model_type = "CLS"
    test_size = 0.2
    mol_rep = 'rdkit2d'

    for dataset_name in clf_dataset_list:
        compare_method_repeat(dataset_name)

    pass


def reg_result():
    global test_size
    global model_type

    model_type = "REG"
    test_size = 0.2

    for dataset_name in reg_dataset_list:
        compare_method_repeat(dataset_name)


def main():
    reg_result()


if __name__ == '__main__':
    repeat_num = 20
    mol_rep = 'rdkit2d'
    split_method = "ood_total"
    main()
