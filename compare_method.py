import ast  # 解析字符串列表
import os

from constant import *
from common import *


def get_reg_metrics(y_true, y_pred, smiles_test, tag):
    # assemble the test_result_df by collecting prediction results for each molecule
    result_df = pd.DataFrame({
        'preds': y_pred,
        'labels': y_true,
        'SMILES': smiles_test,
    }, columns=['preds', 'labels', 'SMILES'])

    rmse = metric_calc(y_true, y_pred, 'RMSE')
    mae = metric_calc(y_true, y_pred, 'MAE')
    r2 = metric_calc(y_true, y_pred, 'R2')
    pearson_r = metric_calc(y_true, y_pred, 'Pearson_R')
    return rmse, mae, r2, pearson_r, result_df


def get_performace(
        dataset_name,
        best_model,
        X,
        Y,
        random_seed,
        smiles_list,
        test_size=0.2
):
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
        X,
        Y,
        smiles_list,
        test_size=test_size,
        random_state=random_seed,
    )

    best_model.fit(X_train, y_train)
    Y_pred = best_model.predict(X_test)

    if model_type == 'REG':
        test_rmse, test_mae, test_r2, test_pearson_r, test_result_df = get_reg_metrics(
            y_test, Y_pred, smiles_test, 'test',
        )
        return test_rmse, test_mae, test_r2, test_pearson_r
    else:
        # get class probability
        Y_scores = best_model.predict_proba(X_test)[:, 1]
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
        return np.nan  # 解析失败返回 NaN


def compare_method_kfold(
        dataset_name,
        mol_rep="rdkit2d",
        test_size=0.2
):
    file_path = f"./data/reg_features/{dataset_name}_{mol_rep}_fp.csv"
    df = pd.read_csv(file_path)

    df['fingerprint'] = df['fingerprint'].apply(safe_eval)
    df = df.dropna(subset=['fingerprint'])

    X = np.array(df['fingerprint'].tolist())
    Y = df['label'].values
    smiles_list = df["smiles"]

    model_list = [
        "XGBoost",
        "RF",
        "SVM",
    ]

    total_result = []
    for model_method in model_list:
        result = {}

        best_model = run_hyperopt(
            dataset_name,
            model_method,
            X, Y, smiles_list,
            model_type,
            test_size,
        )

        for random_seed in range(20):
            rmse, mae, r2, pearson_r = get_performace(
                dataset_name,
                best_model, X, Y, random_seed, smiles_list, test_size
            )
            result[random_seed] = [rmse, mae, r2, pearson_r]

        # get the average performance
        result_df = pd.DataFrame(result).T
        result_df.columns = ['RMSE', 'MAE', 'R2', 'Pearson_R']
        mean_values = result_df.mean()
        std_values = result_df.std()
        formatted_results = {col: f"{mean_values[col]:.4f} ({std_values[col]:.4f})" for col in result_df.columns}
        formatted_df = pd.DataFrame(formatted_results, index=["Performance"])

        merged_df = pd.concat([result_df, formatted_df])
        merged_df.insert(0, "Model", model_method)
        total_result.append(merged_df)

    final_result_df = pd.concat(total_result)
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


def reg_result():
    global test_size
    global model_type

    model_type = "REG"
    test_size = 0.2
    mol_rep = 'rdkit2d'

    for dataset_name in reg_dataset_list:
        compare_method_kfold(dataset_name, mol_rep, test_size)
    pass


def cls_result():
    global test_size
    global model_type

    model_type = "CLS"
    test_size = 0.2
    mol_rep = 'rdkit2d'
    pass


def main():
    reg_result()


if __name__ == '__main__':
    import sys

    sys.stdout = open("log/reg_kfold.log", "a", buffering=1)
    sys.stderr = sys.stdout

    model_type = "REG"
    split_method = "Random_cv_5"
    main()
