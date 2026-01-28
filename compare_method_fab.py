import sys
import numpy as np
sys.path.append('/public/home/tianyao/TabPFN_test/finetune_tabpfn_v2')
from constant import *
from common import *

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")


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
    print(f"tag: {tag}, RMSE': {rmse}, 'MAE': {mae}, 'R2': {r2}, 'Pearson_R': {pearson_r}")
    print(result_df)
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
    print(f"X_test_smiles: {smiles_test.head(3)}, len: {len(smiles_test)}")

    best_model.fit(X_train, y_train)
    Y_pred = best_model.predict(X_test)

    if model_type == 'REG':
        # 回归任务的指标计算  
        test_rmse, test_mae, test_r2, test_pearson_r, test_result_df = get_reg_metrics(
            y_test, Y_pred, smiles_test, 'test',
        )
        return test_rmse, test_mae, test_r2, test_pearson_r
    else:
        # 分类任务的指标计算  
        Y_scores = best_model.predict_proba(X_test)[:, 1]  # 获取正类的概率  

        # 计算分类指标  
        auroc = metric_calc(y_test, Y_scores, 'AUROC')
        auprc = metric_calc(y_test, Y_scores, 'AUPRC')
        precision_ppv = metric_calc(y_test, Y_scores, 'Precision_PPV')
        precision_npv = metric_calc(y_test, Y_scores, 'Precision_NPV')
        mcc = metric_calc(y_test, Y_scores, 'MCC')

        # 打印指标  
        print(
            f"tag: test, 'AUROC': {auroc}, 'AUPRC': {auprc}, 'Precision_PPV': {precision_ppv}, 'Precision_NPV': {precision_npv}, 'MCC': {mcc}")

        # 收集预测概率和标签  
        test_result_df = pd.DataFrame({
            'preds': Y_scores,
            'labels': y_test,
            'SMILES': smiles_test,
        }, columns=['preds', 'labels', 'SMILES'])
        print(test_result_df.head())

        # 返回分类指标  
        return auroc, auprc, precision_ppv, precision_npv, mcc


def safe_eval(x):
    try:
        if isinstance(x, (list, np.ndarray)):
            return np.array(x)

        if isinstance(x, str):
            # 处理多行字符串格式的numpy数组  
            # 1. 删除方括号并替换换行为空格  
            cleaned_str = x.strip('[]').replace('\n', ' ')
            # 2. 通过空格分割并过滤掉空字符串  
            num_strings = [s for s in cleaned_str.split(' ') if s]
            # 3. 转换为浮点数数组  
            return np.array([float(val) for val in num_strings])

        return x
    except Exception as e:
        print(f"Error parsing: {x}")
        print(f"Error details: {e}")
        return np.nan


def random_feature_ablation(X, ablation_ratio=0.1, random_seed=42):
    """  
    X: 原始特征矩阵，形状 (样本数, 特征数)  
    ablation_ratio: 要消融的特征比例，比如 0.1 表示删除10%的特征  
    返回：消融后的特征矩阵，和保留的特征索引列表  
    """
    np.random.seed(random_seed)
    n_features = X.shape[1]
    n_remove = int(n_features * ablation_ratio)
    all_indices = np.arange(n_features)

    if n_remove == 0:
        return X, all_indices  # 不消融  

    remove_indices = np.random.choice(all_indices, size=n_remove, replace=False)
    keep_indices = np.setdiff1d(all_indices, remove_indices)

    X_new = X[:, keep_indices]
    return X_new, keep_indices


def compare_method_kfold(
        dataset_name,
        mol_rep="maccs",
        test_size=0.2,
        ablation_ratios=None  # 可以为None
):
    print(f"dataset_name: {dataset_name}, mol_rep: {mol_rep}, test_size: {test_size}")

    # 根据模型类型选择正确的数据目录
    if model_type == "CLS":
        file_path = f"./data/clf_features/{dataset_name}_{mol_rep}_fp.csv"
    else:
        file_path = f"./data/reg_features/{dataset_name}_{mol_rep}_fp.csv"

    df = pd.read_csv(file_path)
    df['fingerprint'] = df['fingerprint'].apply(safe_eval)
    failed_df = df[df['fingerprint'].isna()]
    print(f"Failed to parse {len(failed_df)} rows")
    df = df.dropna(subset=['fingerprint'])

    # 转换为list-of-list，再转为2D数组
    X_all = np.array(df['fingerprint'].tolist())
    Y_all = df['label'].values
    smiles_list_all = df["smiles"]

    print(df.head(5))

    if (ablation_ratios is None) or (len(ablation_ratios) == 0):
        ablation_ratios = [0.0]

    model_list = ["TabPFN"]  # "XGBoost", "RF", "SVM" "TabPFN"
    total_result = []

    for ablation_ratio in ablation_ratios:
        print(f"\n\nCurrent feature ablation ratio: {ablation_ratio}")

        if ablation_ratio == 0.0:
            X = X_all
            keep_indices = np.arange(X.shape[1])
        else:
            X, keep_indices = random_feature_ablation(X_all, ablation_ratio)
        Y = Y_all
        smiles_list = smiles_list_all

        print(f"Original feature count: {X_all.shape[1]}, After ablation: {X.shape[1]}")

        for model_method in model_list:
            print(
                f"\n\n----------------------\n dataset: {dataset_name}, model_method: {model_method}, ablation_ratio: {ablation_ratio}")
            result = {}

            best_model = hyper_model(
                dataset_name,
                X, Y,
                model_type,
                model_method,
                fine_tune_model_path=None,
                ablation_ratio=ablation_ratio,
                fallback_ablation_ratio=(0.7 if (model_method == "SVM" and abs(ablation_ratio - 0.8) < 1e-8) else None)
            )

            for random_seed in range(20):
                if model_type == 'REG':
                    rmse, mae, r2, pearson_r = get_performace(
                        dataset_name, best_model, X, Y, random_seed, smiles_list, test_size
                    )
                    result[random_seed] = [rmse, mae, r2, pearson_r]
                else:
                    auroc, auprc, precision_ppv, precision_npv, mcc = get_performace(
                        dataset_name, best_model, X, Y, random_seed, smiles_list, test_size
                    )
                    result[random_seed] = [auroc, auprc, precision_ppv, precision_npv, mcc]

            # 汇总结果，合并均值和std
            result_df = pd.DataFrame(result).T
            if model_type == 'REG':
                result_df.columns = ['RMSE', 'MAE', 'R2', 'Pearson_R']
            else:
                result_df.columns = ['AUROC', 'AUPRC', 'Precision_PPV', 'Precision_NPV', 'MCC']

            mean_values = result_df.mean()
            std_values = result_df.std()
            formatted_results = {col: f"{mean_values[col]:.4f} ({std_values[col]:.4f})" for col in result_df.columns}
            formatted_df = pd.DataFrame([formatted_results], index=["Performance"])

            # 合并均值±std与原始结果
            summary_df = pd.concat([result_df, formatted_df], axis=0)
            # 如果你还想加上模型名字在最前列可以这样：
            # summary_df.insert(0, "Model", model_method)

            save_dir = f"result94/{model_type}/{dataset_name}/feature_ablation"
            os.makedirs(save_dir, exist_ok=True)
            result_path = f"{save_dir}/{model_method}_ablation_{int(ablation_ratio * 100)}.csv"
            summary_df.to_csv(result_path, index=True)

            # 也可以往 total_result 汇总
            total_result.append(summary_df)

    # 最终合并所有消融实验/模型的结果
    final_result_df = pd.concat(total_result)
    data_dir = f"result94/{model_type}/{dataset_name}"
    os.makedirs(data_dir, exist_ok=True)

    csv_path = f"{data_dir}/{split_method}_total_results.csv"
    file_exists = os.path.exists(csv_path)
    final_result_df.to_csv(
        csv_path,
        index=True,
        mode='a',
        header=not file_exists,
    )


def reg_result():
    global test_size
    global model_type

    model_type = "REG"
    test_size = 0.2
    mol_rep = 'rdkit2d'
    ablation_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # dataset_name = "IGC50"

    for dataset_name in reg_dataset_list:
        compare_method_kfold(dataset_name, mol_rep, test_size, ablation_ratios)


def cls_result():
    global test_size
    global model_type

    model_type = "CLS"
    test_size = 0.2
    mol_rep = 'rdkit2d'
    ablation_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for dataset_name in clf_dataset_list:
        compare_method_kfold(dataset_name, mol_rep, test_size, ablation_ratios)


def main():
    cls_result()


if __name__ == '__main__':
    model_type = "CLS"
    split_method = "Random_cv_5"
    main()
