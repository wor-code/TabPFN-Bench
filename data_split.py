import json
import os
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
import pandas as pd
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdmolops import FastFindRings
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import AgglomerativeClustering
from dgllife.utils import smiles_to_bigraph

from dataset import ADMETDataset
from featurizer import node_featurizer, edge_featurizer
from constant import reg_dataset_list, clf_dataset_list, proj_dir


def get_distance_matrix(
        scaffolds_fps, molecules_fps,
        scaffold_cluster_flag=False
):
    if scaffold_cluster_flag:
        fps = scaffolds_fps
    else:
        fps = molecules_fps
    distance_matrix = []
    for i, fp in enumerate(fps):
        dist = DataStructs.BulkTanimotoSimilarity(fps[i], fps, returnDistance=1)
        distance_matrix.append(dist)
    return distance_matrix


def group_by_element_value(result):
    groups = {}

    for index, element in enumerate(result):
        if element not in groups:
            groups[element] = []
        groups[element].append(index)

    return list(groups.values())


def cluster_scaffold_molecules(dataset, scaffolds, cluster_result):
    cluster_dict = {}
    cluster_info_dict = {}
    scaffolds_smiles_list = list(scaffolds.keys())

    for cluster_idx, scaffold_cluster in enumerate(cluster_result):

        for scaffold_idx in scaffold_cluster:
            scaffold_smile = scaffolds_smiles_list[scaffold_idx]
            scaffold_mols_idxs = scaffolds[scaffold_smile]
            scaffold_mols_smiles = [dataset.smiles[idx] for idx in scaffold_mols_idxs]

            # 将每个骨架包含的分子添加到骨架聚类组中
            cluster_idx_list = cluster_dict.get(cluster_idx, [])
            cluster_idx_list.extend(scaffold_mols_idxs)
            cluster_dict[cluster_idx] = cluster_idx_list

            cluster_info_dict[cluster_idx] = cluster_info_dict.get(cluster_idx, [])
            cluster_info_dict[cluster_idx].append({
                "scaffold_smile": scaffold_smile,
                "scaffold_mols_idxs": scaffold_mols_idxs,
                "scaffold_mols_smiles": scaffold_mols_smiles,
            })

    save_path_ = f"{save_path}/scaffold_cluster_info_{cluster_method}.json"
    with open(save_path_, "w") as f:
        json.dump(cluster_info_dict, f, indent=4)

    save_path_ = f"{save_path}/{cluster_method}_cluster_dict.json"
    with open(save_path_, "w") as f:
        json.dump(cluster_dict, f, indent=4)

    return cluster_dict


def cluster_split_dataset(
        dataset_name,
        dataset,
        cluster_dict,
        train_cutoff,
        scaffold_list,
        random_seed,
        target_random=False,
        shuffle=True,
        shuffle_threshold=500,
        val_test_percentage=0.5,
):
    train_indices = []
    val_indices = []
    test_indices = []
    target_indices = []

    split_dict = {
        "train": {},
        "val": {},
        "test": {},
        "target": {},
        "target_val": {},
        "target_test": {},
    }
    train_labels = 0
    val_labels = 0
    test_labels = 0

    cluster_dict = {key: sorted(value) for key, value in cluster_dict.items()}
    cluster_dict = {
        idx: cluster_set for (idx, cluster_set) in sorted(
            cluster_dict.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    }

    cluster_list = list(cluster_dict.values())
    cluster_list_len = []
    for idx, cluster_set in enumerate(cluster_list):
        cluster_list_len.append(len(cluster_set))

    if shuffle:
        cluster_sets_new = []
        split_idx = 0
        for idx, cluster_set in enumerate(cluster_list):
            if len(cluster_set) > shuffle_threshold:
                cluster_sets_new.append(cluster_set)
            else:
                split_idx = idx
                break
        scaffold_sets_parts = cluster_list[split_idx:]
        random.shuffle(scaffold_sets_parts)
        cluster_sets_new.extend(scaffold_sets_parts)
        cluster_list = cluster_sets_new

    target_cluster_list = []
    target_cluster_dict = {}
    for idx, group_indices in enumerate(cluster_list):
        if len(train_indices) + len(group_indices) > train_cutoff:
            target_indices.extend(group_indices)
            target_cluster_list.append(group_indices)

            target_cluster_dict[idx] = group_indices
            split_dict["target"][idx] = {
                "idx": group_indices,
                "smiles": [dataset.smiles[i] for i in group_indices],
            }
        else:
            train_indices.extend(group_indices)
            split_dict["train"][idx] = {
                "idx": group_indices,
                "smiles": [dataset.smiles[i] for i in group_indices],
            }
            train_labels += dataset.labels[group_indices].sum()

    if target_random:
        # random shuffle target_indices, split 50% as val, 50% as test
        random.shuffle(target_indices)
        val_cutoff = int(len(target_indices) * val_test_percentage)
        val_indices = target_indices[:val_cutoff]
        test_indices = target_indices[val_cutoff:]
    else:
        # 每个分组都取 80% 到 train_indices, 20% 到 val_indices
        for idx, cluster in enumerate(target_cluster_list):
            random.shuffle(cluster)
            split_index = int(len(cluster) * val_test_percentage)

            val_inx = cluster[:split_index]

            val_labels += dataset.labels[val_inx].sum()
            val_indices.extend(val_inx)

            split_dict["target_val"][idx] = {
                "idx": val_indices,
                "smiles": [dataset.smiles[i] for i in val_indices],
                "scaffold_smile": [scaffold_list[i] for i in val_indices],
            }

            test_inx = cluster[split_index:]
            test_indices.extend(test_inx)
            test_labels += dataset.labels[test_inx].sum()

            split_dict["target_test"][idx] = {
                "idx": test_indices,
                "smiles": [dataset.smiles[i] for i in test_indices],
                "scaffold_smile": [scaffold_list[i] for i in test_indices],
            }

    split_dict["val"] = {
        "idx": val_indices,
        "smiles": [dataset.smiles[i] for i in val_indices],
    }
    split_dict["test"] = {
        "idx": test_indices,
        "smiles": [dataset.smiles[i] for i in test_indices],
    }
    split_dict["train_num"] = len(train_indices)
    split_dict["val_num"] = len(val_indices)
    split_dict["test_num"] = len(test_indices)
    split_dict["train_frac"] = len(train_indices) / len(dataset)

    save_dir = Path(f"{save_path}/{random_seed}")
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path_ = f"{save_dir}/{cluster_method}_scaffold_dataset_split.json"
    with open(save_path_, "w") as f:
        json.dump(split_dict, f, indent=4)

    get_split_data(dataset_name, random_seed, split_dict)


def get_split_data(dataset_name, random_seed, split_dict):
    feature_file = f"data/{model_type}_features/{dataset_name}_{mol_rep}_fp.csv"
    df = pd.read_csv(feature_file)

    train_data = []
    for key in split_dict["train"].keys():
        idx_list = split_dict["train"][key]["idx"]
        filtered_df = df.iloc[idx_list]
        train_data.append(filtered_df)

    save_dir = f"{save_path}/{random_seed}"
    train_data_df = pd.concat(train_data, ignore_index=True)
    output_csv_path = f"{save_dir}/train.csv"
    train_data_df.to_csv(output_csv_path, index=False)

    val_idx = split_dict["val"]["idx"]
    test_idx = split_dict["test"]["idx"]
    val_data_df = df.iloc[val_idx]
    output_csv_path = f"{save_dir}/val.csv"
    val_data_df.to_csv(output_csv_path, index=False)

    test_data_df = df.iloc[test_idx]
    output_csv_path = f"{save_dir}/test.csv"
    test_data_df.to_csv(output_csv_path, index=False)

    target_idx = []
    target_idx.extend(val_idx)
    target_idx.extend(test_idx)
    target_data_df = df.iloc[target_idx]
    output_csv_path = f"{save_dir}/target.csv"
    target_data_df.to_csv(output_csv_path, index=False)


def get_scaffold(
        molecules,
        scaffold_func='decompose'
):
    # 获取所有分子的骨架，并将具有相同骨架的分子分组

    scaffolds = defaultdict(list)
    scaffold_list = []

    for i, mol in enumerate(molecules):
        # For mols that have not been sanitized, we need to compute their ring information
        FastFindRings(mol)

        if scaffold_func == 'decompose':
            # AllChem.MurckoDecompose(mol): 对给定的分子进行Murcko分解
            # Murcko分解是一种将分子转化为其核心骨架的方法，去除分子中的侧链，只保留骨架结构
            mol_scaffold = Chem.MolToSmiles(AllChem.MurckoDecompose(mol))

        if scaffold_func == 'smiles':
            mol_scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol,
                includeChirality=False
            )

        # Group molecules that have the same scaffold
        scaffolds[mol_scaffold].append(i)
        scaffold_list.append(mol_scaffold)

    # 将具有相同骨架的分子分组，并按照每个骨架的分子索引进行排序
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    # 1.按照每个骨架的分子数量进行排序
    # 2.按照每个骨架的分子索引进行升序排序
    scaffolds = {
        scaffold: scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    }
    scaffolds = scaffolds
    return scaffolds, scaffold_list


def prepare_molecules(
        dataset,
        sanitize=True,
):
    molecules = []
    for i, s in enumerate(dataset.smiles):
        molecules.append(Chem.MolFromSmiles(s, sanitize=sanitize))

    fg_func = partial(rdMolDescriptors.GetMorganFingerprintAsBitVect, radius=2, nBits=2048)

    molecules_fps = [fg_func(mol) for mol in molecules]

    scaffolds, scaffold_list = get_scaffold(molecules)

    for scaffold in scaffolds:
        fg_func(Chem.MolFromSmiles(scaffold))

    scaffolds_fps = [
        fg_func(Chem.MolFromSmiles(scaffold)) for scaffold in scaffolds
    ]
    return molecules, molecules_fps, scaffolds, scaffold_list, scaffolds_fps


def load_or_create_cluster_dict(
        dataset_name,
        dataset,
        scaffolds,
        scaffolds_fps,
        molecules_fps,
):
    save_path_ = f"{save_path}/{cluster_method}_cluster_dict.json"
    if os.path.exists(save_path_):
        with open(save_path_, "r") as f:
            cluster_dict = json.load(f)
    else:
        distance_matrix = get_distance_matrix(
            scaffolds_fps, molecules_fps,
            scaffold_cluster_flag=True
        )
        clustering = AgglomerativeClustering(
            n_clusters=cluster_num,
        )
        single_linkage_result = clustering.fit(distance_matrix).labels_
        cluster_result = group_by_element_value(single_linkage_result)
        cluster_dict = cluster_scaffold_molecules(dataset, scaffolds, cluster_result)
    return cluster_dict


def out_of_domain_split(
        dataset_name,
        dataset,
        random_seed,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
):
    molecules, molecules_fps, scaffolds, scaffold_list, scaffolds_fps = prepare_molecules(dataset)

    cluster_dict = load_or_create_cluster_dict(
        dataset_name,
        dataset,
        scaffolds,
        scaffolds_fps,
        molecules_fps,
    )

    train_cutoff = int(frac_train * len(molecules))
    val_cutoff = int((frac_train + frac_valid) * len(molecules))
    return cluster_split_dataset(dataset_name, dataset, cluster_dict, train_cutoff, scaffold_list, random_seed)


def reg_dataset_split():
    for dataset_name in reg_dataset_list:

        cache_file_path = f"{proj_dir}/data/cache/{dataset_name}_dglgraph.bin"

        dataset_file = f"{proj_dir}/data/reg/{dataset_name}.csv"
        # get last column name
        y_col_names = pd.read_csv(dataset_file, nrows=1).columns[-1]

        dataset = ADMETDataset(
            smiles_to_graph=smiles_to_bigraph,
            path=dataset_file,
            smiles_column="smiles",
            task_names=y_col_names,
            cache_file_path=cache_file_path,
            node_featurizer=node_featurizer,
            edge_featurizer=edge_featurizer,
            load=True,
        )

        global save_path
        save_path = f"./split_data/{model_type_dir}/{dataset_name}"
        os.makedirs(save_path, exist_ok=True)

        for i in range(10):
            random_seed = i
            out_of_domain_split(dataset_name, dataset, random_seed)


def cls_dataset_split():
    for dataset_name in clf_dataset_list:

        cache_file_path = f"{proj_dir}/data/cache/{dataset_name}_dglgraph.bin"

        dataset_file = f"{proj_dir}/data/clf/{dataset_name}.csv"
        # get last column name
        y_col_names = pd.read_csv(dataset_file, nrows=1).columns[-1]

        dataset = ADMETDataset(
            smiles_to_graph=smiles_to_bigraph,
            path=dataset_file,
            smiles_column="smiles",
            task_names=y_col_names,
            cache_file_path=cache_file_path,
            node_featurizer=node_featurizer,
            edge_featurizer=edge_featurizer,
            load=True,
        )

        global save_path
        save_path = f"./split_data/{dataset_name}"
        os.makedirs(save_path, exist_ok=True)

        for i in range(1):
            random_seed = i
            out_of_domain_split(dataset_name, dataset, random_seed)


def process_PPB():
    import pandas as pd

    csv_file_path = "./data/reg/PPB.csv"  # 替换成你的文件路径
    df = pd.read_csv(csv_file_path)

    # 重新生成连续的 Index 列
    df["Index"] = range(1, len(df) + 1)

    # 保存到新的 CSV 文件
    output_csv_path = "./data/reg/PPB_new.csv"
    df.to_csv(output_csv_path, index=False)


def main():
    global model_type
    global model_type_dir
    # model_type = "reg"
    # model_type_dir = "REG"
    # reg_dataset_split()

    model_type = "clf"
    model_type_dir = "CLS"
    cls_dataset_split()
    pass


if __name__ == '__main__':
    mol_rep = "rdkit2d"
    cluster_method = "ood"
    cluster_num = 50
    main()
