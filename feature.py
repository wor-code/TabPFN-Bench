import ast
import concurrent.futures
import re

import numpy as np
import pandas as pd
from descriptastorus.descriptors import rdNormalizedDescriptors
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors

from constant import *

MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048


def generate_reps(smiles, mol_rep: str):
    """Generate fixed molecular representations
    Inputs:
        smiles: a molecule in SMILES representation
        mol_rep: representation name - options include morganBits, morganCounts, maccs, physchem, rdkit2d, atomPairs
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol_rep == 'morganBits':
        features_vec = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=MORGAN_RADIUS,
            nBits=MORGAN_NUM_BITS
        )
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    elif mol_rep == 'morganCounts':
        features_vec = AllChem.GetHashedMorganFingerprint(
            mol, radius=MORGAN_RADIUS,
            nBits=MORGAN_NUM_BITS
        )
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    elif mol_rep == 'maccs':
        features_vec = MACCSkeys.GenMACCSKeys(mol)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    elif mol_rep == 'physchem':
        # calculate physchem descriptors values
        # Reference: https://github.com/molML/MoleculeACE/blob/main/MoleculeACE/benchmark/featurization.py
        weight = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        h_bond_donor = Descriptors.NumHDonors(mol)
        h_bond_acceptors = Descriptors.NumHAcceptors(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        atoms = Chem.rdchem.Mol.GetNumAtoms(mol)
        heavy_atoms = Chem.rdchem.Mol.GetNumHeavyAtoms(mol)
        molar_refractivity = Chem.Crippen.MolMR(mol)
        topological_polar_surface_area = Chem.QED.properties(mol).PSA
        formal_charge = Chem.rdmolops.GetFormalCharge(mol)
        rings = Chem.rdMolDescriptors.CalcNumRings(mol)
        # form features matrix
        features = np.array([
            weight, logp,
            h_bond_donor,
            h_bond_acceptors,
            rotatable_bonds,
            atoms, heavy_atoms,
            molar_refractivity,
            topological_polar_surface_area,
            formal_charge,
            rings
        ])
    elif mol_rep == 'rdkit2d':
        # instantiate a descriptors generator
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]
    elif mol_rep == 'atomPairs':
        features_vec = rdMolDescriptors.GetHashedAtomPairFingerprint(
            mol, nBits=2048,
            use2D=True
        )
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    else:
        raise ValueError('Not defined fingerprint!')

    return features


def process_molecule(index, smiles, label):
    fingerprint = generate_reps(smiles, mol_rep)
    return index, smiles, fingerprint, label


def get_features(dataset_name):
    df = pd.read_csv(f'./data/{model_type}/{dataset_name}.csv')
    smiles_list = df["smiles"].tolist()
    label_list = df.iloc[:, -1].tolist()

    results = []
    # 并行计算
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_molecule, idx + 1, smiles, label): smiles
            for idx, (smiles, label) in enumerate(zip(smiles_list, label_list))
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result[2] is not None:
                    results.append(result)
                else:
                    pass
            except Exception as e:
                pass

    # 构造结果 DataFrame，并保存为 CSV 文件
    results.sort(key=lambda x: x[0])
    results_df = pd.DataFrame(results, columns=["index", "smiles", "fingerprint", "label"])
    output_csv = f"data/{model_type}_features/{dataset_name}_{mol_rep}_fp.csv"
    results_df.to_csv(output_csv, index=False)


def get_reg_all_features():
    global model_type
    model_type = "reg"
    for dataset_name in reg_dataset_list:
        get_features(dataset_name)


def get_clf_all_features():
    global model_type
    model_type = "clf"
    clf_dataset_list = [
        "AMES Toxicity",
    ]
    for dataset_name in clf_dataset_list:
        get_features(dataset_name)


def get_qm_features():
    global model_type
    model_type = "qm"
    for dataset_name in qm_dataset_list:
        get_features(dataset_name)


def get_physical_features():
    global model_type
    model_type = "physical"
    for dataset_name in physical_dataset_list:
        get_features(dataset_name)


def get_biophysics_features():
    global model_type
    model_type = "biophysics"
    for dataset_name in biophysics_dataset_list:
        get_features(dataset_name)


def get_pdbbind_features():
    pdbbind_dir = "cwr-02/cwr/data/PDBbind/v2020_processed/v2020_refined/"

    pass


def test_1():
    smiles = "Clc1c(C(=O)O)ccc(Cl)c1"
    mol_rep = "rdkit2d"
    features = generate_reps(smiles, mol_rep)
    pass


def safe_parse_fingerprint(fp_raw, idx=None):
    try:
        if pd.isna(fp_raw):
            return []
        fp = str(fp_raw).replace('\n', ' ').strip()
        fp = re.sub(r'(\d)\. ', r'\1.0, ', fp)
        fp = re.sub(r'(\d)\.]', r'\1.0]', fp)
        fp = re.sub(r'\bnan\b', '0.0', fp)  # ✅ 处理裸 nan
        if not fp.startswith("["):
            fp = "[" + fp
        if not fp.endswith("]"):
            fp = fp + "]"
        return ast.literal_eval(fp)
    except Exception as e:
        print(f"[ERROR] Failed to parse fingerprint at index {idx}: {e}")
        print(f"Raw: {fp_raw}")
        return []


def merge_feature(dataset_name):
    rdkit_csv = f"data/{model_type}_features/{dataset_name}_rdkit2d_fp.csv"
    maccs_csv = f"data/{model_type}_features/{dataset_name}_maccs_fp.csv"

    rdkit_pd = pd.read_csv(rdkit_csv)
    maccs_pd = pd.read_csv(maccs_csv)

    rdkit_pd['fingerprint'] = rdkit_pd.apply(
        lambda row: safe_parse_fingerprint(row['fingerprint'], row['index']), axis=1
    )
    maccs_pd['fingerprint'] = maccs_pd.apply(
        lambda row: safe_parse_fingerprint(row['fingerprint'], row['index']), axis=1
    )

    # check same length
    if len(rdkit_pd) != len(maccs_pd):
        raise ValueError("The two dataframes have different lengths.")

    # merge two dataframes
    rdkit_pd['fingerprint'] = rdkit_pd['fingerprint'] + maccs_pd['fingerprint']
    rdkit_pd['fingerprint'] = rdkit_pd['fingerprint'].apply(lambda x: np.array(x).tolist())

    # save to csv
    mol_rep = "rdkit2d+maccs"
    output_csv = f"data/{model_type}_features/{dataset_name}_{mol_rep}_fp.csv"
    rdkit_pd.to_csv(output_csv, index=False)


def merge_clf_feature():
    # merge rdkit2D and maccs
    global model_type
    model_type = "clf"
    clf_dataset_list = [
        "AMES Toxicity",
    ]
    for dataset_name in clf_dataset_list:
        merge_feature(dataset_name)


def merge_reg_feature():
    # merge rdkit2D and maccs
    global model_type
    model_type = "reg"
    for dataset_name in reg_dataset_list:
        merge_feature(dataset_name)


def merge_qm_feature():
    # merge rdkit2D and maccs
    global model_type
    model_type = "qm"
    for dataset_name in qm_dataset_list:
        merge_feature(dataset_name)


def merge_physical_feature():
    # merge rdkit2D and maccs
    global model_type
    model_type = "physical"
    for dataset_name in physical_dataset_list:
        merge_feature(dataset_name)


def merge_biophysics_feature():
    # merge rdkit2D and maccs
    global model_type
    model_type = "biophysics"
    for dataset_name in biophysics_dataset_list:
        merge_feature(dataset_name)


def merge_all():
    merge_clf_feature()
    pass


def main():
    merge_all()
    pass


if __name__ == '__main__':
    mol_rep = "rdkit2d"
    main()
