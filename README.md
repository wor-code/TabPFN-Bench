# TabPFN-Bench

**Benchmarking TabPFN for Small-Data and OOD Molecular Property Prediction**

This repository provides the complete experimental pipeline used in  
**“TabPFN Opens New Avenues for Small-Data Tabular Learning in Drug Discovery”**  
for benchmarking TabPFN against conventional machine learning models on ADMET,
physicochemical, and quantum-mechanical molecular property prediction tasks.

The focus of this benchmark is **tabular molecular modeling under realistic constraints**:
small sample sizes, heterogeneous descriptors, feature/data ablation, and
out-of-distribution (OOD) generalization.

---

## Key Features

This repository supports:

- **Fixed-length molecular feature generation**
  - RDKit 2D descriptors
  - MACCS keys
- **In-domain evaluation**
  - Repeated random splits with paired statistical comparison
- **Out-of-distribution (OOD) evaluation**
  - cluster-based splits
- **Robustness analysis**
  - Feature ablation (10%–90%)
  - Training data ablation (10%–90%)
- **Multi-model benchmarking**
  - TabPFN
  - XGBoost
  - Random Forest
  - Support Vector Machine (SVM)

The code is designed to reproduce the main figures and supplementary analyses
reported in the paper (Figures 1–5 and Figures S1–S5).
---

## Repository Structure
```
TabPFN-Bench/
├── feature.py # Molecular feature generation
├── common.py # Model builders, metrics, and utilities
├── compare_method.py # In-domain (random split) evaluation
├── compare_method_ood.py # OOD evaluation
├── data_split.py # cluster-based OOD split generation
├── compare_method_dataab.py # Training data ablation experiments
├── compare_method_fab.py # Feature ablation experiments
├── constant.py # Dataset lists and metric labels
├── dataset.py # Dataset wrappers
├── featurizer.py # Molecular featurization utilities
│
├── data/
│ ├── clf/ # Classification datasets (CSV)
│ ├── reg/ # Regression datasets (CSV)
│ └── physical/ # Physicochemical / QM datasets
│
├── split_data/ # Generated OOD splits
└── result/ # Benchmark results (CSV)
```

## Data Format
Each dataset is stored as a CSV file with the following requirements:
```csv
smiles, ..., target
CCO,...,0.123
```
Expected dataset locations:
Classification: data/clf/{dataset_name}.csv
Regression: data/reg/{dataset_name}.csv
Physicochemical / QM: data/physical/{dataset_name}.csv

## Environment & Dependencies

Core dependencies include:
numpy, pandas, scipy, scikit-learn
rdkit, descriptastorus
torch
xgboost, hyperopt
tabpfn, tabpfn_extensions


## Feature Generation
Generate fixed molecular representations:
python feature.py
Generated features are saved to:
data/{model_type}_features/{dataset_name}_{mol_rep}_fp.csv


