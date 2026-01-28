proj_dir = "HOME_DIR"
reg_dataset_list = [
    'BP',
    'LogP',
    'Bioconcentration Factor',
    'Caco-2',
    'Fu',
    'IGC50',
    'LC50DM',
    'LC50FM',
    'Melting_point',
    'Neurotoxicity_mouse',
]
reg_metric_labels = ['RMSE', 'MAE', 'R2', 'Pearson_R']

clf_dataset_list = [
    "AMES Toxicity",
    'FDAMDD',
    'H-HT',
    'HIA',
    'SR-ARE',
    'Eye Corrosion',
    'Genotoxicity',
    'Skin Sensitization',
    'CYP1A2 substrate',
    'OATP1B3',
    'BCRP',
    'LM mouse',
    'SR-HSE',
    'BSEP',
    'CYP3A4 substrate',
    'NR-AR',
    'PAMPA',
    'A549',
    'NR-ER',
    'Pgp-substrate',
    'Pgp-inhibitor',
    'Eye Irritation',
    'SR-MMP',
    'MRP1',
    'NR-AhR',
    'DILI',
]

clf_metric_labels = ['AUROC', 'AUPRC', 'Precision_PPV', 'Precision_NPV', "MCC"]

# QM
qm_dataset_list = [
    "qm7",  # 7160
    'qm8_E1-CC2',  # 21786
    'qm8_E2-CC2',
    'qm8_f1-CC2',
    'qm8_f2-CC2',

    'qm8_E1-PBE0',
    'qm8_E2-PBE0',
    'qm8_f1-PBE0',
    'qm8_f2-PBE0',
    'qm8_E1-PBE0.1',

    'qm8_E2-PBE0.1',
    'qm8_f1-PBE0.1',
    'qm8_f2-PBE0.1',
    'qm8_E1-CAM',
    'qm8_E2-CAM',
    'qm8_f1-CAM',
    'qm8_f2-CAM',
]

QM8_TASKS = [
    "E1-CC2", "E2-CC2", "f1-CC2",
    "f2-CC2", "E1-PBE0",
    "E2-PBE0", "f1-PBE0",
    "f2-PBE0", "E1-PBE0", "E2-PBE0",
    "f1-PBE0", "f2-PBE0",
    "E1-CAM", "E2-CAM",
    "f1-CAM", "f2-CAM"
]
QM7_TASKS = ["u0_atom"]


physical_dataset_list = [
    "esol",  # 1128
    "freesolv",  # 642
    "Lipophilicity",  # 4200
]

