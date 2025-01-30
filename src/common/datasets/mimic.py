import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import numpy as np
import pandas as pd
import torch
import json
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations

from src.common.modules.common import MLP, PatchEmbeddings, Linear
from src.common.utils import get_modality_combinations


def load_and_preprocess_data_mimic(args):
    # Paths
    lab_path = "data/mimic4 Dataset/lab_x"
    note_path = "data/mimic4 Dataset/note_x"
    code_path = "data/mimic4 Dataset/code_x"
    label_df = pd.read_csv(
        "data/mimic4 Dataset/labels.csv",
        index_col="subject_id",
    )
    labels = label_df["one_year_mortality"].values.astype(np.int64)
    n_labels = len(set(labels))

    with open("data/mimic4 Dataset/PTID_splits_mimic.json") as json_file:
        data_split = json.load(json_file)

    train_ids = list(set(data_split["training"]))
    valid_ids = list(set(data_split["validation"]))
    test_ids = list(set(data_split["testing"]))

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}

    id_to_idx = {id: idx for idx, id in enumerate(label_df.index)}
    common_idx_list = []
    observed_idx_arr = np.zeros((labels.shape[0], 4), dtype=bool)  # IGCB order

    # Initialize modality combination list
    modality_combinations = [""] * len(id_to_idx)

    def update_modality_combinations(idx, modality):
        nonlocal modality_combinations
        if modality_combinations[idx] == "":
            modality_combinations[idx] = modality
        else:
            modality_combinations[idx] += modality

    # Load modalities
    if "L" in args.modality or "l" in args.modality:
        path = lab_path
        arr = torch.load(path + ".pt")
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # arr = scaler.fit_transform(arr)
        new_idx = np.arange(arr.shape[0])
        filtered_idx = new_idx[new_idx != -1]
        for idx in filtered_idx:
            update_modality_combinations(idx, "L")
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]
        tmp = np.nan_to_num(tmp, nan=0.0)
        observed_idx_arr[filtered_idx, 0] = True
        data_dict["lab"] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        print(f"Size of lab input: {arr.shape[1]}")
        if args.patch:
            encoder_dict["lab"] = PatchEmbeddings(
                arr.shape[1], args.num_patches, args.hidden_dim
            ).to(args.device)
        else:
            encoder_dict["lab"] = MLP(
                arr.shape[1], args.hidden_dim, args.hidden_dim, args.num_layers_enc
            ).to(args.device)
        input_dims["lab"] = arr.shape[1]

    if "N" in args.modality or "n" in args.modality:
        path = note_path
        arr = torch.load(path + ".pt")
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # arr = scaler.fit_transform(arr)
        new_idx = np.arange(arr.shape[0])
        filtered_idx = new_idx[new_idx != -1]
        for idx in filtered_idx:
            update_modality_combinations(idx, "N")
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]
        tmp = np.nan_to_num(tmp, nan=0.0)
        observed_idx_arr[filtered_idx, 1] = True
        data_dict["note"] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        print(f"Size of note input: {arr.shape[1]}")
        if args.patch:
            encoder_dict["note"] = PatchEmbeddings(
                arr.shape[1], args.num_patches, args.hidden_dim
            ).to(args.device)
        else:
            encoder_dict["note"] = MLP(
                arr.shape[1], args.hidden_dim, args.hidden_dim, args.num_layers_enc
            ).to(args.device)
        input_dims["note"] = arr.shape[1]

    if "C" in args.modality or "c" in args.modality:
        path = code_path
        arr = torch.load(path + ".pt")
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # arr = scaler.fit_transform(arr)
        new_idx = np.arange(arr.shape[0])
        filtered_idx = new_idx[new_idx != -1]
        for idx in filtered_idx:
            update_modality_combinations(idx, "C")
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]
        tmp = np.nan_to_num(tmp, nan=0.0)
        observed_idx_arr[filtered_idx, 2] = True
        data_dict["code"] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        print(f"Size of code input: {arr.shape[1]}")
        if args.patch:
            encoder_dict["code"] = PatchEmbeddings(
                arr.shape[1], args.num_patches, args.hidden_dim
            ).to(args.device)
        else:
            encoder_dict["code"] = MLP(
                arr.shape[1], args.hidden_dim, args.hidden_dim, args.num_layers_enc
            ).to(args.device)
        input_dims["code"] = arr.shape[1]

    combination_to_index = get_modality_combinations(
        args.modality
    )  # 0: full modality index
    modality_combinations = [
        "".join(sorted(set(comb))) for comb in modality_combinations
    ]
    full_modality_index = min(list(combination_to_index.values()))
    assert full_modality_index == 0  # max(list(combination_to_index.values()))
    _keys = combination_to_index.keys()
    data_dict["modality_comb"] = [
        combination_to_index[comb] if comb in _keys else -1
        for comb in modality_combinations
    ]

    train_idxs = [id_to_idx[id] for id in train_ids if id in id_to_idx]
    valid_idxs = [id_to_idx[id] for id in valid_ids if id in id_to_idx]
    test_idxs = [id_to_idx[id] for id in test_ids if id in id_to_idx]

    if args.use_common_ids:
        common_idxs = set.intersection(*common_idx_list)
        train_idxs = list(common_idxs & set(train_idxs))
        valid_idxs = list(common_idxs & set(valid_idxs))
        test_idxs = list(common_idxs & set(test_idxs))

    # Remove rows where all modalities are missing (-2)
    def all_modalities_missing(idx):
        return all(
            data_dict[modality][idx, 0] == -2
            for modality in data_dict.keys()
            if modality != "modality_comb"
        )

    train_idxs = [idx for idx in train_idxs if not all_modalities_missing(idx)]
    mc_num_to_mc = {v: k for k, v in combination_to_index.items()}
    mc_idx_dict = {
        mc_num_to_mc[mc_num]: list(
            np.where(np.array(data_dict["modality_comb"]) == mc_num)[0]
        )
        for mc_num in set(data_dict["modality_comb"])
        if mc_num != -1
    }

    return (
        data_dict,
        encoder_dict,
        labels,
        train_idxs,
        valid_idxs,
        test_idxs,
        n_labels,
        input_dims,
        transforms,
        masks,
        observed_idx_arr,
        mc_idx_dict,
        mc_num_to_mc,
    )
