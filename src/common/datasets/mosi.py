import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import numpy as np
import pandas as pd
import torch
from itertools import combinations
import pickle

from src.common.modules.common import PatchEmbeddings, GRU
from src.common.utils import get_modality_combinations


def drop_entry(dataset):
    """Drop entries where there's no text in the data."""
    drop = []
    for ind, k in enumerate(dataset["text"]):
        if k.sum() == 0:
            drop.append(ind)
    # for ind, k in enumerate(dataset["vision"]):
    #     if k.sum() == 0:
    #         if ind not in drop:
    #             drop.append(ind)
    # for ind, k in enumerate(dataset["audio"]):
    #     if k.sum() == 0:
    #         if ind not in drop:
    #             drop.append(ind)

    for modality in list(dataset.keys()):
        dataset[modality] = np.delete(dataset[modality], drop, 0)
    return dataset


def load_and_preprocess_data_mosi_regression(args):
    filepath = "data/cmu-mosi/mosi_data.pkl"
    with open(filepath, "rb") as f:
        alldata = pickle.load(f)

    # alldata['train'].keys(): dict_keys(['vision', 'audio', 'text', 'labels', 'id'])
    alldata["train"] = drop_entry(alldata["train"])
    alldata["valid"] = drop_entry(alldata["valid"])
    alldata["test"] = drop_entry(alldata["test"])

    # breakpoint()

    train_labels = alldata["train"]["labels"].flatten()
    valid_labels = alldata["valid"]["labels"].flatten()
    test_labels = alldata["test"]["labels"].flatten()
    labels = np.concatenate((train_labels, valid_labels, test_labels))
    n_labels = 1

    train_ids = alldata["train"]["id"]
    val_ids = alldata["valid"]["id"]
    test_ids = alldata["test"]["id"]

    train_ids = ["".join(list(arr.astype(str))) for arr in train_ids]
    val_ids = ["".join(list(arr.astype(str))) for arr in val_ids]
    test_ids = ["".join(list(arr.astype(str))) for arr in test_ids]

    all_ids = train_ids + val_ids + test_ids

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}

    id_to_idx = {id: idx for idx, id in enumerate(all_ids)}
    common_idx_list = []
    observed_idx_arr = np.zeros((labels.shape[0], 3), dtype=bool)

    # Initialize modality combination list
    modality_combinations = [""] * len(id_to_idx)

    def update_modality_combinations(idx, modality):
        nonlocal modality_combinations
        if modality_combinations[idx] == "":
            modality_combinations[idx] = modality
        else:
            modality_combinations[idx] += modality

    # Load modalities
    if "V" in args.modality or "v" in args.modality:
        train_vision = alldata["train"]["vision"]
        train_vision = [
            np.nan_to_num(train_vision[i]) for i in range(train_vision.shape[0])
        ]
        valid_vision = alldata["valid"]["vision"]
        valid_vision = [
            np.nan_to_num(valid_vision[i]) for i in range(valid_vision.shape[0])
        ]
        test_vision = alldata["test"]["vision"]
        test_vision = [
            np.nan_to_num(test_vision[i]) for i in range(test_vision.shape[0])
        ]

        arr = train_vision + valid_vision + test_vision
        observed_idx_arr[:, 0] = [True] * len(arr)
        data_dict["vision"] = np.array(arr)
        filtered_idx = set(list(id_to_idx.values()))
        common_idx_list.append(filtered_idx)
        for idx in filtered_idx:
            update_modality_combinations(idx, "V")

        if args.patch:
            encoder_dict["vision"] = torch.nn.Sequential(
                GRU(
                    20,
                    256,
                    dropout=True,
                    has_padding=False,
                    batch_first=True,
                    last_only=True,
                ).cuda(),
                PatchEmbeddings(256, args.num_patches, args.hidden_dim).cuda(),
            )
        else:
            encoder_dict["vision"] = GRU(
                20,
                args.hidden_dim,
                dropout=True,
                has_padding=False,
                batch_first=True,
                last_only=True,
            ).cuda()
        input_dims["vision"] = arr[0].shape[1]
        # transforms['vision'] = None
        # masks['image'] = mask
    if "A" in args.modality or "a" in args.modality:
        train_audio = alldata["train"]["audio"]
        train_audio = [
            np.nan_to_num(train_audio[i]) for i in range(train_audio.shape[0])
        ]
        valid_audio = alldata["valid"]["audio"]
        valid_audio = [
            np.nan_to_num(valid_audio[i]) for i in range(valid_audio.shape[0])
        ]
        test_audio = alldata["test"]["audio"]
        test_audio = [np.nan_to_num(test_audio[i]) for i in range(test_audio.shape[0])]

        arr = train_audio + valid_audio + test_audio
        observed_idx_arr[:, 0] = [True] * len(arr)
        data_dict["audio"] = np.array(arr)
        filtered_idx = set(list(id_to_idx.values()))
        common_idx_list.append(filtered_idx)
        for idx in filtered_idx:
            update_modality_combinations(idx, "A")
        if args.patch:
            encoder_dict["audio"] = torch.nn.Sequential(
                GRU(
                    5,
                    256,
                    dropout=True,
                    has_padding=False,
                    batch_first=True,
                    last_only=True,
                ).cuda(),
                PatchEmbeddings(256, args.num_patches, args.hidden_dim).cuda(),
            )
        else:
            encoder_dict["audio"] = GRU(
                5,
                args.hidden_dim,
                dropout=True,
                has_padding=False,
                batch_first=True,
                last_only=True,
            ).cuda()
        input_dims["audio"] = arr[0].shape[1]
    if "T" in args.modality or "t" in args.modality:
        train_text = alldata["train"]["text"]
        train_text = [np.nan_to_num(train_text[i]) for i in range(train_text.shape[0])]
        valid_text = alldata["valid"]["text"]
        valid_text = [np.nan_to_num(valid_text[i]) for i in range(valid_text.shape[0])]
        test_text = alldata["test"]["text"]
        test_text = [np.nan_to_num(test_text[i]) for i in range(test_text.shape[0])]

        arr = train_text + valid_text + test_text
        observed_idx_arr[:, 0] = [True] * len(arr)
        data_dict["text"] = np.array(arr)
        filtered_idx = set(list(id_to_idx.values()))
        common_idx_list.append(filtered_idx)
        for idx in filtered_idx:
            update_modality_combinations(idx, "T")

        if args.patch:
            encoder_dict["text"] = torch.nn.Sequential(
                GRU(
                    300,
                    256,
                    dropout=True,
                    has_padding=False,
                    batch_first=True,
                    last_only=True,
                ).cuda(),
                PatchEmbeddings(256, args.num_patches, args.hidden_dim).cuda(),
            )
        else:
            encoder_dict["text"] = GRU(
                300,
                args.hidden_dim,
                dropout=True,
                has_padding=False,
                batch_first=True,
                last_only=True,
            ).cuda()
        input_dims["text"] = arr[0].shape[1]

    combination_to_index = get_modality_combinations(
        args.modality
    )  # 0: full modality index
    modality_combinations = [
        "".join(sorted(set(comb))) for comb in modality_combinations
    ]

    _keys = combination_to_index.keys()
    data_dict["modality_comb"] = [
        combination_to_index[comb] if comb in _keys else -1
        for comb in modality_combinations
    ]

    train_idxs = [id_to_idx[id] for id in train_ids if id in id_to_idx]
    valid_idxs = [id_to_idx[id] for id in val_ids if id in id_to_idx]
    test_idxs = [id_to_idx[id] for id in test_ids if id in id_to_idx]

    # Remove rows where all modalities are missing (-2)
    def all_modalities_missing(idx):
        return data_dict["modality_comb"][idx] == -1

    train_idxs = [idx for idx in train_idxs if not all_modalities_missing(idx)]
    valid_idxs = [idx for idx in valid_idxs if not all_modalities_missing(idx)]
    test_idxs = [idx for idx in test_idxs if not all_modalities_missing(idx)]

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


def load_and_preprocess_data_mosi(args):
    filepath = "data/cmu-mosi/mosi_data.pkl"
    with open(filepath, "rb") as f:
        alldata = pickle.load(f)

    # alldata['train'].keys(): dict_keys(['vision', 'audio', 'text', 'labels', 'id'])
    alldata["train"] = drop_entry(alldata["train"])
    alldata["valid"] = drop_entry(alldata["valid"])
    alldata["test"] = drop_entry(alldata["test"])

    train_labels = alldata["train"]["labels"].flatten()
    train_labels = np.array([0 if label <= 0 else 1 for label in train_labels])
    valid_labels = alldata["valid"]["labels"].flatten()
    valid_labels = np.array([0 if label <= 0 else 1 for label in valid_labels])
    test_labels = alldata["test"]["labels"].flatten()
    test_labels = np.array([0 if label <= 0 else 1 for label in test_labels])
    labels = np.concatenate((train_labels, valid_labels, test_labels))
    n_labels = len(set(labels))

    train_ids = alldata["train"]["id"]
    val_ids = alldata["valid"]["id"]
    test_ids = alldata["test"]["id"]

    train_ids = ["".join(list(arr.astype(str))) for arr in train_ids]
    val_ids = ["".join(list(arr.astype(str))) for arr in val_ids]
    test_ids = ["".join(list(arr.astype(str))) for arr in test_ids]

    all_ids = train_ids + val_ids + test_ids

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}

    id_to_idx = {id: idx for idx, id in enumerate(all_ids)}
    common_idx_list = []
    observed_idx_arr = np.zeros((labels.shape[0], 3), dtype=bool)

    # Initialize modality combination list
    modality_combinations = [""] * len(id_to_idx)

    def update_modality_combinations(idx, modality):
        nonlocal modality_combinations
        if modality_combinations[idx] == "":
            modality_combinations[idx] = modality
        else:
            modality_combinations[idx] += modality

    # Load modalities
    if "V" in args.modality or "v" in args.modality:
        train_vision = alldata["train"]["vision"]
        train_vision = [
            np.nan_to_num(train_vision[i]) for i in range(train_vision.shape[0])
        ]
        valid_vision = alldata["valid"]["vision"]
        valid_vision = [
            np.nan_to_num(valid_vision[i]) for i in range(valid_vision.shape[0])
        ]
        test_vision = alldata["test"]["vision"]
        test_vision = [
            np.nan_to_num(test_vision[i]) for i in range(test_vision.shape[0])
        ]

        arr = train_vision + valid_vision + test_vision
        observed_idx_arr[:, 0] = [True] * len(arr)
        data_dict["vision"] = np.array(arr)
        filtered_idx = set(list(id_to_idx.values()))
        common_idx_list.append(filtered_idx)
        for idx in filtered_idx:
            update_modality_combinations(idx, "V")
        if args.patch:
            encoder_dict["vision"] = torch.nn.Sequential(
                GRU(
                    20,
                    256,
                    dropout=True,
                    has_padding=False,
                    batch_first=True,
                    last_only=True,
                ).cuda(),
                PatchEmbeddings(256, args.num_patches, args.hidden_dim).cuda(),
            )
        else:
            encoder_dict["vision"] = GRU(
                20,
                args.hidden_dim,
                dropout=True,
                has_padding=False,
                batch_first=True,
                last_only=True,
            ).cuda()
        input_dims["vision"] = arr[0].shape[1]

    if "A" in args.modality or "a" in args.modality:
        train_audio = alldata["train"]["audio"]
        train_audio = [
            np.nan_to_num(train_audio[i]) for i in range(train_audio.shape[0])
        ]
        valid_audio = alldata["valid"]["audio"]
        valid_audio = [
            np.nan_to_num(valid_audio[i]) for i in range(valid_audio.shape[0])
        ]
        test_audio = alldata["test"]["audio"]
        test_audio = [np.nan_to_num(test_audio[i]) for i in range(test_audio.shape[0])]

        arr = train_audio + valid_audio + test_audio
        observed_idx_arr[:, 1] = [True] * len(arr)
        data_dict["audio"] = np.array(arr)
        filtered_idx = set(list(id_to_idx.values()))
        common_idx_list.append(filtered_idx)
        for idx in filtered_idx:
            update_modality_combinations(idx, "A")
        if args.patch:
            encoder_dict["audio"] = torch.nn.Sequential(
                GRU(
                    5,
                    256,
                    dropout=True,
                    has_padding=False,
                    batch_first=True,
                    last_only=True,
                ).cuda(),
                PatchEmbeddings(256, args.num_patches, args.hidden_dim).cuda(),
            )
        else:
            encoder_dict["audio"] = GRU(
                5,
                args.hidden_dim,
                dropout=True,
                has_padding=False,
                batch_first=True,
                last_only=True,
            ).cuda()
        input_dims["audio"] = arr[0].shape[1]
    if "T" in args.modality or "t" in args.modality:
        train_text = alldata["train"]["text"]
        train_text = [np.nan_to_num(train_text[i]) for i in range(train_text.shape[0])]
        valid_text = alldata["valid"]["text"]
        valid_text = [np.nan_to_num(valid_text[i]) for i in range(valid_text.shape[0])]
        test_text = alldata["test"]["text"]
        test_text = [np.nan_to_num(test_text[i]) for i in range(test_text.shape[0])]

        arr = train_text + valid_text + test_text
        observed_idx_arr[:, 2] = [True] * len(arr)
        data_dict["text"] = np.array(arr)
        filtered_idx = set(list(id_to_idx.values()))
        common_idx_list.append(filtered_idx)
        for idx in filtered_idx:
            update_modality_combinations(idx, "T")
        if args.patch:
            encoder_dict["text"] = torch.nn.Sequential(
                GRU(
                    300,
                    256,
                    dropout=True,
                    has_padding=False,
                    batch_first=True,
                    last_only=True,
                ).cuda(),
                PatchEmbeddings(256, args.num_patches, args.hidden_dim).cuda(),
            )
        else:
            encoder_dict["text"] = GRU(
                300,
                args.hidden_dim,
                dropout=True,
                has_padding=False,
                batch_first=True,
                last_only=True,
            ).cuda()
        input_dims["text"] = arr[0].shape[1]

    combination_to_index = get_modality_combinations(
        args.modality
    )  # 0: full modality index
    modality_combinations = [
        "".join(sorted(set(comb))) for comb in modality_combinations
    ]

    _keys = combination_to_index.keys()
    data_dict["modality_comb"] = [
        combination_to_index[comb] if comb in _keys else -1
        for comb in modality_combinations
    ]

    train_idxs = [id_to_idx[id] for id in train_ids if id in id_to_idx]
    valid_idxs = [id_to_idx[id] for id in val_ids if id in id_to_idx]
    test_idxs = [id_to_idx[id] for id in test_ids if id in id_to_idx]

    # Remove rows where all modalities are missing (-2)
    def all_modalities_missing(idx):
        return data_dict["modality_comb"][idx] == -1

    train_idxs = [idx for idx in train_idxs if not all_modalities_missing(idx)]
    valid_idxs = [idx for idx in valid_idxs if not all_modalities_missing(idx)]
    test_idxs = [idx for idx in test_idxs if not all_modalities_missing(idx)]

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


def load_and_preprocess_data_humor(args):
    filepath = "data/humor/humor.pkl"
    with open(filepath, "rb") as f:
        alldata = pickle.load(f)

    # alldata['train'].keys(): dict_keys(['vision', 'audio', 'text', 'labels', 'id'])
    alldata["train"] = drop_entry(alldata["train"])
    alldata["valid"] = drop_entry(alldata["valid"])
    alldata["test"] = drop_entry(alldata["test"])

    train_labels = alldata["train"]["labels"].flatten()
    train_labels = np.array([0 if label <= 0 else 1 for label in train_labels])
    valid_labels = alldata["valid"]["labels"].flatten()
    valid_labels = np.array([0 if label <= 0 else 1 for label in valid_labels])
    test_labels = alldata["test"]["labels"].flatten()
    test_labels = np.array([0 if label <= 0 else 1 for label in test_labels])
    labels = np.concatenate((train_labels, valid_labels, test_labels))
    n_labels = len(set(labels))

    train_ids = alldata["train"]["id"]
    val_ids = alldata["valid"]["id"]
    test_ids = alldata["test"]["id"]

    train_ids = ["".join(list(arr.astype(str))) for arr in train_ids]
    val_ids = ["".join(list(arr.astype(str))) for arr in val_ids]
    test_ids = ["".join(list(arr.astype(str))) for arr in test_ids]

    all_ids = train_ids + val_ids + test_ids

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}

    id_to_idx = {id: idx for idx, id in enumerate(all_ids)}
    common_idx_list = []
    observed_idx_arr = np.zeros((labels.shape[0], 3), dtype=bool)

    # Initialize modality combination list
    modality_combinations = [""] * len(id_to_idx)

    def update_modality_combinations(idx, modality):
        nonlocal modality_combinations
        if modality_combinations[idx] == "":
            modality_combinations[idx] = modality
        else:
            modality_combinations[idx] += modality

    # Load modalities
    if "V" in args.modality or "v" in args.modality:
        train_vision = alldata["train"]["vision"]
        train_vision = [
            np.nan_to_num(train_vision[i]) for i in range(train_vision.shape[0])
        ]
        valid_vision = alldata["valid"]["vision"]
        valid_vision = [
            np.nan_to_num(valid_vision[i]) for i in range(valid_vision.shape[0])
        ]
        test_vision = alldata["test"]["vision"]
        test_vision = [
            np.nan_to_num(test_vision[i]) for i in range(test_vision.shape[0])
        ]

        arr = train_vision + valid_vision + test_vision
        observed_idx_arr[:, 0] = [True] * len(arr)
        data_dict["vision"] = np.array(arr)
        filtered_idx = set(list(id_to_idx.values()))
        common_idx_list.append(filtered_idx)
        for idx in filtered_idx:
            update_modality_combinations(idx, "V")
        if args.patch:
            encoder_dict["vision"] = torch.nn.Sequential(
                GRU(
                    371,
                    256,
                    dropout=True,
                    has_padding=False,
                    batch_first=True,
                    last_only=True,
                ).cuda(),
                PatchEmbeddings(256, args.num_patches, args.hidden_dim).cuda(),
            )
        else:
            encoder_dict["vision"] = GRU(
                371,
                256,
                dropout=True,
                has_padding=False,
                batch_first=True,
                last_only=True,
            ).cuda()
        input_dims["vision"] = arr[0].shape[1]

    if "A" in args.modality or "a" in args.modality:
        train_audio = alldata["train"]["audio"]
        train_audio = [
            np.nan_to_num(train_audio[i]) for i in range(train_audio.shape[0])
        ]
        valid_audio = alldata["valid"]["audio"]
        valid_audio = [
            np.nan_to_num(valid_audio[i]) for i in range(valid_audio.shape[0])
        ]
        test_audio = alldata["test"]["audio"]
        test_audio = [np.nan_to_num(test_audio[i]) for i in range(test_audio.shape[0])]

        arr = train_audio + valid_audio + test_audio
        observed_idx_arr[:, 1] = [True] * len(arr)
        data_dict["audio"] = np.array(arr)
        filtered_idx = set(list(id_to_idx.values()))
        common_idx_list.append(filtered_idx)
        for idx in filtered_idx:
            update_modality_combinations(idx, "A")
        if args.patch:
            encoder_dict["audio"] = torch.nn.Sequential(
                GRU(
                    81,
                    256,
                    dropout=True,
                    has_padding=False,
                    batch_first=True,
                    last_only=True,
                ).cuda(),
                PatchEmbeddings(256, args.num_patches, args.hidden_dim).cuda(),
            )
        else:
            encoder_dict["audio"] = GRU(
                81,
                256,
                dropout=True,
                has_padding=False,
                batch_first=True,
                last_only=True,
            ).cuda()
        input_dims["audio"] = arr[0].shape[1]
    if "T" in args.modality or "t" in args.modality:
        train_text = alldata["train"]["text"]
        train_text = [np.nan_to_num(train_text[i]) for i in range(train_text.shape[0])]
        valid_text = alldata["valid"]["text"]
        valid_text = [np.nan_to_num(valid_text[i]) for i in range(valid_text.shape[0])]
        test_text = alldata["test"]["text"]
        test_text = [np.nan_to_num(test_text[i]) for i in range(test_text.shape[0])]

        arr = train_text + valid_text + test_text
        observed_idx_arr[:, 2] = [True] * len(arr)
        data_dict["text"] = np.array(arr)
        filtered_idx = set(list(id_to_idx.values()))
        common_idx_list.append(filtered_idx)
        for idx in filtered_idx:
            update_modality_combinations(idx, "T")
        if args.patch:
            encoder_dict["text"] = torch.nn.Sequential(
                GRU(
                    300,
                    256,
                    dropout=True,
                    has_padding=False,
                    batch_first=True,
                    last_only=True,
                ).cuda(),
                PatchEmbeddings(256, args.num_patches, args.hidden_dim).cuda(),
            )
        else:
            encoder_dict["text"] = GRU(
                300,
                256,
                dropout=True,
                has_padding=False,
                batch_first=True,
                last_only=True,
            ).cuda()
        input_dims["text"] = arr[0].shape[1]

    combination_to_index = get_modality_combinations(
        args.modality
    )  # 0: full modality index
    modality_combinations = [
        "".join(sorted(set(comb))) for comb in modality_combinations
    ]

    _keys = combination_to_index.keys()
    data_dict["modality_comb"] = [
        combination_to_index[comb] if comb in _keys else -1
        for comb in modality_combinations
    ]

    train_idxs = [id_to_idx[id] for id in train_ids if id in id_to_idx]
    valid_idxs = [id_to_idx[id] for id in val_ids if id in id_to_idx]
    test_idxs = [id_to_idx[id] for id in test_ids if id in id_to_idx]

    # Remove rows where all modalities are missing (-2)
    def all_modalities_missing(idx):
        return data_dict["modality_comb"][idx] == -1

    train_idxs = [idx for idx in train_idxs if not all_modalities_missing(idx)]
    valid_idxs = [idx for idx in valid_idxs if not all_modalities_missing(idx)]
    test_idxs = [idx for idx in test_idxs if not all_modalities_missing(idx)]

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


def load_and_preprocess_data_sarcasm(args):
    filepath = "data/mustard/sarcasm.pkl"
    with open(filepath, "rb") as f:
        alldata = pickle.load(f)

    # alldata['train'].keys(): dict_keys(['vision', 'audio', 'text', 'labels', 'id'])
    alldata["train"] = drop_entry(alldata["train"])
    alldata["valid"] = drop_entry(alldata["valid"])
    alldata["test"] = drop_entry(alldata["test"])

    train_labels = alldata["train"]["labels"].flatten()
    train_labels = np.array([0 if label <= 0 else 1 for label in train_labels])
    valid_labels = alldata["valid"]["labels"].flatten()
    valid_labels = np.array([0 if label <= 0 else 1 for label in valid_labels])
    test_labels = alldata["test"]["labels"].flatten()
    test_labels = np.array([0 if label <= 0 else 1 for label in test_labels])
    labels = np.concatenate((train_labels, valid_labels, test_labels))
    n_labels = len(set(labels))

    train_ids = alldata["train"]["id"]
    val_ids = alldata["valid"]["id"]
    test_ids = alldata["test"]["id"]

    train_ids = ["".join(list(arr.astype(str))) for arr in train_ids]
    val_ids = ["".join(list(arr.astype(str))) for arr in val_ids]
    test_ids = ["".join(list(arr.astype(str))) for arr in test_ids]

    all_ids = train_ids + val_ids + test_ids

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}

    id_to_idx = {id: idx for idx, id in enumerate(all_ids)}
    common_idx_list = []
    observed_idx_arr = np.zeros((labels.shape[0], 3), dtype=bool)

    # Initialize modality combination list
    modality_combinations = [""] * len(id_to_idx)

    def update_modality_combinations(idx, modality):
        nonlocal modality_combinations
        if modality_combinations[idx] == "":
            modality_combinations[idx] = modality
        else:
            modality_combinations[idx] += modality

    # Load modalities
    if "V" in args.modality or "v" in args.modality:
        train_vision = alldata["train"]["vision"]
        train_vision = [
            np.nan_to_num(train_vision[i]) for i in range(train_vision.shape[0])
        ]
        valid_vision = alldata["valid"]["vision"]
        valid_vision = [
            np.nan_to_num(valid_vision[i]) for i in range(valid_vision.shape[0])
        ]
        test_vision = alldata["test"]["vision"]
        test_vision = [
            np.nan_to_num(test_vision[i]) for i in range(test_vision.shape[0])
        ]

        arr = train_vision + valid_vision + test_vision
        observed_idx_arr[:, 0] = [True] * len(arr)
        data_dict["vision"] = np.array(arr)
        filtered_idx = set(list(id_to_idx.values()))
        common_idx_list.append(filtered_idx)
        for idx in filtered_idx:
            update_modality_combinations(idx, "V")
        if args.patch:
            encoder_dict["vision"] = torch.nn.Sequential(
                GRU(
                    371,
                    256,
                    dropout=True,
                    has_padding=False,
                    batch_first=True,
                    last_only=True,
                ).cuda(),
                PatchEmbeddings(256, args.num_patches, args.hidden_dim).cuda(),
            )
        else:
            encoder_dict["vision"] = GRU(
                371,
                256,
                dropout=True,
                has_padding=False,
                batch_first=True,
                last_only=True,
            ).cuda()
        input_dims["vision"] = arr[0].shape[1]

    if "A" in args.modality or "a" in args.modality:
        train_audio = alldata["train"]["audio"]
        train_audio = [
            np.nan_to_num(train_audio[i]) for i in range(train_audio.shape[0])
        ]
        valid_audio = alldata["valid"]["audio"]
        valid_audio = [
            np.nan_to_num(valid_audio[i]) for i in range(valid_audio.shape[0])
        ]
        test_audio = alldata["test"]["audio"]
        test_audio = [np.nan_to_num(test_audio[i]) for i in range(test_audio.shape[0])]

        arr = train_audio + valid_audio + test_audio
        observed_idx_arr[:, 1] = [True] * len(arr)
        data_dict["audio"] = np.array(arr)
        filtered_idx = set(list(id_to_idx.values()))
        common_idx_list.append(filtered_idx)
        for idx in filtered_idx:
            update_modality_combinations(idx, "A")
        if args.patch:
            encoder_dict["audio"] = torch.nn.Sequential(
                GRU(
                    81,
                    256,
                    dropout=True,
                    has_padding=False,
                    batch_first=True,
                    last_only=True,
                ).cuda(),
                PatchEmbeddings(256, args.num_patches, args.hidden_dim).cuda(),
            )
        else:
            encoder_dict["audio"] = GRU(
                81,
                256,
                dropout=True,
                has_padding=False,
                batch_first=True,
                last_only=True,
            ).cuda()
        input_dims["audio"] = arr[0].shape[1]
    if "T" in args.modality or "t" in args.modality:
        train_text = alldata["train"]["text"]
        train_text = [np.nan_to_num(train_text[i]) for i in range(train_text.shape[0])]
        valid_text = alldata["valid"]["text"]
        valid_text = [np.nan_to_num(valid_text[i]) for i in range(valid_text.shape[0])]
        test_text = alldata["test"]["text"]
        test_text = [np.nan_to_num(test_text[i]) for i in range(test_text.shape[0])]

        arr = train_text + valid_text + test_text
        observed_idx_arr[:, 2] = [True] * len(arr)
        data_dict["text"] = np.array(arr)
        filtered_idx = set(list(id_to_idx.values()))
        common_idx_list.append(filtered_idx)
        for idx in filtered_idx:
            update_modality_combinations(idx, "T")
        if args.patch:
            encoder_dict["text"] = torch.nn.Sequential(
                GRU(
                    300,
                    256,
                    dropout=True,
                    has_padding=False,
                    batch_first=True,
                    last_only=True,
                ).cuda(),
                PatchEmbeddings(256, args.num_patches, args.hidden_dim).cuda(),
            )
        else:
            encoder_dict["text"] = GRU(
                300,
                256,
                dropout=True,
                has_padding=False,
                batch_first=True,
                last_only=True,
            ).cuda()
        input_dims["text"] = arr[0].shape[1]

    combination_to_index = get_modality_combinations(
        args.modality
    )  # 0: full modality index
    modality_combinations = [
        "".join(sorted(set(comb))) for comb in modality_combinations
    ]

    _keys = combination_to_index.keys()
    data_dict["modality_comb"] = [
        combination_to_index[comb] if comb in _keys else -1
        for comb in modality_combinations
    ]

    train_idxs = [id_to_idx[id] for id in train_ids if id in id_to_idx]
    valid_idxs = [id_to_idx[id] for id in val_ids if id in id_to_idx]
    test_idxs = [id_to_idx[id] for id in test_ids if id in id_to_idx]

    # Remove rows where all modalities are missing (-2)
    def all_modalities_missing(idx):
        return data_dict["modality_comb"][idx] == -1

    train_idxs = [idx for idx in train_idxs if not all_modalities_missing(idx)]
    valid_idxs = [idx for idx in valid_idxs if not all_modalities_missing(idx)]
    test_idxs = [idx for idx in test_idxs if not all_modalities_missing(idx)]

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
