import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
import nibabel as nib
import json
from torchvision.transforms import Compose, ToTensor, Normalize
import scanpy as sc

from src.common.modules.common import Custom3DCNN, PatchEmbeddings, MLP
from src.common.utils import get_modality_combinations


def load_and_preprocess_image_data(image_path, label_df, id_to_idx):
    # Load and preprocess image data
    image_data = np.load(os.path.join(image_path, "ADNI_G.npy"), mmap_mode="r")
    mask_path = os.path.join(
        image_path, "BLSA_SPGR+MPRAGE_averagetemplate_muse_seg_DS222.nii.gz"
    )

    subject_ids = []
    dates = []
    with open("data/adni/image/ADNI_subj.txt", "r") as file:
        for line in file:
            line = line.strip()
            parts = line.split("_")
            subject_id = "_".join(parts[:3])
            date = parts[-1]
            subject_ids.append(subject_id)
            dates.append(date)

    df = pd.DataFrame({"PTID": subject_ids, "date": dates})

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date", ascending=False)
    idx = df.groupby("PTID")["date"].idxmax()

    # Creating the subset DataFrame using the indexes
    subdf = df.loc[idx]
    subdf = subdf.sort_index()
    subdf = subdf.reset_index()

    merged_df = pd.merge(subdf, label_df, on="PTID", how="left")

    image_data = image_data[merged_df["index"]]
    final_subject_ids = list(subdf.PTID)

    new_idx = np.array(convert_ids_to_index(final_subject_ids, id_to_idx))
    filtered_idx = [x for x in new_idx if x != -1]
    tmp = np.zeros((len(id_to_idx), image_data.shape[1])) - 2
    tmp[filtered_idx] = image_data[np.array(new_idx) != -1]

    data = nib.load(mask_path).get_fdata()
    mean = image_data.mean()
    std = image_data.std()
    # mean = data.mean()
    # std = data.std()
    mask_gm = (data == 150).ravel()

    return tmp, filtered_idx, mean, std, mask_gm


def convert_ids_to_index(ids, index_map):
    return [index_map[id] if id in index_map else -1 for id in ids]


def load_and_preprocess_data_adni(args):
    # Paths
    image_path = "data/adni/image"
    genomic_path = "data/adni/genomic/genomic_merged.h5ad"
    clinical_path = "data/adni/clinical/clinical_merged"
    biospecimen_path = "data/adni/biospecimen/biospecimen_merged"
    label_df = pd.read_csv("data/adni/label.csv", index_col="PTID")
    label_df["DIAGNOSIS"] -= 1
    labels = label_df["DIAGNOSIS"].values.astype(np.int64)
    n_labels = len(set(labels))

    with open("data/adni/PTID_splits.json") as json_file:
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
    if "I" in args.modality or "i" in args.modality:
        arr, filtered_idx, mean, std, mask = load_and_preprocess_image_data(
            image_path, label_df, id_to_idx
        )
        observed_idx_arr[:, 0] = arr[:, 0] != -2
        data_dict["image"] = np.array(arr)
        common_idx_list.append(set(filtered_idx))
        for idx in filtered_idx:
            update_modality_combinations(idx, "I")
        if args.patch:
            encoder_dict["image"] = torch.nn.Sequential(
                Custom3DCNN(hidden_dim=args.hidden_dim).to(args.device),
                PatchEmbeddings(
                    feature_size=args.hidden_dim,
                    num_patches=args.num_patches,
                    embed_dim=args.hidden_dim,
                ).to(args.device),
            )
        else:
            encoder_dict["image"] = Custom3DCNN(hidden_dim=args.hidden_dim).to(
                args.device
            )
        input_dims["image"] = arr.shape[1]
        transforms["image"] = Compose(
            [
                ToTensor(),
                Normalize(mean=[mean], std=[std]),
            ]
        )
        masks["image"] = mask

    if "G" in args.modality or "g" in args.modality:
        df = sc.read_h5ad(genomic_path).to_df()
        if args.initial_filling == "mean":
            df = df.apply(lambda x: x.fillna(x.mode().iloc[0]), axis=0)
        arr = df.values
        scaler = MinMaxScaler(feature_range=(-1, 1))
        arr = scaler.fit_transform(arr)
        new_idx = np.array(convert_ids_to_index(df.index, id_to_idx))
        filtered_idx = new_idx[new_idx != -1]
        for idx in filtered_idx:
            update_modality_combinations(idx, "G")
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]

        observed_idx_arr[filtered_idx, 1] = True
        data_dict["genomic"] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        if args.patch:
            encoder_dict["genomic"] = PatchEmbeddings(
                df.shape[1], args.num_patches, args.hidden_dim
            ).to(args.device)
        else:
            encoder_dict["genomic"] = MLP(
                df.shape[1], args.hidden_dim, args.hidden_dim, args.num_layers_enc
            ).to(args.device)
        input_dims["genomic"] = df.shape[1]

    if "C" in args.modality or "c" in args.modality:
        if args.initial_filling == "mean":
            path = clinical_path + "_mean.csv"
        else:
            path = clinical_path + ".csv"
        df = pd.read_csv(path, index_col=0)
        columns_to_exclude = [
            col
            for col in df.columns
            if col.startswith("PTCOGBEG")
            or col.startswith("PTADDX")
            or col.startswith("PTADBEG")
        ]
        if len(columns_to_exclude) > 0:
            df = df.drop(columns_to_exclude, axis=1)
        arr = df.values.astype(np.float32)
        new_idx = np.array(convert_ids_to_index(df.index, id_to_idx))
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, 2] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, "C")
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]

        data_dict["clinical"] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        if args.patch:
            encoder_dict["clinical"] = PatchEmbeddings(
                df.shape[1], args.num_patches, args.hidden_dim
            ).to(args.device)
        else:
            encoder_dict["clinical"] = MLP(
                df.shape[1], args.hidden_dim, args.hidden_dim, args.num_layers_enc
            ).to(args.device)
        input_dims["clinical"] = df.shape[1]

    if "B" in args.modality or "b" in args.modality:
        if args.initial_filling == "mean":
            path = biospecimen_path + "_mean.csv"
        else:
            path = biospecimen_path + ".csv"
        df = pd.read_csv(path, index_col=0)
        arr = df.values
        new_idx = np.array(convert_ids_to_index(df.index, id_to_idx))
        filtered_idx = new_idx[new_idx != -1]
        for idx in filtered_idx:
            update_modality_combinations(idx, "B")
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]
        observed_idx_arr[filtered_idx, 3] = True
        data_dict["biospecimen"] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        if args.patch:
            encoder_dict["biospecimen"] = PatchEmbeddings(
                df.shape[1], args.num_patches, args.hidden_dim
            ).to(args.device)
        else:
            encoder_dict["biospecimen"] = MLP(
                df.shape[1], args.hidden_dim, args.hidden_dim, args.num_layers_enc
            ).to(args.device)
        input_dims["biospecimen"] = df.shape[1]

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
