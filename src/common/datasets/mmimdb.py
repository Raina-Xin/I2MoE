import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import numpy as np
import pandas as pd
import torch
from itertools import combinations
import h5py
import json

from src.common.modules.common import PatchEmbeddings, MaxOut_MLP, Linear
from src.common.utils import get_modality_combinations


def load_and_preprocess_data_mmimdb(args):
    data_dir = "data/mm-imdb"
    hdf5_file = os.path.join(data_dir, "multimodal_imdb.hdf5")
    dataset = h5py.File(hdf5_file, "r")

    # text = dataset["features"][ind+self.start_ind]
    # image = dataset["vgg_features"][ind+self.start_ind]
    # # self.dataset["images"][ind+self.start_ind] if not self.vggfeature
    # label = dataset["genres"][ind+self.start_ind]

    # keys = list(range(len(dataset["features"])))
    keys = [str(int(imdb_id)).zfill(7) for imdb_id in dataset["imdb_ids"]]

    # train split is at the front
    train_start_index = 0
    if args.debug:
        train_stop_index = 1000
    else:
        train_stop_index = 15552
    train_keys = keys[train_start_index:train_stop_index]

    # val split is in the middle
    if args.debug:
        val_start_index = 1000
        val_stop_index = 1400
    else:
        val_start_index = 15552
        val_stop_index = 18160
    val_keys = keys[val_start_index:val_stop_index]

    # test split is at the end
    if args.debug:
        test_start_index = 1400
        test_stop_index = 2000
    else:
        test_start_index = 18160
        test_stop_index = 25959
    test_keys = keys[test_start_index:test_stop_index]

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}

    id_to_idx = {id: idx for idx, id in enumerate(keys)}
    idx_to_id = {idx: id for id, idx in id_to_idx.items()}
    common_idx_list = []
    observed_idx_arr = np.zeros((len(keys), 2), dtype=bool)  # L, I order

    labels = dataset["genres"]
    n_labels = 23

    # Load modalities
    if ("L" in args.modality) and ("I" in args.modality):
        # Initialize modality combination list
        modality_combinations = ["LI"] * len(id_to_idx)
        observed_idx_arr[:, 0] = [True] * len(labels)
        observed_idx_arr[:, 1] = [True] * len(labels)

        data_dict["language"] = dataset["features"]
        data_dict["img"] = dataset["vgg_features"]

        filtered_idx = set(list(id_to_idx.values()))
        common_idx_list.append(filtered_idx)

        if args.patch:
            encoder_dict["language"] = torch.nn.Sequential(
                MaxOut_MLP(512, 512, 300, linear_layer=False).to(args.device),
                PatchEmbeddings(
                    512, num_patches=args.num_patches, embed_dim=args.hidden_dim
                ).to(args.device),
            )
            encoder_dict["img"] = torch.nn.Sequential(
                # MaxOut_MLP(512, 512, 4096, linear_layer=False).to(args.device),
                MaxOut_MLP(512, 1024, 4096, 512, False).to(args.device),
                PatchEmbeddings(
                    512, num_patches=args.num_patches, embed_dim=args.hidden_dim
                ).to(args.device),
            )
            input_dims["language"] = 512
            input_dims["img"] = 512
        else:
            encoder_dict["language"] = MaxOut_MLP(
                args.hidden_dim, 512, 300, linear_layer=True
            ).to(args.device)
            encoder_dict["img"] = MaxOut_MLP(
                args.hidden_dim, 1024, 4096, linear_layer=True
            ).to(args.device)
            input_dims["language"] = 300
            input_dims["img"] = 4096

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

    train_idxs = [id_to_idx[id] for id in train_keys if id in id_to_idx]
    valid_idxs = [id_to_idx[id] for id in val_keys if id in id_to_idx]
    test_idxs = [id_to_idx[id] for id in test_keys if id in id_to_idx]

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

    test_idx_to_mmimdb_ids = {idx: idx_to_id[idx] for idx in test_idxs}
    # Save to a file
    with open("outputs/test_idx_to_mmimdb_ids.json", "w") as f:
        json.dump(test_idx_to_mmimdb_ids, f)

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
