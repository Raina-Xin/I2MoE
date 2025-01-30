import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import numpy as np
import pandas as pd
import torch
from itertools import combinations
from PIL import Image
import random
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
    Resize,
    RandomHorizontalFlip,
    RandomCrop,
    GaussianBlur,
    RandomVerticalFlip,
)
import csv

from src.common.modules.common import PatchEmbeddings, VGG11Slim, Linear
from src.common.utils import get_modality_combinations


def load_and_preprocess_data_enrico(args):
    data_dir = "data/enrico"
    csv_file = os.path.join(data_dir, "design_topics.csv")
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        example_list = list(reader)
    # self.img_missing_rate = img_missing_rate
    # self.wireframe_missing_rate = wireframe_missing_rate
    img_dim_x = 128
    img_dim_y = 256
    random_seed = 42
    train_split = 0.65
    val_split = 0.15
    test_split = 0.2
    normalize_image = False
    seq_len = 64
    csv_file = os.path.join(data_dir, "design_topics.csv")
    img_dir = os.path.join(data_dir, "screenshots")
    wireframe_dir = os.path.join(data_dir, "wireframes")
    hierarchy_dir = os.path.join(data_dir, "hierarchies")

    # the wireframe files are corrupted for these files
    IGNORES = set(["50105", "50109"])
    example_list = [e for e in example_list if e["screen_id"] not in IGNORES]

    keys = list(range(len(example_list)))
    # shuffle and create splits
    random.Random(random_seed).shuffle(keys)

    # train split is at the front
    train_start_index = 0
    train_stop_index = int(len(example_list) * train_split)
    train_keys = keys[train_start_index:train_stop_index]

    # val split is in the middle
    val_start_index = int(len(example_list) * train_split)
    val_stop_index = int(len(example_list) * (train_split + val_split))
    val_keys = keys[val_start_index:val_stop_index]

    # test split is at the end
    test_start_index = int(len(example_list) * (train_split + val_split))
    test_stop_index = len(example_list)
    test_keys = keys[test_start_index:test_stop_index]

    # Define image transformations
    img_transforms_train = Compose(
        [
            ToTensor(),
            Resize((img_dim_y, img_dim_x)),
            RandomCrop((img_dim_y, img_dim_x)),
            RandomHorizontalFlip(),
            GaussianBlur(3),
            RandomVerticalFlip(),
        ]
    )

    img_transforms_val_test = Compose(
        [ToTensor(), Resize((img_dim_y, img_dim_x))]  # Resize for validation and test
    )

    if normalize_image:
        img_transforms_train.append(Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        img_transforms_val_test.append(Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    # make maps
    topics = set()
    for e in example_list:
        topics.add(e["topic"])
    topics = sorted(list(topics))

    idx2Topic = {}
    topic2Idx = {}

    for i in range(len(topics)):
        idx2Topic[i] = topics[i]
        topic2Idx[topics[i]] = i

    idx2Topic = idx2Topic
    topic2Idx = topic2Idx

    UI_TYPES = [
        "Text",
        "Text Button",
        "Icon",
        "Card",
        "Drawer",
        "Web View",
        "List Item",
        "Toolbar",
        "Bottom Navigation",
        "Multi-Tab",
        "List Item",
        "Toolbar",
        "Bottom Navigation",
        "Multi-Tab",
        "Background Image",
        "Image",
        "Video",
        "Input",
        "Number Stepper",
        "Checkbox",
        "Radio Button",
        "Pager Indicator",
        "On/Off Switch",
        "Modal",
        "Slider",
        "Advertisement",
        "Date Picker",
        "Map View",
    ]

    idx2Label = {}
    label2Idx = {}

    for i in range(len(UI_TYPES)):
        idx2Label[i] = UI_TYPES[i]
        label2Idx[UI_TYPES[i]] = i

    ui_types = UI_TYPES

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}

    id_to_idx = {id: idx for idx, id in enumerate(keys)}
    common_idx_list = []
    observed_idx_arr = np.zeros((len(keys), 2), dtype=bool)  # IGCB order

    # Initialize modality combination list
    modality_combinations = [""] * len(id_to_idx)

    def update_modality_combinations(idx, modality):
        nonlocal modality_combinations
        if modality_combinations[idx] == "":
            modality_combinations[idx] = modality
        else:
            modality_combinations[idx] += modality

    # Load modalities
    if ("S" in args.modality) and ("W" in args.modality):
        s_list = []
        w_list = []
        label_list = []
        for idx in range(len(keys)):
            example = example_list[keys[idx]]
            screenId = example["screen_id"]
            # image modality
            # Load and transform images

            screenImg = Image.open(os.path.join(img_dir, screenId + ".jpg")).convert(
                "RGB"
            )
            screenImg = img_transforms_train(screenImg)
            screenWireframeImg = Image.open(
                os.path.join(wireframe_dir, screenId + ".png")
            ).convert("RGB")
            if idx in train_keys:
                # Apply data augmentation for training data
                screenWireframeImg = img_transforms_train(screenWireframeImg)
            else:
                # Apply only resizing for validation/test data
                screenWireframeImg = img_transforms_val_test(screenWireframeImg)

            screenLabel = topic2Idx[example["topic"]]

            update_modality_combinations(idx, "S")
            update_modality_combinations(idx, "W")

            s_list.append(screenImg)
            w_list.append(screenWireframeImg)
            label_list.append(screenLabel)

        observed_idx_arr[:, 0] = [True] * len(label_list)
        observed_idx_arr[:, 1] = [True] * len(label_list)

        data_dict["screenshot"] = np.array(s_list)
        data_dict["wireframe"] = np.array(w_list)
        filtered_idx = set(list(id_to_idx.values()))
        common_idx_list.append(filtered_idx)

        if args.patch:
            encoder_dict["screenshot"] = torch.nn.Sequential(
                VGG11Slim(
                    1024, dropout=True, dropoutp=0.2, freeze_features=True
                ).cuda(),
                PatchEmbeddings(
                    1024, num_patches=args.num_patches, embed_dim=args.hidden_dim
                ).to(args.device),
            )
            encoder_dict["wireframe"] = torch.nn.Sequential(
                VGG11Slim(
                    1024, dropout=True, dropoutp=0.2, freeze_features=True
                ).cuda(),
                PatchEmbeddings(
                    1024, num_patches=args.num_patches, embed_dim=args.hidden_dim
                ).to(args.device),
            )
        else:
            encoder_dict["screenshot"] = torch.nn.Sequential(
                VGG11Slim(
                    1024, dropout=True, dropoutp=0.2, freeze_features=True
                ).cuda(),
                Linear(1024, args.hidden_dim, xavier_init=True).cuda(),
            )
            encoder_dict["wireframe"] = torch.nn.Sequential(
                VGG11Slim(
                    1024, dropout=True, dropoutp=0.2, freeze_features=True
                ).cuda(),
                Linear(1024, args.hidden_dim, xavier_init=True).cuda(),
            )
        input_dims["screenshot"] = args.hidden_dim
        input_dims["wireframe"] = args.hidden_dim

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

    labels = np.array(label_list)
    n_labels = 20
    # breakpoint()
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
