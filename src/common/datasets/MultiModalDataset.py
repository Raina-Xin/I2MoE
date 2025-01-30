import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MultiModalDataset(Dataset):
    def __init__(
        self,
        data_dict,
        observed_idx,
        ids,
        labels,
        input_dims,
        transforms,
        masks,
        use_common_ids=True,
    ):
        self.data_dict = data_dict
        self.mc = np.array(data_dict["modality_comb"])
        self.observed = observed_idx
        self.ids = np.array(ids)
        self.labels = np.array(labels)
        self.input_dims = input_dims
        self.transforms = transforms
        self.masks = masks
        self.use_common_ids = use_common_ids
        self.data = {
            modality: np.array(data)[ids]
            for modality, data in self.data_dict.items()
            if "modality" not in modality
        }
        self.label = self.labels[ids]
        self.mc = self.mc[ids]
        self.observed = self.observed[ids]

    def process_2d_to_3d(self, data, idx, masks, transforms):
        subj1 = data[idx]
        subj_gm_3d = np.zeros(masks.shape, dtype=np.float32)
        subj_gm_3d.ravel()[masks] = subj1
        subj_gm_3d = subj_gm_3d.reshape((91, 109, 91))
        if transforms:
            subj_gm_3d = transforms(subj_gm_3d)
        sample = subj_gm_3d[None, :, :, :]  # Add channel dimension
        output = np.array(sample)

        return output

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_data = {}
        for modality, data in self.data.items():
            sample_data[modality] = np.nan_to_num(data[idx])
            if modality == "image":
                sample_data[modality] = self.process_2d_to_3d(
                    data, idx, self.masks, self.transforms
                )

        sampele_id = self.ids[idx]
        label = self.label[idx]
        mc = self.mc[idx]
        observed = self.observed[idx]

        return sampele_id, sample_data, label, mc, observed


def collate_fn(batch):
    _, data, labels, mcs, observeds = zip(*batch)
    modalities = data[0].keys()
    collated_data = {
        modality: torch.tensor(
            np.stack([d[modality] for d in data]), dtype=torch.float32
        )
        for modality in modalities
    }

    labels = torch.tensor(labels, dtype=torch.long)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))
    return collated_data, labels, mcs, observeds


def collate_fn_test(batch):
    sampele_ids, data, labels, mcs, observeds = zip(*batch)
    modalities = data[0].keys()
    collated_data = {
        modality: torch.tensor(
            np.stack([d[modality] for d in data]), dtype=torch.float32
        )
        for modality in modalities
    }
    sampele_ids = torch.tensor(sampele_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))
    return collated_data, sampele_ids, labels, mcs, observeds


def collate_fn_mosi_regression(batch):
    _, data, labels, mcs, observeds = zip(*batch)
    modalities = data[0].keys()
    collated_data = {
        modality: torch.tensor(
            np.stack([d[modality] for d in data]), dtype=torch.float32
        )
        for modality in modalities
    }
    labels = torch.tensor(labels, dtype=torch.float)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))
    return collated_data, labels, mcs, observeds


def collate_fn_mosi_regression_test(batch):
    sample_ids, data, labels, mcs, observeds = zip(*batch)
    modalities = data[0].keys()
    collated_data = {
        modality: torch.tensor(
            np.stack([d[modality] for d in data]), dtype=torch.float32
        )
        for modality in modalities
    }
    sample_ids = torch.tensor(sample_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))
    return collated_data, sample_ids, labels, mcs, observeds


def collate_fn_mmimdb(batch):
    _, data, labels, mcs, observeds = zip(*batch)
    modalities = data[0].keys()
    collated_data = {
        modality: torch.tensor(
            np.stack([d[modality] for d in data]), dtype=torch.float32
        )
        for modality in modalities
    }
    # labels = torch.tensor(labels, dtype=torch.long)
    labels = np.array(labels)  # Convert list to a single NumPy array
    labels = torch.tensor(labels, dtype=torch.float)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))
    return collated_data, labels, mcs, observeds


def collate_fn_mmimdb_test(batch):
    sample_ids, data, labels, mcs, observeds = zip(*batch)
    modalities = data[0].keys()
    collated_data = {
        modality: torch.tensor(
            np.stack([d[modality] for d in data]), dtype=torch.float32
        )
        for modality in modalities
    }
    sample_ids = torch.tensor(sample_ids, dtype=torch.float)
    # labels = torch.tensor(labels, dtype=torch.long)
    labels = np.array(labels)  # Convert list to a single NumPy array
    labels = torch.tensor(labels, dtype=torch.float)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))
    return collated_data, sample_ids, labels, mcs, observeds


def create_loaders(
    data_dict,
    observed_idx,
    labels,
    train_ids,
    valid_ids,
    test_ids,
    batch_size,
    num_workers,
    pin_memory,
    input_dims,
    transforms,
    masks,
    use_common_ids=True,
    dataset="mosi_regression",
):
    if "image" in list(data_dict.keys()) and dataset == "adni":
        train_transfrom = val_transform = test_transform = transforms["image"]
        # val_transform = test_transform = False
        mask = masks["image"]
    else:
        train_transfrom = val_transform = test_transform = False
        mask = None

    train_dataset = MultiModalDataset(
        data_dict,
        observed_idx,
        train_ids,
        labels,
        input_dims,
        train_transfrom,
        mask,
        use_common_ids,
    )
    valid_dataset = MultiModalDataset(
        data_dict,
        observed_idx,
        valid_ids,
        labels,
        input_dims,
        val_transform,
        mask,
        use_common_ids,
    )
    test_dataset = MultiModalDataset(
        data_dict,
        observed_idx,
        test_ids,
        labels,
        input_dims,
        test_transform,
        mask,
        use_common_ids,
    )

    if dataset == "mosi_regression":
        # if False:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_mosi_regression,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_mosi_regression,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_mosi_regression_test,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    elif dataset == "mmimdb":
        # if False:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_mmimdb,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_mmimdb,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_mmimdb_test,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_test,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, test_loader
