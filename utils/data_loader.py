import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch.utils.data.dataloader import default_collate


class TimeSeriesDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = torch.FloatTensor(xs)
        self.ys = torch.FloatTensor(ys)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


class StandardScaler:
    def __init__(self, mean, std, median=None):
        self.mean = mean
        self.std = std
        self._median = median

    def get_median(self):
        return self._median

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            return (data * self.std) + self.mean
        return (data * self.std) + self.mean


def load_dataset(
    dataset_dir,
    batch_size,
    valid_batch_size=None,
    test_batch_size=None,
    train_ratio=100,
):
    data = {}
    g = torch.Generator()
    g.manual_seed(42)
    for category in ["train", "val", "test"]:
        cat_data = np.load(os.path.join(dataset_dir, category + ".npz"), mmap_mode="r")
        data["x_" + category] = cat_data["x"].astype(np.float32)
        data["y_" + category] = cat_data["y"].astype(np.float32)

    # Apply train_ratio to training data
    if train_ratio < 100:
        assert train_ratio % 10 == 0, "train_ratio must be a multiple of 10"
        assert 10 <= train_ratio <= 100, "train_ratio must be between 10 and 100"

        total_samples = len(data["x_train"])
        num_samples = int(total_samples * train_ratio / 100)

        # Use consistent random sampling
        np.random.seed(42)
        indices = np.random.permutation(total_samples)[:num_samples]
        indices = np.sort(indices)  # Sort to maintain temporal order

        data["x_train"] = data["x_train"][indices]
        data["y_train"] = data["y_train"][indices]

        print(
            f"Using {train_ratio}% of training data: {num_samples} out of {total_samples} samples"
        )
    scaler = StandardScaler(
        mean=np.float32(data["x_train"][..., 0].mean()),
        std=np.float32(data["x_train"][..., 0].std()),
        median=np.float32(np.median(data["x_train"][..., 0])),
    )
    for category in ["train", "val", "test"]:
        data["x_" + category][..., 0] = scaler.transform(data["x_" + category][..., 0])
        data["x_" + category] = torch.FloatTensor(data["x_" + category])
        data["y_" + category] = torch.FloatTensor(data["y_" + category])

    train_dataset = TimeSeriesDataset(data["x_train"], data["y_train"])
    val_dataset = TimeSeriesDataset(data["x_val"], data["y_val"])
    test_dataset = TimeSeriesDataset(data["x_test"], data["y_test"])
    dataloader_configs = {
        "train": {
            "dataset": train_dataset,
            "batch_size": batch_size,
            "shuffle": True,
            "generator": g,
        },
        "val": {
            "dataset": val_dataset,
            "batch_size": valid_batch_size or batch_size,
            "shuffle": False,
        },
        "test": {
            "dataset": test_dataset,
            "batch_size": test_batch_size or batch_size,
            "shuffle": False,
        },
    }

    common_config = {
        "num_workers": 12,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }
    for key, config in dataloader_configs.items():
        data[f"{key}_loader"] = TorchDataLoader(**config, **common_config)
    data["scaler"] = scaler
    return data
