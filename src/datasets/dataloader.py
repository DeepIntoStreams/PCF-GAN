import torch
import ml_collections
from typing import Tuple
from torch.utils.data import DataLoader
from src.datasets.rough import Rough_S
from src.datasets.stock import Stock
from src.datasets.beijing_air_quality import Beijing_air_quality
from src.datasets.ou import OU
from src.datasets.eeg import EEG


def get_dataset(
    config: ml_collections.ConfigDict,
    num_workers: int = 1,
    data_root="./data",
) -> Tuple[dict, torch.utils.data.DataLoader]:
    """
    Create datasets loaders for the chosen datasets
    :return: Tuple ( dict(train_loader, val_loader) , test_loader)
    """
    dataset = {
        "ROUGH": Rough_S,
        "Air_Quality": Beijing_air_quality,
        "Stock": Stock,
        "OU": OU,
        "EEG": EEG,
    }[config.dataset]
    training_set = dataset(
        partition="train",
        n_lags=config.n_lags,
    )
    test_set = dataset(
        partition="test",
        n_lags=config.n_lags,
    )

    training_loader = DataLoader(
        training_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    config.input_dim = training_loader.dataset[0][0].shape[-1]
    n_lags = next(iter(test_loader))[0].shape[1]

    config.update({"n_lags": n_lags}, allow_val_change=True)
    # print("data shape:", next(iter(test_loader))[0].shape)
    return training_loader, test_loader


if __name__ == "__main__":
    from src.evaluations.test_metrics import acf_torch, ccf_mean

    # calculate the key statistics on four benchmark datasets
    dataset_dict = {
        "ROUGH": Rough_S,
        "Air_Quality": Beijing_air_quality,
        "Google_Stock": Stock,
        "EEG": EEG,
    }
    for key in dataset_dict:
        dataset = dataset_dict[key]("train")
        data = dataset.tensors[0]
        print(key)
        print("mean acf for lag 1:", acf_torch(data, 10).mean(1)[1])
        print("mean acf for lag 5:", acf_torch(data, 10).mean(1)[5])
        print("mean cross correlation:", ccf_mean(data))
