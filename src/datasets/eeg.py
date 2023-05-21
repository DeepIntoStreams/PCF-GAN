import torch
import pathlib
import os
from src.datasets.utils import load_data, save_data, train_test_split
import pandas as pd
import urllib.request


class EEG(torch.utils.data.TensorDataset):
    def __init__(
        self,
        partition: str,
        n_lags: int = 20,
        **kwargs,
    ):
        self.n_lags = n_lags
        self.root = pathlib.Path("data")

        data_loc = pathlib.Path("data/EEG/processed_data")

        if os.path.exists(data_loc):
            pass
        else:
            self.download()
            if not os.path.exists(data_loc.parent):
                os.mkdir(data_loc.parent)
            if not os.path.exists(data_loc):
                os.mkdir(data_loc)
            x_real = self._process_data()
            train_X, test_X = train_test_split(x_real, 0.8)
            save_data(
                data_loc,
                train_X=train_X,
                test_X=test_X,
            )
        X = self.load_data(data_loc, partition)[:-1, :, :]
        super(EEG, self).__init__(X)

    @staticmethod
    def load_data(data_loc, partition):
        tensors = load_data(data_loc)
        if partition == "train":
            X = tensors["train_X"]
        elif partition == "test":
            X = tensors["test_X"]
        else:
            raise NotImplementedError("the set {} is not implemented.".format(set))

        return X

    def download(self):
        root = self.root
        base_loc = root / "EEG"
        loc = base_loc / "EEG Eye State.arff"

        from scipy.io import arff

        data = arff.loadarff(loc)
        df = pd.DataFrame(data[0])
        df.drop(["eyeDetection"], axis=1).to_csv(base_loc / "EEG_Eye_State.csv")

    def _process_data(self):
        data_loc = "data/EEG/EEG_Eye_State.csv"
        data = pd.read_csv(data_loc, index_col=0)

        data = (data - data.mean()) / (3 * data.std())

        return torch.tanh(torch.tensor(data.values, dtype=torch.float32)).reshape(
            -1, 20, 14
        )
