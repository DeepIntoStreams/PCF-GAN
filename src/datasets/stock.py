import torch
import pathlib
import os
from src.datasets.utils import load_data, save_data, train_test_split
import pandas as pd
import pandas as pd
import yfinance as yf
import datetime


class Stock(torch.utils.data.TensorDataset):
    def __init__(
        self,
        partition: str,
        **kwargs,
    ):
        n_lags = 20
        self.root = pathlib.Path("data")

        data_loc = pathlib.Path("data/stock/processed_data_{}".format(n_lags))

        if os.path.exists(data_loc):
            pass
        else:
            if not os.path.exists(data_loc.parent):
                os.mkdir(data_loc.parent)
            if not os.path.exists(data_loc):
                os.mkdir(data_loc)
            x_real = self._download_data()
            train_X, test_X = train_test_split(x_real, 0.8)
            save_data(
                data_loc,
                train_X=train_X,
                test_X=test_X,
            )
        X = self.load_data(data_loc, partition)
        super(Stock, self).__init__(X)

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

    def _download_data(self, n_lags=20):
        start = datetime.datetime(2013, 1, 3)
        end = datetime.datetime(2021, 12, 23)
        Symbols = [
            "GOOG",
            "AAPL",
            "AMZN",
            "TSLA",
            "META",
            "MSFT",
            "NVDA",
            "JPM",
            "V",
            "PG",
        ]
        stock_final = pd.DataFrame()
        # iterate over each symbol
        for i in Symbols:
            # print the symbol which is being downloaded
            print(str(Symbols.index(i)) + str(" : ") + i, sep=",", end=",", flush=True)

            stock = []
            stock = yf.download(i, start=start, end=end, progress=False)

            stock = (stock - stock.min()) / (
                stock.max() - stock.min()
            )  # minmax normalization
            stock_final = stock_final.append(stock, sort=False)

        return torch.tensor(stock_final.values, dtype=torch.float32).reshape(
            -1, n_lags, 6
        )
