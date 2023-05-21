import torch
from collections import defaultdict
import time
from src.utils import to_numpy
import seaborn as sns
import matplotlib.pyplot as plt
from os import path as pt


class BaseTrainer:
    def __init__(
        self,
        batch_size,
        G,
        G_optimizer,
        test_metrics_train,
        test_metrics_test,
        n_gradient_steps,
        foo=lambda x: x,
    ):
        self.batch_size = batch_size

        self.G = G
        self.G_optimizer = G_optimizer
        self.n_gradient_steps = n_gradient_steps

        self.losses_history = defaultdict(list)

        self.test_metrics_train = test_metrics_train
        self.test_metrics_test = test_metrics_test
        self.foo = foo

        self.init_time = time.time()

    def evaluate(self, x_fake, x_real, step, config, **kwargs):
        if "condition" in kwargs:
            condition = kwargs["condition"]
        else:
            condition = None

        self.losses_history["time"].append(time.time() - self.init_time)

        plt_sample = self.plot_sample
        plt_rec = self.plot_reconstructed_sample

        plt_sample(x_real, x_fake[: config.batch_size], self.config, step)

        if self.config.gan_algo == "RPathChar_GAN":
            plt_rec(
                x_real[0],
                self.G(
                    batch_size=x_real.shape[0],
                    n_lags=config.n_lags,
                    device=config.device,
                    z=self.D(x_real),
                )[0],
                self.config,
                step,
            )

    @staticmethod
    def plot_sample(real_X, fake_X, config, step):
        sns.set()

        x_real_dim = real_X.shape[-1]
        for i in range(x_real_dim):
            plt.plot(
                to_numpy(fake_X[: config.batch_size, :, i]).T, "C%s" % i, alpha=0.3
            )
        plt.savefig(pt.join(config.exp_dir, "x_fake_" + str(step) + ".png"))
        plt.close()

        for i in range(x_real_dim):
            random_indices = torch.randint(0, real_X.shape[0], (config.batch_size,))
            plt.plot(to_numpy(real_X[random_indices, :, i]).T, "C%s" % i, alpha=0.3)
        plt.savefig(pt.join(config.exp_dir, "x_real_" + str(step) + ".png"))
        plt.close()

    @staticmethod
    def plot_sample1(real_X, fake_X, config, step):
        sns.set()
        fig, axs = plt.subplots(2, 5)
        x_real_dim = real_X.shape[-1]
        for j in range(10):
            for i in range(x_real_dim):
                axs.flatten()[j].plot(to_numpy(fake_X[j, :, i]).T)
        plt.savefig(pt.join(config.exp_dir, "x_fake_" + str(step) + ".png"))
        plt.close()

        fig, axs = plt.subplots(2, 5)
        for j in range(10):
            random_indices = torch.randint(0, real_X.shape[0], (10,))
            for i in range(x_real_dim):
                axs.flatten()[j].plot(to_numpy(real_X[random_indices[j], :, i]).T)
        plt.savefig(pt.join(config.exp_dir, "x_real_" + str(step) + ".png"))
        plt.close()

    @staticmethod
    def plot_reconstructed_sample(
        real_X: torch.tensor, rec_X: torch.tensor, config, step
    ):
        sns.set()
        fig, axs = plt.subplots(1, 2)
        x_real_dim = real_X.shape[-1]
        for i in range(x_real_dim):
            axs[0].plot(to_numpy(real_X[:, i]).T)
        for i in range(x_real_dim):
            axs[1].plot(to_numpy(rec_X[:, i]).T)
        plt.savefig(
            pt.join(config.exp_dir, "reconstruction_sample_" + str(step) + ".png")
        )
        plt.close()
