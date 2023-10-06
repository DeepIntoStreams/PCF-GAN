import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import copy
from src.utils import loader_to_tensor, to_numpy, loader_to_cond_tensor
import matplotlib.pyplot as plt
from os import path as pt
import seaborn as sns
from src.evaluations.test_metrics import (
    cacf_torch,
    Sig_mmd,
    kurtosis_torch,
    skew_torch,
    ccf_metric,
    acf_metric,
    HistoLoss
)
from matplotlib.ticker import MaxNLocator
import numpy as np
import logging
from src.utils import get_experiment_dir
from sklearn.manifold import TSNE


def _train_classifier(model, train_loader, test_loader, config, epochs=100):
    """
    Trains a classifier model using the given train and test loaders.

    Args:
        model (nn.Module): The classifier model to be trained.
        train_loader (DataLoader): The data loader for the training dataset.
        test_loader (DataLoader): The data loader for the test/validation dataset.
        config: Configuration object containing training parameters.
        epochs (int): The number of training epochs.

    Returns:
        float: The test accuracy of the trained model.
        float: The test loss of the trained model.
    """
    # Training parameter
    device = config.device
    # clip = config.clip
    # iterate over epochs
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    dataloader = {"train": train_loader, "validation": test_loader}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 999
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 30)
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            # Accumulate accuracy and loss
            running_loss = 0
            running_corrects = 0
            total = 0
            # iterate over data
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                if config.dataset == "MNIST":
                    inputs = inputs.squeeze(1).permute(0, 2, 1)
                optimizer.zero_grad()
                train = phase == "train"
                with torch.set_grad_enabled(train):
                    # FwrdPhase:
                    # inputs = torch.dropout(inputs, config.dropout_in, train)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    # BwrdPhase:
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)
                # statistics of the epoch
            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "validation" and epoch_acc >= best_acc:
                # Updates to the weights will not happen if the accuracy is equal but loss does not diminish
                if (epoch_acc == best_acc) and (epoch_loss > best_loss):
                    pass
                else:
                    best_acc = epoch_acc
                    best_loss = epoch_loss

                    best_model_wts = copy.deepcopy(model.state_dict())

                    # Log best results so far and the weights of the model.

                    # Clean CUDA Memory
                    del inputs, outputs, labels
                    torch.cuda.empty_cache()

    print("Best Val Acc: {:.4f}".format(best_acc))
    # Load best model weights
    model.load_state_dict(best_model_wts)
    test_acc, test_loss = _test_classifier(model, test_loader, config)
    return test_acc, test_loss


def _test_classifier(model, test_loader, config):
    """
    Evaluates the performance of a trained classifier model on a test dataset.

    Args:
        model (nn.Module): The trained classifier model to be evaluated.
        test_loader (DataLoader): The data loader for the test dataset.
        config: Configuration object containing evaluation parameters.

    Returns:
        float: The test accuracy of the model.
        float: The test loss of the model.
    """
    # send model to device
    device = config.device

    model.eval()
    model.to(device)

    # Summarize results
    correct = 0
    total = 0
    running_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        # Iterate through data
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if config.dataset == "MNIST":
                inputs = inputs.squeeze(1).permute(0, 2, 1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print results
    test_acc = correct / total
    test_loss = running_loss / total
    print(
        "Accuracy of the network on the {} test samples: {}".format(
            total, (100 * test_acc)
        )
    )
    return test_acc, test_loss


def _train_regressor(model, train_loader, test_loader, config, epochs=100):
    """
    Trains a regressor model using the given train and test loaders.

    Args:
        model (nn.Module): The regressor model to be trained.
        train_loader (DataLoader): The data loader for the training dataset.
        test_loader (DataLoader): The data loader for the test/validation dataset.
        config: Configuration object containing training parameters.
        epochs (int): The number of training epochs.

    Returns:
        float: The test performance metric of the trained model.
    """
    # Training parameter
    device = config.device
    # clip = config.clip
    # iterate over epochs
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 999
    dataloader = {"train": train_loader, "validation": test_loader}
    criterion = torch.nn.L1Loss()
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 30)
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0
            total = 0
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                train = phase == "train"
                with torch.set_grad_enabled(True):
                    # FwrdPhase:
                    # inputs = torch.dropout(inputs, config.dropout_in, train)
                    outputs = model(inputs)
                    # print(outputs.shape, labels.shape)
                    loss = criterion(outputs, labels)
                    # Regularization:
                    # BwrdPhase:
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                total += labels.size(0)
            epoch_loss = running_loss / total
            print("{} Loss: {:.4f}".format(phase, epoch_loss))
        if phase == "validation" and epoch_loss <= best_loss:
            # Updates to the weights will not happen if the accuracy is equal but loss does not diminish

            best_loss = epoch_loss

            best_model_wts = copy.deepcopy(model.state_dict())

            # Log best results so far and the weights of the model.

            # Clean CUDA Memory
            del inputs, outputs, labels
            torch.cuda.empty_cache()
    print("Best Val MSE: {:.4f}".format(best_loss))
    # Load best model weights
    model.load_state_dict(best_model_wts)
    epoch_loss = _test_regressor(model, test_loader, config)

    return best_loss


def _test_regressor(model, test_loader, config):
    """
    Evaluates the performance of a trained regressor model on a test dataset.

    Args:
        model (nn.Module): The trained regressor model to be evaluated.
        test_loader (DataLoader): The data loader for the test dataset.
        config: Configuration object containing evaluation parameters.

    Returns:
        float: The test performance metric of the model.
    """
    # send model to device
    device = config.device

    model.eval()
    model.to(device)

    # Summarize results
    total = 0
    running_loss = 0
    criterion = torch.nn.L1Loss()
    with torch.no_grad():
        # Iterate through data
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            total += labels.size(0)

    # Print results
    test_loss = running_loss / total

    return test_loss


def fake_loader(generator, num_samples, n_lags, batch_size, config, **kwargs):
    if "recovery" in kwargs:
        recovery = kwargs["recovery"]
    if "condition" in kwargs:
        condition = kwargs["condition"]
        with torch.no_grad():
            if config.gan_algo == "TimeGAN":
                fake_data = generator(
                    batch_size=num_samples,
                    n_lags=n_lags,
                    condition=condition,
                    device="cpu",
                )
                fake_data = recovery(fake_data)
            else:
                fake_data = generator(
                    num_samples, n_lags, condition=condition, device="cpu"
                )

    else:
        with torch.no_grad():
            if config.gan_algo == "TimeGAN":
                fake_data = generator(
                    batch_size=num_samples, n_lags=n_lags, device="cpu"
                )
                fake_data = recovery(fake_data)
            else:
                fake_data = generator(num_samples, n_lags, device="cpu")
    tensor_x = torch.Tensor(fake_data)
    return DataLoader(TensorDataset(tensor_x), batch_size=batch_size)


def compute_discriminative_score(
    real_train_dl,
    real_test_dl,
    fake_train_dl,
    fake_test_dl,
    config,
    hidden_size=64,
    num_layers=2,
    epochs=30,
    batch_size=512,
):
    """Compute the discriminative score for evaluating fake data.

    Args:
        real_train_dl (torch.utils.data.DataLoader): DataLoader for real training data.
        real_test_dl (torch.utils.data.DataLoader): DataLoader for real testing data.
        fake_train_dl (torch.utils.data.DataLoader): DataLoader for fake training data.
        fake_test_dl (torch.utils.data.DataLoader): DataLoader for fake testing data.
        config: Configuration object.
        hidden_size (int): Hidden size for the discriminator model.
        num_layers (int): Number of layers for the discriminator model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        float: Mean absolute difference between the accuracy and 0.5.
        float: Standard deviation of the accuracy.
    """

    def create_dl(real_dl, fake_dl, batch_size):
        train_x, train_y = [], []
        for data in real_dl:
            train_x.append(data[0])
            train_y.append(
                torch.ones(
                    data[0].shape[0],
                )
            )
        for data in fake_dl:
            train_x.append(data[0])
            train_y.append(
                torch.zeros(
                    data[0].shape[0],
                )
            )
        x, y = torch.cat(train_x), torch.cat(train_y).long()
        idx = torch.randperm(x.shape[0])

        return DataLoader(
            TensorDataset(x[idx].view(x.size()), y[idx].view(y.size())),
            batch_size=batch_size,
        )

    train_dl = create_dl(real_train_dl, fake_train_dl, batch_size)
    test_dl = create_dl(real_test_dl, fake_test_dl, batch_size)

    class Discriminator(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, out_size=2):
            super(Discriminator, self).__init__()
            self.rnn = nn.GRU(
                input_size=input_size,
                num_layers=num_layers,
                hidden_size=hidden_size,
                batch_first=True,
            )
            self.linear = nn.Linear(hidden_size, out_size)

        def forward(self, x):
            x = self.rnn(x)[0][:, -1]
            return self.linear(x)

    test_acc_list = []
    for i in range(1):
        model = Discriminator(train_dl.dataset[0][0].shape[-1], hidden_size, num_layers)

        test_acc, test_loss = _train_classifier(
            model.to(config.device), train_dl, test_dl, config, epochs=epochs
        )
        test_acc_list.append(test_acc)
    mean_acc = np.mean(np.array(test_acc_list))
    std_acc = np.std(np.array(test_acc_list))
    return abs(mean_acc - 0.5), std_acc


def plot_samples(real_dl, fake_dl, config):
    sns.set()
    real_X, fake_X = loader_to_tensor(real_dl), loader_to_tensor(fake_dl)
    x_real_dim = real_X.shape[-1]
    for i in range(x_real_dim):
        plt.plot(to_numpy(fake_X[:250, :, i]).T, "C%s" % i, alpha=0.1)
    plt.savefig(pt.join(config.exp_dir, "x_fake.png"))
    plt.close()

    for i in range(x_real_dim):
        random_indices = torch.randint(0, real_X.shape[0], (250,))
        plt.plot(to_numpy(real_X[random_indices, :, i]).T, "C%s" % i, alpha=0.1)
    plt.savefig(pt.join(config.exp_dir, "x_real.png"))
    plt.close()


def plot_samples1(real_dl, fake_dl, config):
    sns.set()
    real_X, fake_X = loader_to_tensor(real_dl), loader_to_tensor(fake_dl)
    x_real_dim = real_X.shape[-1]
    for i in range(x_real_dim):
        random_indices = torch.randint(0, real_X.shape[0], (100,))
        plt.plot(to_numpy(fake_X[:100, :, i]).T, "C%s" % i, alpha=0.1)
        plt.plot(to_numpy(real_X[random_indices, :, i]).T, "C%s" % i, alpha=0.1)
        plt.savefig(pt.join(config.exp_dir, "sample_plot{}.png".format(i)))
        plt.close()


def set_style(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def compare_hists(x_real, x_fake, ax=None, log=False, label=None):
    """Computes histograms and plots those."""
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if label is not None:
        label_historical = "Real" + label
        label_generated = "Generated " + label
    else:
        label_historical = "Real"
        label_generated = "Generated"
    ax.hist(x_real.flatten(), bins=80, alpha=0.6, density=True, label=label_historical)[
        1
    ]
    ax.hist(x_fake.flatten(), bins=80, alpha=0.6, density=True, label=label_generated)
    ax.grid()
    set_style(ax)
    ax.legend()
    if log:
        ax.set_ylabel("log-pdf")
        ax.set_yscale("log")
    else:
        ax.set_ylabel("pdf")
    return ax


def compare_acf(
    x_real, x_fake, ax=None, max_lag=64, CI=True, dim=(0, 1), drop_first_n_lags=0
):
    """Computes ACF of historical and (mean)-ACF of generated and plots those."""
    if ax is None:
        _, ax = plt.subplots(1, 1)
    acf_real_list = cacf_torch(x_real, max_lag=max_lag, dim=dim).cpu().numpy()
    acf_real = np.mean(acf_real_list, axis=0)

    acf_fake_list = cacf_torch(x_fake, max_lag=max_lag, dim=dim).cpu().numpy()
    acf_fake = np.mean(acf_fake_list, axis=0)

    ax.plot(acf_real[drop_first_n_lags:], label="Historical")
    ax.plot(acf_fake[drop_first_n_lags:], label="Generated", alpha=0.8)

    if CI:
        acf_fake_std = np.std(acf_fake_list, axis=0)
        ub = acf_fake + acf_fake_std
        lb = acf_fake - acf_fake_std

        for i in range(acf_real.shape[-1]):
            ax.fill_between(
                range(acf_fake[:, i].shape[0]),
                ub[:, i],
                lb[:, i],
                color="orange",
                alpha=0.3,
            )
    set_style(ax)
    ax.set_xlabel("Lags")
    ax.set_ylabel("ACF")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    ax.legend()
    return ax


def plot_hists_marginals(x_real, x_fake):
    sns.set()
    n_hists = 10
    n_lags = x_real.shape[1]
    len_interval = n_lags // n_hists
    fig = plt.figure(figsize=(20, 8))

    for i in range(n_hists):
        ax = fig.add_subplot(2, 5, i + 1)
        compare_hists(
            to_numpy(x_real[:, i * len_interval, 0]),
            to_numpy(x_fake[:, i * len_interval, 0]),
            ax=ax,
        )
        ax.set_title("Step {}".format(i * len_interval))
    fig.tight_layout()
    # fig.savefig(pt.join(config.exp_dir, 'marginal_comparison.png'))
    # plt.close(fig)
    return fig


def plot_summary(fake_dl, real_dl, config, max_lag=None):
    x_real, x_fake = loader_to_tensor(real_dl), loader_to_tensor(fake_dl)
    if max_lag is None:
        max_lag = min(128, x_fake.shape[1])

    dim = x_real.shape[2]
    _, axes = plt.subplots(dim, 3, figsize=(25, dim * 5))

    if len(axes.shape) == 1:
        axes = axes[None, ...]
    for i in range(dim):
        x_real_i = x_real[..., i : i + 1]
        x_fake_i = x_fake[..., i : i + 1]

        compare_hists(
            x_real=to_numpy(x_real_i), x_fake=to_numpy(x_fake_i), ax=axes[i, 0]
        )

        def text_box(x, height, title):
            textstr = "\n".join(
                (
                    r"%s" % (title,),
                    # t'abs_metric=%.2f' % abs_metric
                    r"$s=%.2f$" % (skew_torch(x).item(),),
                    r"$\kappa=%.2f$" % (kurtosis_torch(x).item(),),
                )
            )
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            axes[i, 0].text(
                0.05,
                height,
                textstr,
                transform=axes[i, 0].transAxes,
                fontsize=14,
                verticalalignment="top",
                bbox=props,
            )

        text_box(x_real_i, 0.95, "Historical")
        text_box(x_fake_i, 0.70, "Generated")

        compare_hists(
            x_real=to_numpy(x_real_i),
            x_fake=to_numpy(x_fake_i),
            ax=axes[i, 1],
            log=True,
        )
        # compare_acf(x_real=x_real_i, x_fake=x_fake_i,
        #           ax=axes[i, 2], max_lag=max_lag, CI=False, dim=(0, 1))
    plt.savefig(pt.join(config.exp_dir, "comparison.png"))
    plt.close()

    for i in range(x_real.shape[2]):
        fig = plot_hists_marginals(
            x_real=x_real[..., i : i + 1], x_fake=x_fake[..., i : i + 1]
        )
        fig.savefig(pt.join(config.exp_dir, "hists_marginals_dim{}.pdf".format(i)))
        plt.close()
    plot_samples(real_dl, fake_dl, config)
    plot_tsne(real_dl,fake_dl,2000)


def compute_predictive_score(
    real_train_dl,
    real_test_dl,
    fake_train_dl,
    fake_test_dl,
    config,
    hidden_size=64,
    num_layers=3,
    epochs=100,
    batch_size=128,
):
    """Compute the predictive score for evaluating fake data.

    Args:
        real_train_dl (torch.utils.data.DataLoader): DataLoader for real training data.
        real_test_dl (torch.utils.data.DataLoader): DataLoader for real testing data.
        fake_train_dl (torch.utils.data.DataLoader): DataLoader for fake training data.
        fake_test_dl (torch.utils.data.DataLoader): DataLoader for fake testing data.
        config: Configuration object.
        hidden_size (int): Hidden size for the predictor model.
        num_layers (int): Number of layers for the predictor model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        float: Mean of the test loss across multiple runs.
        float: Standard deviation of the test loss across multiple runs.
    """

    def create_dl(train_dl, test_dl, batch_size):
        x, y = [], []
        _, T, C = next(iter(train_dl))[0].shape

        T_cutoff = int(T / 10)
        for data in train_dl:
            x.append(data[0][:, :-T_cutoff])
            y.append(data[0][:, -T_cutoff:].reshape(data[0].shape[0], -1))
        for data in test_dl:
            x.append(data[0][:, :-T_cutoff])
            y.append(data[0][:, -T_cutoff:].reshape(data[0].shape[0], -1))
        x, y = (
            torch.cat(x),
            torch.cat(y),
        )
        idx = torch.randperm(x.shape[0])
        dl = DataLoader(
            TensorDataset(x[idx].view(x.size()), y[idx].view(y.size())),
            batch_size=batch_size,
        )

        return dl

    train_dl = create_dl(fake_train_dl, fake_test_dl, batch_size)
    test_dl = create_dl(real_train_dl, real_test_dl, batch_size)

    class predictor(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, out_size):
            super(predictor, self).__init__()
            self.rnn = nn.LSTM(
                input_size=input_size,
                num_layers=num_layers,
                hidden_size=hidden_size,
                batch_first=True,
            )
            self.linear = nn.Linear(hidden_size, out_size)

        def forward(self, x):
            x = self.rnn(x)[0][:, -1]
            return self.linear(x)

    test_loss_list = []
    for i in range(1):
        model = predictor(
            train_dl.dataset[0][0].shape[-1],
            hidden_size,
            num_layers,
            out_size=train_dl.dataset[0][1].shape[-1],
        )
        test_loss = _train_regressor(
            model.to(config.device), train_dl, test_dl, config, epochs=epochs
        )
        test_loss_list.append(test_loss)
    mean_loss = np.mean(np.array(test_loss_list))
    std_loss = np.std(np.array(test_loss_list))
    return mean_loss, std_loss

def plot_tsne(real_dl, fake_dl, config,num_sample=1000, plot_show=False):
    # Analysis sample size (for faster computation)
    sns.set()
    ori_data = loader_to_tensor(real_dl)
    generated_data = loader_to_tensor(fake_dl)
    ori_data = torch.flatten(ori_data,1).numpy()
    generated_data = torch.flatten(generated_data,1).numpy()
    anal_sample_no = min([num_sample, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    # Data preprocessing
    ori_data = ori_data[idx]
    generated_data = generated_data[idx]
    
    
    prep_data = np.asarray(ori_data)
    prep_data_hat = np.asarray(generated_data)



    #no, seq_len, dim = ori_data.shape

    '''
    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(
                np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))
    '''
    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + \
        ["blue" for i in range(anal_sample_no)]
    # Do t-SNE Analysis together
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

    # TSNE anlaysis
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(prep_data_final)

    # Plotting
    f, ax = plt.subplots(1)
    plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                c=colors[:anal_sample_no], alpha=0.2, label="Original")
    plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")
    ax.legend()

    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y-tsne')
    # you might want to save it into a specific path
    if plot_show:
        plt.show()
    else:
        plt.savefig(pt.join(config.exp_dir, "tsne.png"), dpi=200)
    plt.close()


def full_evaluation(generator, real_train_dl, real_test_dl, config, **kwargs):
    """Evaluate the synthetic generation including discriminative score, predictive score, predictive FID, and predictive KID.

    Args:
        generator: The generator model.
        real_train_dl: DataLoader for real training data.
        real_test_dl: DataLoader for real testing data.
        config: Configuration object.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple: Mean of the discriminative score, mean of the predictive score, mean of the Sig MMD score.
    """
    d_scores = []
    p_scores = []
    Sig_MMDs = []
    acf1=[]
    acf5=[]
    ccf1=[]
    ccf5=[]
    m_scores = []
    if config.generator == "Conditional_LSTM":
        train_data, train_condition = loader_to_cond_tensor(real_train_dl)
        test_data, test_condition = loader_to_cond_tensor(real_test_dl)
        real_data = torch.cat([train_data, test_data])
        condition = torch.cat([train_condition, test_condition])
    else:
        real_data = torch.cat(
            [loader_to_tensor(real_train_dl), loader_to_tensor(real_test_dl)]
        )
    dim = real_data.shape[-1]

    sig_depth = 5
    d_dim = 16
    n_layers = 2
    for i in tqdm(range(10)):
        # take random 10000 samples from real dataset
        idx = torch.randint(real_data.shape[0], (10000,))
        real_train_dl = DataLoader(
            TensorDataset(real_data[idx[:-2000]]), batch_size=128
        )
        real_test_dl = DataLoader(TensorDataset(real_data[idx[-2000:]]), batch_size=128)

        if "recovery" in kwargs:
            recovery = kwargs["recovery"]
            if config.generator == "Conditional_LSTM":
                train_condition = condition[idx[:-2000]]
                test_condition = condition[idx[-2000:]]
                fake_train_dl = fake_loader(
                    generator,
                    num_samples=8000,
                    n_lags=config.n_lags,
                    batch_size=128,
                    config=config,
                    condition=train_condition,
                    recovery=recovery,
                )
                fake_test_dl = fake_loader(
                    generator,
                    num_samples=2000,
                    n_lags=config.n_lags,
                    batch_size=128,
                    config=config,
                    condition=test_condition,
                    recovery=recovery,
                )
            else:
                fake_train_dl = fake_loader(
                    generator,
                    num_samples=8000,
                    n_lags=config.n_lags,
                    batch_size=128,
                    config=config,
                    recovery=recovery,
                )
                fake_test_dl = fake_loader(
                    generator,
                    num_samples=2000,
                    n_lags=config.n_lags,
                    batch_size=128,
                    config=config,
                    recovery=recovery,
                )
        else:
            if config.generator == "Conditional_LSTM":
                train_condition = condition[idx[:-2000]]
                test_condition = condition[idx[-2000:]]
                fake_train_dl = fake_loader(
                    generator,
                    num_samples=8000,
                    n_lags=config.n_lags,
                    batch_size=128,
                    config=config,
                    condition=train_condition,
                )
                fake_test_dl = fake_loader(
                    generator,
                    num_samples=2000,
                    n_lags=config.n_lags,
                    batch_size=128,
                    config=config,
                    condition=test_condition,
                )
            else:
                fake_train_dl = fake_loader(
                    generator,
                    num_samples=8000,
                    n_lags=config.n_lags,
                    batch_size=128,
                    config=config,
                )
                fake_test_dl = fake_loader(
                    generator,
                    num_samples=2000,
                    n_lags=config.n_lags,
                    batch_size=128,
                    config=config,
                )
        real = torch.cat(
            [loader_to_tensor(real_train_dl), loader_to_tensor(real_test_dl)]
        )
        fake = torch.cat(
            [loader_to_tensor(fake_train_dl), loader_to_tensor(fake_test_dl)]
        )
        '''
        d_score_mean, d_score_std = compute_discriminative_score(
            real_train_dl,
            real_test_dl,
            fake_train_dl,
            fake_test_dl,
            config,
            d_dim,
            n_layers,
            epochs=30,
            batch_size=128,
        )
        d_scores.append(d_score_mean)
        p_score_mean, p_score_std = compute_predictive_score(
            real_train_dl,
            real_test_dl,
            fake_train_dl,
            fake_test_dl,
            config,
            32,
            2,
            epochs=50,
            batch_size=128,
        )
        p_scores.append(p_score_mean)


        sig_mmd = Sig_mmd(real, fake, depth=sig_depth)

        while sig_mmd.abs() > 1e3:
            sig_mmd = Sig_mmd(real, fake, depth=sig_depth)
        Sig_MMDs.append(sig_mmd)
        '''
        #acf_score1 = acf_metric(real,fake,1)
        #acf_score5 = acf_metric(real,fake,5)
        #ccf_score1 = ccf_metric(real,fake,0)
        #ccf_score5 = ccf_metric(real,fake,5)
        #acf1.append(acf_score1)
        #acf5.append(acf_score5)
        #ccf1.append(ccf_score1)
        #ccf5.append(ccf_score5)
        
        #m_metric = HistoLoss(real)
        #m_score = m_metric.compute(fake)
        #m_scores.append(m_score.detach().numpy())



    # plot_samples(real, fake, config)
    #d_mean, d_std = np.array(d_scores).mean(), np.array(d_scores).std()
    #p_mean, p_std = np.array(p_scores).mean(), np.array(p_scores).std()
    #sig_mmd_mean, sig_mmd_std = np.array(Sig_MMDs).mean(), np.array(Sig_MMDs).std()
    #acf1_mean,acf1_std = np.array(acf1).mean(),np.array(acf1).std()
    #acf5_mean,acf5_std = np.array(acf5).mean(),np.array(acf5).std()
    #ccf1_mean,ccf1_std = np.array(ccf1).mean(),np.array(ccf1).std()
    #m_mean,m_std = np.array(m_scores).mean(),np.array(m_scores).std()
    #plot_summary(fake_test_dl, real_test_dl, config)
    plot_tsne(real_test_dl,fake_test_dl,config=config)
    #logging.info("Evaluation results on model:{} ".format(config.gan_algo))
    #logging.info("m_score with mean: {},std: {}".format(m_mean, m_std))
    #logging.info("discriminative score with mean:{},std: {}".format(d_mean, d_std))
    #logging.info("predictive score with mean:{},std: {}".format(p_mean, p_std))
    #logging.info("sig mmd with mean: {},std: {}".format(sig_mmd_mean, sig_mmd_std))
    #logging.info("acf1 with mean: {},std: {}".format(acf1_mean, acf1_std))
    #logging.info("acf5 with mean: {},std: {}".format(acf5_mean, acf5_std))
    #logging.info("ccf0 with mean: {},std: {}".format(ccf1_mean, ccf1_std))
    #logging.info("ccf5 with mean: {},std: {}".format(ccf5_mean, ccf5_std))
    return None, None , None


def get_reconstructed_data(config):
    get_experiment_dir(config)
    from src.datasets.dataloader import get_dataset

    train_dl, test_dl = get_dataset(config, num_workers=4)
    from src.models import get_trainer

    trainer, generator = get_trainer(config, train_dl, test_dl)

    if config.gan_algo == "TimeGAN":
        generator = generator.to(device="cpu")
        generator.load_state_dict(
            torch.load(pt.join(config.exp_dir, "generator_state_dict.pt")), strict=True
        )
        embedder = trainer.embedder.to(device="cpu")
        embedder.load_state_dict(
            torch.load(pt.join(config.exp_dir, "embedder_state_dict.pt")), strict=True
        )
        recovery = trainer.recovery.to(device="cpu")
        recovery.load_state_dict(
            torch.load(pt.join(config.exp_dir, "recovery_state_dict.pt")), strict=True
        )

        d = nn.Sequential(embedder, recovery)
        generator.eval()
    else:
        print(config.exp_dir)
        generator = generator.to(device="cpu")
        generator.load_state_dict(
            torch.load(pt.join(config.exp_dir, "generator_state_dict.pt")), strict=True
        )
        d = trainer.D.to(device="cpu")
        d.load_state_dict(
            torch.load(pt.join(config.exp_dir, "discriminator_state_dict.pt")),
            strict=True,
        )
        generator.eval()
    real_X = torch.cat([loader_to_tensor(train_dl), loader_to_tensor(test_dl)])
    if config.gan_algo == "TimeGAN":
        rec_X = recovery(real_X)

    else:
        z = d(real_X)
        rec_X = generator(
            batch_size=real_X.shape[0], n_lags=config.n_lags, device="cpu", z=z
        )

    return real_X.detach(), rec_X.detach()


def evaluate_reconstruction(config):
    """Evaluate the reconstruction performance of the generator or recovery model.

    Args:
        config: Configuration object.

    Returns:
        Tuple: Mean of the discriminative score, mean of the predictive score, mean of the Sig MMD score.
    """
    real_data, rec_data = get_reconstructed_data(config)
    real_data = real_data.to(config.device)
    rec_data = rec_data.to(config.device)
    d_scores = []
    p_scores = []
    Sig_MMDs = []
    sig_depth = 5
    d_dim = 16
    n_layers = 2
    for i in tqdm(range(10)):
        # take random 10000 samples from real dataset
        idx = torch.randint(real_data.shape[0], (10000,))
        real_train_dl = DataLoader(
            TensorDataset(real_data[idx[:-2000]]), batch_size=128
        )
        real_test_dl = DataLoader(TensorDataset(real_data[idx[-2000:]]), batch_size=128)
        fake_train_dl = DataLoader(TensorDataset(rec_data[idx[:-2000]]), batch_size=128)
        fake_test_dl = DataLoader(TensorDataset(rec_data[idx[-2000:]]), batch_size=128)

        d_score_mean, d_score_std = compute_discriminative_score(
            real_train_dl,
            real_test_dl,
            fake_train_dl,
            fake_test_dl,
            config,
            d_dim,
            n_layers,
            epochs=30,
            batch_size=128,
        )
        d_scores.append(d_score_mean)
        p_score_mean, p_score_std = compute_predictive_score(
            real_train_dl,
            real_test_dl,
            fake_train_dl,
            fake_test_dl,
            config,
            32,
            2,
            epochs=50,
            batch_size=128,
        )
        p_scores.append(p_score_mean)
        real = torch.cat(
            [loader_to_tensor(real_train_dl), loader_to_tensor(real_test_dl)]
        )
        fake = torch.cat(
            [loader_to_tensor(fake_train_dl), loader_to_tensor(fake_test_dl)]
        )

        sig_mmd = Sig_mmd(real, fake, depth=sig_depth)
        while sig_mmd.abs() > 1e3:
            sig_mmd = Sig_mmd(real, fake, depth=sig_depth)
        Sig_MMDs.append(sig_mmd)

    # plot_samples(real, fake, config)
    d_mean, d_std = np.array(d_scores).mean(), np.array(d_scores).std()
    p_mean, p_std = np.array(p_scores).mean(), np.array(p_scores).std()
    sig_mmd_mean, sig_mmd_std = np.array(Sig_MMDs).mean(), np.array(Sig_MMDs).std()
    plot_summary(fake_test_dl, real_test_dl, config)
    logging.info("Evaluation results on model:{} ".format(config.gan_algo))
    logging.info("discriminative score with mean:{},std: {}".format(d_mean, d_std))
    logging.info("predictive score with mean:{},std: {}".format(p_mean, p_std))
    logging.info("sig mmd with mean: {},std: {}".format(sig_mmd_mean, sig_mmd_std))
    return d_mean, p_mean, sig_mmd_mean
