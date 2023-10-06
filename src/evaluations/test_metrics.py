from functools import partial
from typing import Tuple
from src.utils import to_numpy
import torch
from torch import nn
import numpy as np
import ksig
from src.utils import AddTime


def q_var_torch(x: torch.Tensor):
    """
    :param x: torch.Tensor [B, S, D]
    :return: quadratic variation of x. [B, D]
    """
    return torch.sum(torch.pow(x[:, 1:] - x[:, :-1], 2), 1)


def acf_torch(x: torch.Tensor, max_lag: int, dim: Tuple[int] = (0, 1)) -> torch.Tensor:
    """
    :param x: torch.Tensor [B, S, D]
    :param max_lag: int. specifies number of lags to compute the acf for
    :return: acf of x. [max_lag, D]
    """
    acf_list = list()
    x = x - x.mean((0, 1))
    std = torch.var(x, unbiased=False, dim=(0, 1))
    for i in range(max_lag):
        y = x[:, i:] * x[:, :-i] if i > 0 else torch.pow(x, 2)
        acf_i = torch.mean(y, dim) / std
        acf_list.append(acf_i)
    if dim == (0, 1):
        return torch.stack(acf_list)
    else:
        return torch.cat(acf_list, 1)


def ccf_mean(x: torch.tensor):
    ccf = torch.from_numpy(np.corrcoef(x.mean(1).permute(1, 0))).float()
    n = x.shape[-1]
    # print()
    num_cross = (n**2 - n) / 2
    # print(ccf.triu(1).mean())
    return ccf.triu(1).mean() * ((n**2) / num_cross)

def ccf_metric(x,y,lag=1):
    ccf_x = cacf_torch(
            x, lags=6)
    ccf_y = cacf_torch(
            y, lags=6)
    
    return torch.abs(ccf_x-ccf_y).sum(1)[lag]

def acf_metric(x,y,dim):
    acf_x = acf_torch(x,10)
    acf_y = acf_torch(y,10)
    return torch.abs(acf_x-acf_y).sum(1)[dim]

def cacf_torch(x, lags: list, dim=(0, 1)):
    
    # Define a helper function to get the lower triangular indices for a given dimension
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    # Get the lower triangular indices for the input tensor x
    ind = get_lower_triangular_indices(x.shape[2])

    # Standardize the input tensor x along the given dimensions
    #x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)
    x = x - x.mean((0, 1))
    # Split the input tensor into left and right parts based on the lower triangular indices
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]

    # Compute the cross-correlation at each lag and store in a list
    cacf_list = list()
    std = torch.std(x_l, unbiased=False, dim=(0, 1))*torch.std(x_r, unbiased=False, dim=(0, 1))
    for i in range(lags):
        # Compute the element-wise product of the left and right parts, shifted by the lag if i > 0
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r

        # Compute the mean of the product along the time dimension
        cacf_i = torch.mean(y, dim) / std

        # Append the result to the list of cross-correlations
        cacf_list.append(cacf_i)

    # Concatenate the cross-correlations across lags and reshape to the desired output shape
    if dim == (0, 1):
        return torch.stack(cacf_list)
    else:
        return torch.cat(cacf_list, 1)
    #return cacf.reshape(cacf.shape[0], -1, len(ind[0]))



def histogram_torch(x, n_bins, density=True):
    a, b = x.min().item(), x.max().item()
    b = b + 1e-5 if b == a else b
    # delta = (b - a) / n_bins
    bins = torch.linspace(a, b, n_bins + 1)
    delta = bins[1] - bins[0]
    # bins = torch.arange(a, b + 1.5e-5, step=delta)
    count = torch.histc(x, bins=n_bins, min=a, max=b).float()
    if density:
        count = count / delta / float(x.shape[0] * x.shape[1])
    return count, bins


class Loss(nn.Module):
    def __init__(
        self,
        name,
        reg=1.0,
        transform=lambda x: x,
        threshold=10.0,
        backward=False,
        norm_foo=lambda x: x,
    ):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)

class HistoLoss(nn.Module):
    def __init__(self, x_real, n_bins=80, **kwargs):
        super(HistoLoss, self).__init__(**kwargs)
        self.densities = list()
        self.locs = list()
        self.deltas = list()
        for i in range(x_real.shape[2]):
            tmp_densities = list()
            tmp_locs = list()
            tmp_deltas = list()
            # Exclude the initial point
            for t in range(x_real.shape[1]):
                x_ti = x_real[:, t, i].reshape(-1, 1)
                d, b = histogram_torch(x_ti, n_bins, density=True)
                tmp_densities.append(nn.Parameter(d).to(x_real.device))
                delta = b[1:2] - b[:1]
                loc = 0.5 * (b[1:] + b[:-1])
                tmp_locs.append(loc)
                tmp_deltas.append(delta)
            self.densities.append(tmp_densities)
            self.locs.append(tmp_locs)
            self.deltas.append(tmp_deltas)

    def compute(self, x_fake):
        loss = list()

        def relu(x):
            return x * (x >= 0.0).float()

        for i in range(x_fake.shape[2]):
            tmp_loss = list()
            # Exclude the initial point
            for t in range(x_fake.shape[1]):
                loc = self.locs[i][t].view(1, -1).to(x_fake.device)
                x_ti = x_fake[:, t, i].contiguous().view(-1, 1).repeat(1, loc.shape[1])
                dist = torch.abs(x_ti - loc)
                counter = (
                    relu(self.deltas[i][t].to(x_fake.device) / 2.0 - dist) > 0.0
                ).float()
                density = counter.mean(0) / self.deltas[i][t].to(x_fake.device)
                abs_metric = torch.abs(density - self.densities[i][t].to(x_fake.device))
                loss.append(torch.mean(abs_metric, 0))
        loss_componentwise = torch.stack(loss)
        return loss_componentwise


def non_stationary_acf_torch(X, covariance: bool = False, symmetric=False):
    """
    Compute the correlation matrix between any two time points of the time series
    Parameters
    ----------
    X (torch.Tensor): [B, T, D]
    symmetric (bool): whether to return the upper triangular matrix of the full matrix

    Returns
    -------
    Correlation matrix of the shape [T, T, D] where each entry (t_i, t_j, d_i) is the correlation between the d_i-th coordinate of X_{t_i} and X_{t_j}
    """
    # Get the batch size, sequence length, and input dimension from the input tensor
    B, T, D = X.shape

    # Create a tensor to hold the correlations
    correlations = torch.zeros(T, T, D)

    # Loop through each time step from lag to T-1
    for t in range(T):
        # Loop through each lag from 1 to lag
        for tau in range(t, T):
            # Compute the correlation between X_{t, d} and X_{t-tau, d}
            if covariance:
                correlation = torch.sum(
                    (X[:, t, :] - torch.mean(X[:, t, :], 0))
                    * (X[:, tau, :] - torch.mean(X[:, tau, :], 0)),
                    dim=0,
                )
            else:
                correlation = torch.sum(
                    (X[:, t, :] - torch.mean(X[:, t, :], 0))
                    * (X[:, tau, :] - torch.mean(X[:, tau, :], 0)),
                    dim=0,
                ) / (
                    torch.norm(X[:, t, :] - torch.mean(X[:, t, :], 0), dim=0)
                    * torch.norm(X[:, tau, :] - torch.mean(X[:, tau, :], 0), dim=0)
                )
            # print(correlation)
            # Store the correlation in the output tensor
            correlations[t, tau, :] = correlation
            if symmetric:
                correlations[tau, t, :] = correlation

    return correlations


class Loss(nn.Module):
    def __init__(
        self,
        name,
        reg=1.0,
        transform=lambda x: x,
        threshold=10.0,
        backward=False,
        norm_foo=lambda x: x,
    ):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)


def Sig_mmd(X, Y, depth):
    # convert torch tensor to numpy
    N, L, C = X.shape
    N1, _, C1 = Y.shape
    X = torch.cat([torch.zeros((N, 1, C)).to(X.device), X], dim=1)
    Y = torch.cat([torch.zeros((N1, 1, C1)).to(X.device), Y], dim=1)
    X = to_numpy(AddTime(X))
    Y = to_numpy(AddTime(Y))
    n_components = 20
    static_kernel = ksig.static.kernels.RBFKernel()
    # an RBF base kernel for vector-valued data which is lifted to a kernel for sequences
    static_feat = ksig.static.features.NystroemFeatures(
        static_kernel, n_components=n_components
    )
    # Nystroem features with an RBF base kernel

    proj = ksig.projections.CountSketchRandomProjection(n_components=n_components)
    # a CountSketch random projection

    lr_sig_kernel = ksig.kernels.LowRankSignatureKernel(
        n_levels=depth, static_features=static_feat, projection=proj
    )
    # sig_kernel = ksig.kernels.SignatureKernel(
    #   n_levels=depth, static_kernel=static_kernel)
    # a SignatureKernel object, which works as a callable for computing the signature kernel matrix
    lr_sig_kernel.fit(X)
    K_XX = lr_sig_kernel(X)  # K_XX has shape (10, 10)
    K_XY = lr_sig_kernel(X, Y)
    K_YY = lr_sig_kernel(Y)
    m = K_XX.shape[0]
    diag_X = np.diagonal(K_XX)
    diag_Y = np.diagonal(K_YY)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()
    mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
    mmd2 -= 2 * K_XY_sum / (m * m)

    return torch.tensor(mmd2)


def Sig_mmd_small(X, Y, depth):
    # convert torch tensor to numpy
    N, L, C = X.shape
    N1, _, C1 = Y.shape
    X = torch.cat([torch.zeros((N, 1, C)).to(X.device), X], dim=1)
    Y = torch.cat([torch.zeros((N1, 1, C1)).to(X.device), Y], dim=1)
    X = to_numpy(AddTime(X))
    Y = to_numpy(AddTime(Y))
    n_components = 100
    static_kernel = ksig.static.kernels.RBFKernel()
    # an RBF base kernel for vector-valued data which is lifted to a kernel for sequences
    sig_kernel = ksig.kernels.SignatureKernel(depth, static_kernel=static_kernel)
    # Nystroem features with an RBF base kernel

    K_XX = sig_kernel(X)  # K_XX has shape (10, 10)
    K_XY = sig_kernel(X, Y)
    K_YY = sig_kernel(Y)
    m = K_XX.shape[0]
    diag_X = np.diagonal(K_XX)
    diag_Y = np.diagonal(K_YY)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()
    mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
    mmd2 -= 2 * K_XY_sum / (m * m)

    return torch.tensor(mmd2)


class Sig_MMD_loss(Loss):
    def __init__(self, x_real, depth, **kwargs):
        super(Sig_MMD_loss, self).__init__(**kwargs)
        self.x_real = x_real
        self.depth = depth

    def compute(self, x_fake):
        return Sig_mmd_small(self.x_real, x_fake, self.depth)


class cross_correlation(Loss):
    def __init__(self, x_real, **kwargs):
        super(cross_correlation).__init__(**kwargs)
        self.x_real = x_real

    def compute(self, x_fake):
        fake_corre = torch.from_numpy(np.corrcoef(x_fake.mean(1).permute(1, 0))).float()
        real_corre = torch.from_numpy(
            np.corrcoef(self.x_real.mean(1).permute(1, 0))
        ).float()
        return torch.abs(fake_corre - real_corre)


def skew_torch(x, dim=(0, 1), dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_3 = torch.pow(x, 3).mean(dim, keepdims=True)
    x_std_3 = torch.pow(x.std(dim, unbiased=True, keepdims=True), 3)
    skew = x_3 / x_std_3
    if dropdims:
        skew = skew[0, 0]
    return skew


def kurtosis_torch(x, dim=(0, 1), excess=True, dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_4 = torch.pow(x, 4).mean(dim, keepdims=True)
    x_var2 = torch.pow(torch.var(x, dim=dim, unbiased=False, keepdims=True), 2)
    kurtosis = x_4 / x_var2
    if excess:
        kurtosis = kurtosis - 3
    if dropdims:
        kurtosis = kurtosis[0, 0]
    return kurtosis


def diff(x):
    return x[:, 1:] - x[:, :-1]


test_metrics = {
    "Sig_mmd": partial(Sig_MMD_loss, name="Sig_mmd", depth=4),
}


def is_multivariate(x: torch.Tensor):
    """Check if the path / tensor is multivariate."""
    return True if x.shape[-1] > 1 else False


def get_standard_test_metrics(x: torch.Tensor, **kwargs):
    """Initialise list of standard test metrics for evaluating the goodness of the generator."""
    test_metrics_list = [
        test_metrics["Sig_mmd"](x, depth=4),
    ]
    return test_metrics_list
