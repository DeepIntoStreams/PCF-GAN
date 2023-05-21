from functools import partial
import torch
import torch.nn as nn
import numpy as np
from src.PCFGAN.unitary import unitary, unitary_lie_init_


def matrix_power_two_batch(A, k):
    """
    Computes the matrix power of A for each element in k using batch processing.

    Args:
        A (torch.Tensor): Input tensor of shape (..., m, m).
        k (torch.Tensor): Exponent tensor of shape (...).

    Returns:
        torch.Tensor: Resulting tensor of shape (..., m, m).
    """
    orig_size = A.size()
    A, k = A.flatten(0, -3), k.flatten()
    ksorted, idx = torch.sort(k)
    # Abusing bincount...
    count = torch.bincount(ksorted)
    nonzero = torch.nonzero(count, as_tuple=False)
    A = torch.matrix_power(A, 2 ** ksorted[0])
    last = ksorted[0]
    processed = count[nonzero[0]]
    for exp in nonzero[1:]:
        new, last = exp - last, exp
        A[idx[processed:]] = torch.matrix_power(A[idx[processed:]], 2 ** new.item())
        processed += count[exp]
    return A.reshape(orig_size)


def rescaled_matrix_exp(f, A):
    """
    Computes the rescaled matrix exponential of A.
    By following formula exp(A) = (exp(A/k))^k

    Args:
        f (callable): Function to compute the matrix exponential.
        A (torch.Tensor): Input tensor of shape (..., m, m).

    Returns:
        torch.Tensor: Resulting tensor of shape (..., m, m).
    """
    normA = torch.max(torch.sum(torch.abs(A), axis=-2), axis=-1).values
    more = normA > 1
    s = torch.ceil(torch.log2(normA)).long()
    s = normA.new_zeros(normA.size(), dtype=torch.long)
    s[more] = torch.ceil(torch.log2(normA[more])).long()
    A_1 = torch.pow(0.5, s.float()).unsqueeze_(-1).unsqueeze_(-1).expand_as(A) * A
    return matrix_power_two_batch(f(A_1), s)


class projection(nn.Module):
    def __init__(self, input_size, hidden_size, channels=1, init_range=1, **kwargs):
        """
        Projection module used to project the path increments to the Lie group path increments
        using trainable weights from the Lie algebra.

        Args:
            input_size (int): Input size.
            hidden_size (int): Size of the hidden Lie algebra matrix.
            channels (int, optional): Number of channels to produce independent Lie algebra weights. Defaults to 1.
            init_range (int, optional): Range for weight initialization. Defaults to 1.
        """
        self.__dict__.update(kwargs)

        A = torch.empty(
            input_size, channels, hidden_size, hidden_size, dtype=torch.cfloat
        )
        self.channels = channels
        super(projection, self).__init__()
        self.param_map = unitary(hidden_size)
        self.A = nn.Parameter(A)

        self.triv = torch.linalg.matrix_exp
        self.init_range = init_range
        self.reset_parameters()

        self.hidden_size = hidden_size

    def reset_parameters(self):
        unitary_lie_init_(self.A, partial(nn.init.normal_, std=1))

    def M_initialize(self, A):
        init_range = np.linspace(0, 10, self.channels + 1)
        for i in range(self.channels):
            A[:, i] = unitary_lie_init_(
                A[:, i], partial(nn.init.uniform_, a=init_range[i], b=init_range[i + 1])
            )
        return A

    def forward(self, dX: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the projection module.

        Args:
            dX (torch.Tensor): Tensor of shape (N, input_size).

        Returns:
            torch.Tensor: Tensor of shape (N, channels, hidden_size, hidden_size).
        """
        A = self.param_map(self.A).permute(1, 2, -1, 0)  # C,m,m,in
        AX = A.matmul(dX.T).permute(-1, 0, 1, 2)  # ->C,m,m,N->N,C,m,m

        return rescaled_matrix_exp(self.triv, AX)


class development_layer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        channels: int = 1,
        include_inital: bool = False,
        time_batch=1,
        return_sequence=False,
        init_range=1,
    ):
        """
        Development layer module used for computation of unitary feature on time series.

        Args:
            input_size (int): Input size.
            hidden_size (int): Size of the hidden matrix.
            channels (int, optional): Number of channels. Defaults to 1.
            include_inital (bool, optional): Whether to include the initial value in the input. Defaults to False.
            time_batch (int, optional): Truncation value for batch processing. Defaults to 1.
            return_sequence (bool, optional): Whether to return the entire sequence or just the final output. Defaults to False.
            init_range (int, optional): Range for weight initialization. Defaults to 1.
        """
        super(development_layer, self).__init__()
        self.input_size = input_size
        self.channels = channels
        self.hidden_size = hidden_size
        self.projection = projection(
            input_size, hidden_size, channels, init_range=init_range
        )
        self.include_inital = include_inital
        self.truncation = time_batch
        self.complex = True
        self.return_sequence = return_sequence

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the development layer module.

        Args:
            input (torch.Tensor): Tensor with shape (N, T, input_size).

        Returns:
            torch.Tensor: Tensor with shape (N, T, hidden_size, hidden_size).
        """
        if self.complex:
            input = input.cfloat()

        N, T, C = input.shape
        if self.include_inital:
            input = torch.cat([torch.zeros((N, 1, C)).to(input.device), input], dim=1)

        dX = input[:, 1:] - input[:, :-1]
        # N,T-1,input_size

        M_dX = self.projection(dX.reshape(-1, dX.shape[-1])).reshape(
            N, -1, self.channels, self.hidden_size, self.hidden_size
        )

        return self.dyadic_prod(M_dX)

    @staticmethod
    def dyadic_prod(X: torch.Tensor) -> torch.Tensor:
        """
        Computes the cumulative product on matrix time series with dyadic partitioning.

        Args:
            X (torch.Tensor): Batch of matrix time series of shape (N, T, C, m, m).

        Returns:
            torch.Tensor: Cumulative product on the time dimension of shape (N, T, m, m).
        """
        N, T, C, m, m = X.shape
        max_level = int(torch.ceil(torch.log2(torch.tensor(T))))
        I = (
            torch.eye(m, device=X.device, dtype=X.dtype)
            .reshape(1, 1, 1, m, m)
            .repeat(N, 1, C, 1, 1)
        )
        for i in range(max_level):
            if X.shape[1] % 2 == 1:
                X = torch.cat([X, I], 1)
            X = X.reshape(-1, 2, C, m, m)
            X = torch.einsum("bcij,bcjk->bcik", X[:, 0], X[:, 1])
            X = X.reshape(N, -1, C, m, m)

        return X[:, 0]
