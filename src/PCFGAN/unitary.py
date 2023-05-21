import torch
from torch import nn
import math


def unitary_lie_init_(tensor: torch.tensor, init_=None):
    """
    Fills in the input tensor in place with initialization on the unitary Lie algebra.

    Args:
        tensor (torch.Tensor): A multi-dimensional tensor.
        init_ (callable): Optional. A function that initializes the tensor according to some distribution.

    Raises:
        ValueError: If the tensor has less than 2 dimensions or the last two dimensions are not square.

    """
    if tensor.ndim < 2 or tensor.size(-1) != tensor.size(-2):
        raise ValueError(
            "Only tensors with 2 or more dimensions which are square in "
            "the last two dimensions are supported. "
            "Got a tensor of shape {}".format(tuple(tensor.size()))
        )

    n = tensor.size(-2)
    tensorial_size = tensor.size()[:-2]

    # Non-zero elements that we are going to set on the diagonal
    n_diag = n
    diag = tensor.new(tensorial_size + (n_diag,))
    if init_ is None:
        torch.nn.init.uniform_(diag, -math.pi, math.pi)
    else:
        init_(diag)
    diag = diag.imag * torch.tensor([1j], device=tensor.device)
    # set values for upper trianguler matrix
    off_diag = tensor.new(tensorial_size + (2 * n, n))
    if init_ is None:
        torch.nn.init.uniform_(off_diag, -math.pi, math.pi)

    else:
        init_(off_diag)

    upper_tri_real = torch.triu(off_diag[..., :n, :n], 1).real.cfloat()
    upper_tri_complex = torch.triu(
        off_diag[..., n:, :n], 1
    ).imag.cfloat() * torch.tensor([1j], device=tensor.device)

    real_part = (upper_tri_real - upper_tri_real.transpose(-2, -1)) / torch.tensor(
        [2], device=tensor.device
    ).cfloat().sqrt()
    complex_part = (
        upper_tri_complex + upper_tri_complex.transpose(-2, -1)
    ) / torch.tensor([2], device=tensor.device).cfloat().sqrt()

    with torch.no_grad():
        # First non-central diagonal
        x = real_part + complex_part + torch.diag_embed(diag)
        if unitary(n).in_lie_algebra(x):
            tensor = tensor.cfloat()
            tensor.copy_(x)
            return tensor
        else:
            raise ValueError("initialize not in Lie")


class unitary(nn.Module):

    """
    A PyTorch module for parametrizing a unitary Lie algebra using a general linear matrix.
    """

    def __init__(self, size):
        """
        Initializes the unitary module.

        Args:
            size (torch.size): Size of the tensor to be parametrized.
        """
        super().__init__()
        self.size = size

    @staticmethod
    def frame(X: torch.tensor) -> torch.tensor:
        """
        Parametrizes the unitary Lie algebra from the general linear matrix X.

        Args:
            X (torch.Tensor): A tensor of shape (...,2n,2n).

        Returns:
            torch.Tensor: A tensor of shape (...,2n,2n).
        """
        X = (X - torch.conj(X.transpose(-2, -1))) / 2

        return X

    def forward(self, X: torch.tensor) -> torch.tensor:
        """
        Applies the frame method to the input tensor X.

        Args:
            X (torch.Tensor): A tensor to be parametrized.

        Returns:
            torch.Tensor: A tensor parametrized in the unitary Lie algebra.

        Raises:
            ValueError: If the input tensor has fewer than 2 dimensions or the last two dimensions are not square.
        """
        if len(X.size()) < 2:
            raise ValueError("weights has dimension < 2")
        if X.size(-2) != X.size(-1):
            raise ValueError("not sqaured matrix")
        return self.frame(X)

    @staticmethod
    def in_lie_algebra(X, eps=1e-5):
        """
        Checks if a given tensor is in the Lie algebra.

        Args:
            X: The tensor to check.
            eps (float): Optional. The tolerance for checking closeness to zero.

        Returns:
            bool: True if the tensor is in the Lie algebra, False otherwise.
        """
        return (
            X.dim() >= 2
            and X.size(-2) == X.size(-1)
            and torch.allclose(torch.conj(X.transpose(-2, -1)), -X, atol=eps)
        )
