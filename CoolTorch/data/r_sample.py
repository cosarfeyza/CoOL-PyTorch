import numpy as np
import torch

"""
    CoOL-style random generator.
    Distributionally equivalent to R:
        rgamma(1, shape=1, rate=rate)

    Returns a tensor of given shape.

    To use in initialization of the CoOL model.
"""
def random(
    shape,
    rate: float = 100.0,
    seed: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
):


    if seed is not None:
        torch.manual_seed(seed)

    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float64

    gamma = torch.distributions.Gamma(
        concentration=torch.tensor(1.0, device=device, dtype=dtype),
        rate=torch.tensor(float(rate), device=device, dtype=dtype),
    )

    return gamma.sample(shape)

