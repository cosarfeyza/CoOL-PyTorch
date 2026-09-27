# R equivalent: CoOL_1_initiate_neural_network (architecture + init; CoOL_functions.R)
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import parametrize


class NonNegativeNN(nn.Module):
    def __init__(self, n_inputs: int, n_hidden: int, use_c: bool = False, dtype=torch.float64):
        super().__init__()
        self.use_c = use_c
        self.fc1 = nn.Linear(n_inputs, n_hidden, bias=True, dtype=dtype)   # weight (H,D), bias (H,)
        self.fc2 = nn.Linear(n_hidden, 1, bias=True, dtype=dtype)          # weight (1,H), bias scalar
        if self.use_c:
            # c2 is a scalar multiplier for the confounder c
            self.c2 = nn.Parameter(torch.ones(1, dtype=dtype))
        else:
            self.register_parameter("c2", None)

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (N, D)
        c: (N,) or (N,1) if use_c=True, else ignored
        """
        h = F.relu(self.fc1(x))           # CoOL typically uses non-negativity
        y = self.fc2(h)                   # (N,1)
        if self.use_c and c is not None:
            y = y + self.c2 * c.view(-1, 1)
        # relu on the output too - matches CoOL_4_predict_risks in the R package
        # and the trainer (cool_step_arma.cpp: rcpprelu(h*W2 + B2 + c*C2))
        return F.relu(y)
    

    '''
    R-like CoOL initialization of the neural network weights
    - fc1.weight: small positive values (Exponential distribution)
    - fc1.bias: small negative values (negative Exponential distribution)
    - fc2.weight: all ones
    - fc2.bias: baseline risk = mean(Y)
    - c2 (if use_c): small positive value (Exponential distribution)
    rate: rate parameter for Exponential distribution (default 100.0)
    device, dtype: if provided, tensors are created on the specified device and dtype   
    seed: random seed for reproducibility
    '''


    def initiate_neural_network(
        self,
        output,
        rate: float = 100.0,
        seed: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):

        # device / dtype otomatik al
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        if seed is not None:
            torch.manual_seed(seed)

        y = torch.as_tensor(output, dtype=dtype, device=device).view(-1)
        exp_dist = torch.distributions.Exponential(rate)

        with torch.no_grad():
            # fc1.weight: small positive values
            w1_sample = exp_dist.sample(self.fc1.weight.shape).to(device=device, dtype=dtype)
            self.fc1.weight.copy_(w1_sample)

            # fc1.bias: small negative values
            b1_sample = -exp_dist.sample(self.fc1.bias.shape).to(device=device, dtype=dtype)
            self.fc1.bias.copy_(b1_sample)

            # fc2.weight: all ones
            w2_ones = torch.ones_like(self.fc2.weight, device=device, dtype=dtype)
            self.fc2.weight.copy_(w2_ones)

            # fc2.bias: baseline = mean(Y)
            self.fc2.bias.copy_(y.mean())

            # c2 varsa: small positive value
            if self.use_c and getattr(self, "c2", None) is not None:
                c2_sample = exp_dist.sample(self.c2.shape).to(device=device, dtype=dtype)
                self.c2.copy_(c2_sample)

        return self     
