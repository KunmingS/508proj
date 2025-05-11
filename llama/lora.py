import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=16, alpha=32, dropout=0.05, bias=False):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)

        #freeze the weights and only forward pass.
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

        #LoRA learnable weights.
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, r) * 0.01)

        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=bias)

    def forward(self, x):
        lora = self.dropout(x) @ self.A.T @ self.B.T * self.scale
        return F.linear(x, self.weight, self.bias) + lora