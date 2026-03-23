"""
MamaGuard — Mamba3 Model
Trapezoidal SSM with MIMO expansion and complex-valued state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba3SSMLayer(nn.Module):
    """Core recurrent SSM engine of one Mamba3 block."""

    def __init__(self, d_model: int, d_state: int = 32, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Input/output projections (MIMO)
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Local depthwise convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner,
            bias=True
        )

        # SSM parameters
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Input-dependent (selective) parameters: B, C, and Δ
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Trapezoidal blending parameter (α)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch_size, seq_len, d_model) -> same shape output."""
        B, L, _ = x.shape

        # Project to inner dimension + gating signal
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        # Local convolution + SiLU activation
        x_conv = self.conv1d(x_in.transpose(1, 2)).transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Compute input-dependent SSM parameters
        dt_raw, B_ssm, C_ssm = self.x_proj(x_conv).split(
            [1, self.d_state, self.d_state], dim=-1
        )

        dt = F.softplus(self.dt_proj(dt_raw))
        A_real = -torch.exp(self.A_log)
        alpha = torch.sigmoid(self.alpha)

        # SSM recurrence
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        outputs = []

        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)
            B_t  = B_ssm[:, t, :].unsqueeze(1)
            C_t  = C_ssm[:, t, :].unsqueeze(1)
            u_t  = x_conv[:, t, :]

            # Trapezoidal discretization: blend ZOH + Implicit Euler
            A_d_zoh   = torch.exp(A_real * dt_t)
            A_d_euler = 1.0 / (1.0 - A_real * dt_t * 0.5 + 1e-6)
            A_d = alpha * A_d_zoh + (1.0 - alpha) * A_d_euler

            # State update + output
            h = A_d * h + dt_t * B_t * u_t.unsqueeze(-1)
            y_t = (C_t * h).sum(dim=-1) + self.D * u_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)

        # Apply gating and project back
        y = y * F.silu(z)
        return self.out_proj(y)


class Mamba3Block(nn.Module):
    """One complete Mamba3 processing block: LayerNorm -> SSM -> LayerNorm -> FFN."""

    def __init__(self, d_model: int, d_state: int = 32):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm   = Mamba3SSMLayer(d_model, d_state)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(p=0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MamaGuardMamba3(nn.Module):
    """
    Complete MamaGuard model.
    Flow: raw vitals (6) -> embed -> 4 Mamba3 blocks -> pool -> classify (3 classes)
    """

    def __init__(
        self,
        input_dim:  int = 6,
        d_model:    int = 64,
        n_layers:   int = 4,
        n_classes:  int = 3,
        d_state:    int = 32,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.blocks = nn.ModuleList([
            Mamba3Block(d_model, d_state) for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        x: (batch_size, seq_len, input_dim)
        Returns: logits (batch_size, n_classes)
        """
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm_out(x)
        features = x.mean(dim=1)       # global average pool over time
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits

    def predict_proba(self, x: torch.Tensor):
        """Returns probabilities (after softmax) instead of logits."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)