"""
Multivariate Mixture Density Network for V2 jet surrogates.

Ported from MDN_V1 with:
- Full Cholesky covariance per component (retained)
- Softplus + epsilon floor + hard ceiling for diagonal (retained)
- Gumbel-max sampling (retained)
- phi-wrapping in mixture_log_prob REMOVED (|Δφ| < 0.2, never triggers)
- Backbone: plain MLP (configurable for robustness check)
- Input dim: 26 (was 13)
- 6 flavour one-hots (was 5)
- mu_bias / sigma_init retuned to new target moments

"""

import math
import torch
import torch.nn as nn

TARGET_DIM = 4


# Mixture log-probability  (NO phi wrapping)
def mixture_log_prob(
    pi: torch.Tensor,
    mu: torch.Tensor,
    L: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Per-event Gaussian-mixture log density.

    Parameters
    ----------
    pi : (B, K) mixture weights (summing to 1)
    mu : (B, K, D) component means
    L : (B, K, D, D) lower-triangular Cholesky factors
    target : (B, D) or (B, M, D)

    Returns
    -------
    logp : (B,) or (B, M) log-densities
    """
    single = target.dim() == mu.dim() - 1  # (B, D) vs (B, M, D)
    if single:
        target = target.unsqueeze(1)  # (B, 1, D)

    D = target.shape[-1]

    # diff: (B, M, K, D)
    diff = target.unsqueeze(2) - mu.unsqueeze(1)

    # Solve L z = diff for z
    z = torch.linalg.solve_triangular(
        L.unsqueeze(1),
        diff.unsqueeze(-1),
        upper=False,
    ).squeeze(
        -1
    )  # (B, M, K, D)

    # Log-determinant from Cholesky diagonal
    log_det = L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1).unsqueeze(1)  # (B, 1, K)

    # Per-component log Gaussian
    log_gauss = (
        -0.5 * z.pow(2).sum(dim=-1) - log_det - 0.5 * D * math.log(2.0 * math.pi)
    )  # (B, M, K)

    # Log-mixture
    log_mix = torch.log(pi.clamp(min=1e-10)).unsqueeze(1) + log_gauss  # (B, M, K)
    logp = torch.logsumexp(log_mix, dim=-1)  # (B, M)

    return logp.squeeze(1) if single else logp


# Calibration penalty on the mixture's normalized second moment
def mixture_calibration_penalty(
    pi: torch.Tensor,
    mu: torch.Tensor,
    L: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Penalty driving the batch mean of the normalized squared residual to D.

    Parameters
    ----------
    pi : (B, K)
    mu : (B, K, D)
    L : (B, K, D, D) lower-triangular Cholesky factors
    target : (B, D)

    Returns
    -------
    scalar penalty  (mean(D_sq) - D)^2
    """
    pi_w = pi.float().unsqueeze(-1)  # (B, K, 1)
    mu_f = mu.float()
    mu_bar = (pi_w * mu_f).sum(dim=1)  # (B, D)

    # diag(L L^T) — per-component marginal variances
    comp_var = L.float().pow(2).sum(dim=-1)  # (B, K, D)

    # Law of total variance
    pvar = (pi_w * (comp_var + (mu_f - mu_bar.unsqueeze(1)).pow(2))).sum(
        dim=1
    )  # (B, D)

    d_sq = ((target.float() - mu_bar).pow(2) / (pvar + 1e-8)).sum(dim=-1)  # (B,)
    D = target.shape[-1]
    return (d_sq.mean() - float(D)).pow(2)


# MDN Head (3-layer MLP per output)
class MDNHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        mid = in_dim // 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid),
            nn.SiLU(),
            nn.Linear(mid, mid),
            nn.SiLU(),
            nn.Linear(mid, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Main MDN model
class Multivariate_MDN(nn.Module):
    """
    Mixture Density Network with full Cholesky covariance.

    Backbone: plain MLP (width × depth, SiLU).
    Three output heads: π (weights), μ (means), L (Cholesky factors).
    """

    def __init__(
        self,
        input_dim: int = 26,
        n_components: int = 32,
        depth: int = 6,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        target_dim: int = TARGET_DIM,
        backbone: str = "plain_mlp",
    ):
        super().__init__()

        if backbone != "plain_mlp":
            raise NotImplementedError(
                f"Backbone '{backbone}' not implemented. "
                "Only 'plain_mlp' is supported. Other architectures can be "
                "added later as an appendix robustness check."
            )

        self.k = n_components
        self.d = target_dim
        self.n_tril = (target_dim * (target_dim + 1)) // 2

        # lain MLP backbone
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        self.backbone = nn.Sequential(*layers)
        self.final_norm = nn.LayerNorm(hidden_dim)

        # DN output heads
        self.pi_head = MDNHead(hidden_dim, n_components)
        self.mu_head = MDNHead(hidden_dim, n_components * target_dim)
        self.tril_head = MDNHead(hidden_dim, n_components * self.n_tril)

        # Precompute tril indices
        row, col = torch.tril_indices(row=target_dim, col=target_dim)
        self.register_buffer("tril_row", row)
        self.register_buffer("tril_col", col)

        diag_pos = torch.tensor(
            [
                (row == i).nonzero(as_tuple=True)[0][col[row == i] == i].item()
                for i in range(target_dim)
            ],
            dtype=torch.long,
        )
        self.register_buffer("diag_pos", diag_pos)

        # Epsilon floor per dimension (prevents variance collapse)
        epsilon_diag = torch.tensor(
            [1e-2, 5e-3, 5e-3, 1e-2],
            dtype=torch.float32,
        )
        self.register_buffer("epsilon_diag", epsilon_diag)

        self._init_heads()

    def _init_heads(self) -> None:
        """Initialize MDN heads to sensible starting values."""
        with torch.no_grad():
            # π: uniform initialization
            nn.init.zeros_(self.pi_head.net[-1].weight)
            nn.init.zeros_(self.pi_head.net[-1].bias)

            init_ranges = [(-0.4, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-2.5, -1.3)]
            mu_bias = torch.stack(
                [torch.linspace(lo, hi, self.k) for lo, hi in init_ranges],
                dim=1,
            ).reshape(-1)
            self.mu_head.net[-1].bias.copy_(mu_bias)

            # L: Cholesky diagonal initialization
            nn.init.normal_(self.tril_head.net[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.tril_head.net[-1].bias)

            # Target initial σ per dimension (slightly wider than true stds)
            sigma_init = [0.20, 0.06, 0.06, 0.20]
            eps_vals = self.epsilon_diag.tolist()

            tril_bias = self.tril_head.net[-1].bias.clone()
            for k in range(self.k):
                for d_i in range(self.d):
                    flat_idx = k * self.n_tril + int(self.diag_pos[d_i].item())
                    # softplus(bias) + epsilon = sigma_init
                    # => bias = softplus^{-1}(sigma_init - epsilon)
                    target = sigma_init[d_i] - eps_vals[d_i]
                    tril_bias[flat_idx] = math.log(math.expm1(max(target, 1e-6)))
            self.tril_head.net[-1].bias.copy_(tril_bias)

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.final_norm(h)

    def _build_L(self, h: torch.Tensor) -> torch.Tensor:
        """Build lower-triangular Cholesky factors from raw network output."""
        B = h.shape[0]
        raw = self.tril_head(h).float().view(B, self.k, self.n_tril)

        processed = raw.clone()

        # Diagonal: softplus + epsilon floor + hard ceiling
        raw_diag = raw[:, :, self.diag_pos]
        L_diag = torch.nn.functional.softplus(raw_diag) + self.epsilon_diag
        L_diag = L_diag.clamp(max=10.0)  # prevent σ → ∞
        processed[:, :, self.diag_pos] = L_diag

        # Build full lower-triangular matrix
        L = torch.zeros(B, self.k, self.d, self.d, device=h.device, dtype=torch.float32)
        L[:, :, self.tril_row, self.tril_col] = processed

        return L.to(h.dtype)

    def forward(self, x: torch.Tensor, pi_temperature: float = 1.0):
        """
        Forward pass.

        Returns
        -------
        pi : (B, K) mixture weights
        mu : (B, K, D) component means
        L : (B, K, D, D) lower-triangular Cholesky factors
        """
        h = self._get_features(x)
        pi_logits = self.pi_head(h)
        pi = torch.softmax(pi_logits / pi_temperature, dim=-1)
        mu = self.mu_head(h).view(-1, self.k, self.d)
        L = self._build_L(h)
        return pi, mu, L

    def neg_log_likelihood(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        L: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood loss (averaged over batch)."""
        return -mixture_log_prob(pi, mu, L, target).mean()

    @torch.no_grad()
    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Sample from the mixture distribution.

        Parameters
        ----------
        x : (B, input_dim) conditioner
        n_samples : number of samples per event

        Returns
        -------
        samples : (B, n_samples, D) or (B, D) if n_samples == 1
        """
        pi, mu, L = self.forward(x)
        B, K, D = mu.shape

        # Gumbel-max trick for component selection
        log_pi = pi.clamp(min=1e-38).log()
        gumbels = (
            -torch.empty(B, n_samples, K, device=x.device, dtype=x.dtype)
            .exponential_()
            .log()
        )
        comp = (log_pi.unsqueeze(1) + gumbels).argmax(dim=-1)  # (B, n_samples)

        # Gather selected component params
        idx_D = comp.unsqueeze(-1).expand(B, n_samples, D)
        mu_sel = (
            mu.unsqueeze(1)
            .expand(B, n_samples, K, D)
            .gather(2, idx_D.unsqueeze(2))
            .squeeze(2)
        )  # (B, n_samples, D)

        idx_DD = comp.unsqueeze(-1).unsqueeze(-1).expand(B, n_samples, D, D)
        L_sel = (
            L.unsqueeze(1)
            .expand(B, n_samples, K, D, D)
            .gather(2, idx_DD.unsqueeze(2))
            .squeeze(2)
        )  # (B, n_samples, D, D)

        # Reparametrisation: y = mu + L @ eps
        eps = torch.randn(B, n_samples, D, 1, device=x.device, dtype=x.dtype)
        samples = mu_sel + (L_sel @ eps).squeeze(-1)

        if n_samples == 1:
            return samples.squeeze(1)
        return samples

    @torch.no_grad()
    def sample_predictions(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        return self.sample(x, n_samples)

    @torch.no_grad()
    def predictive_std(self, x: torch.Tensor) -> torch.Tensor:
        """Predictive standard deviation (law of total variance)."""
        pi, mu, L = self.forward(x)
        pi_w = pi.unsqueeze(-1)  # (B, K, 1)
        mu_bar = (pi_w * mu).sum(dim=1, keepdim=True)  # (B, 1, D)
        comp_var = L.pow(2).sum(dim=-1)  # (B, K, D) — diagonal of L L^T
        var = (pi_w * (comp_var + (mu - mu_bar).pow(2))).sum(dim=1)  # (B, D)
        return var.sqrt()

    @torch.no_grad()
    def mixture_params(self, x: torch.Tensor):
        return self.forward(x)
