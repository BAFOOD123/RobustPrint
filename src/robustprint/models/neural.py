"""Neural regressors with Gaussian likelihood heads: deep ensemble and MC dropout."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from robustprint.models.base import Prediction
from robustprint.physics.composition import FloatArray

MIN_LOG_VAR = -9.0
MAX_LOG_VAR = 3.0


class GaussianMLP(nn.Module):
    """MLP that outputs a mean and a log-variance."""

    def __init__(self, n_in: int, hidden: int = 64, depth: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_in
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and clamped log-variance."""
        out = self.head(self.body(x))
        return out[:, 0], out[:, 1].clamp(MIN_LOG_VAR, MAX_LOG_VAR)


def _train(net: GaussianMLP, X: torch.Tensor, y: torch.Tensor, epochs: int, lr: float, weight_decay: float,
           seed: int, val_fraction: float = 0.2, check_every: int = 10) -> GaussianMLP:
    """Full-batch Adam on the Gaussian NLL with early stopping on a random validation split.

    Without early stopping, NLL-trained networks on ~100 points shrink their predicted
    variance on the training rows and become badly overconfident (observed in our first
    benchmark; docs/DECISIONS.md ADR-015). The best validation-NLL state is restored.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    perm = torch.randperm(n, generator=gen)
    n_val = max(int(round(val_fraction * n)), 2) if n >= 10 else 0
    val, tr = perm[:n_val], perm[n_val:]
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.GaussianNLLLoss(full=False, eps=1e-6)
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    best = float("inf")
    for epoch in range(epochs):
        net.train()
        opt.zero_grad()
        mean, log_var = net(X[tr])
        loss = loss_fn(mean, y[tr], log_var.exp())
        loss.backward()
        opt.step()
        if n_val and (epoch + 1) % check_every == 0:
            net.eval()
            with torch.no_grad():
                vm, vlv = net(X[val])
                vloss = float(loss_fn(vm, y[val], vlv.exp()))
            if vloss < best:
                best = vloss
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
    if n_val:
        net.load_state_dict(best_state)
    return net


class _TorchRegressor:
    """Shared scaling logic for the neural models."""

    def _prepare(self, X: FloatArray, y: FloatArray) -> tuple[torch.Tensor, torch.Tensor]:
        self.x_scaler_ = StandardScaler().fit(X)
        self.y_mean_ = float(np.mean(y))
        self.y_std_ = float(np.std(y) or 1.0)
        Xt = torch.as_tensor(self.x_scaler_.transform(X), dtype=torch.float32)
        yt = torch.as_tensor((y - self.y_mean_) / self.y_std_, dtype=torch.float32)
        return Xt, yt

    def _inputs(self, X: FloatArray) -> torch.Tensor:
        return torch.as_tensor(self.x_scaler_.transform(X), dtype=torch.float32)


class DeepEnsemble(_TorchRegressor):
    """Ensemble of Gaussian MLPs (Lakshminarayanan et al., 2017), without adversarial training.

    Each member is trained from a different random initialisation. The predictive mixture is
    summarised by its mean and variance: aleatoric = mean of member variances, epistemic =
    variance of member means.
    """

    name = "deep_ensemble"

    def __init__(self, n_members: int = 5, hidden: int = 64, epochs: int = 400, lr: float = 5e-3,
                 weight_decay: float = 1e-3, random_state: int = 0) -> None:
        self.n_members = n_members
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state

    def fit(self, X: FloatArray, y: FloatArray, n_replicates: np.ndarray | None = None) -> DeepEnsemble:
        """Train all members."""
        Xt, yt = self._prepare(X, y)
        self.members_: list[GaussianMLP] = []
        for m in range(self.n_members):
            seed = self.random_state * 1000 + m
            torch.manual_seed(seed)
            net = GaussianMLP(X.shape[1], self.hidden)
            self.members_.append(_train(net, Xt, yt, self.epochs, self.lr, self.weight_decay, seed))
        return self

    def member_predictions(self, X: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Per-member means and variances, shape (members, n), on the original scale."""
        Xt = self._inputs(X)
        means, variances = [], []
        with torch.no_grad():
            for net in self.members_:
                net.eval()
                mu, log_var = net(Xt)
                means.append(mu.numpy() * self.y_std_ + self.y_mean_)
                variances.append(log_var.exp().numpy() * self.y_std_**2)
        return np.asarray(means, dtype=float), np.asarray(variances, dtype=float)

    def predict(self, X: FloatArray) -> Prediction:
        """Mixture mean, epistemic and aleatoric variance."""
        means, variances = self.member_predictions(X)
        return Prediction(means.mean(0), means.var(0), variances.mean(0))


class MCDropout(_TorchRegressor):
    """Gaussian MLP with dropout kept active at prediction time (Gal & Ghahramani, 2016)."""

    name = "mc_dropout"

    def __init__(self, hidden: int = 64, dropout: float = 0.1, epochs: int = 400, lr: float = 5e-3,
                 weight_decay: float = 1e-3, n_passes: int = 50, random_state: int = 0) -> None:
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_passes = n_passes
        self.random_state = random_state

    def fit(self, X: FloatArray, y: FloatArray, n_replicates: np.ndarray | None = None) -> MCDropout:
        """Train one network with dropout."""
        Xt, yt = self._prepare(X, y)
        torch.manual_seed(self.random_state)
        self.net_ = _train(GaussianMLP(X.shape[1], self.hidden, dropout=self.dropout), Xt, yt, self.epochs,
                           self.lr, self.weight_decay, self.random_state)
        return self

    def predict(self, X: FloatArray) -> Prediction:
        """Average ``n_passes`` stochastic forward passes (seeded for reproducibility)."""
        Xt = self._inputs(X)
        torch.manual_seed(self.random_state + 17)
        self.net_.train()
        means, variances = [], []
        with torch.no_grad():
            for _ in range(self.n_passes):
                mu, log_var = self.net_(Xt)
                means.append(mu.numpy())
                variances.append(log_var.exp().numpy())
        m = np.asarray(means) * self.y_std_ + self.y_mean_
        v = np.asarray(variances) * self.y_std_**2
        return Prediction(m.mean(0), m.var(0), v.mean(0))
