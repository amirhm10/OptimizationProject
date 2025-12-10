import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from utils.helpers_net import build_network


class LatentRegressor(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden_dims=(64, 64),
            activation: str = "relu",
            use_layernorm: bool = False,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.net = build_network(
            in_dim=in_dim,
            hidden_dims=list(hidden_dims),
            out_dim=out_dim,
            activation=activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
            prefix="reg",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_latent_regressor(
        feats: np.ndarray,
        Z: np.ndarray,
        hidden_dims=(64, 64),
        batch_size: int = 64,
        num_epochs: int = 200,
        lr: float = 1e-3,
        device: str | None = None,
        standardize_Z: bool = True,
):
    """
    Train MLP to map feats -> Z.
    """
    feats = np.asarray(feats, dtype=np.float32)
    Z = np.asarray(Z, dtype=np.float32)
    assert feats.shape[0] == Z.shape[0]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # --- scale targets Z ---
    if standardize_Z:
        z_scaler = StandardScaler()
        Z_scaled = z_scaler.fit_transform(Z)
    else:
        z_scaler = None
        Z_scaled = Z

    in_dim = feats.shape[1]
    out_dim = Z.shape[1]

    model = LatentRegressor(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dims=hidden_dims,
        activation="relu",
        use_layernorm=False,
        dropout=0.0,
    ).to(device)

    ds = TensorDataset(
        torch.from_numpy(feats),
        torch.from_numpy(Z_scaled.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_n = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)

        if (epoch + 1) % max(1, num_epochs // 10) == 0:
            avg = total_loss / max(total_n, 1)
            print(f"[Regressor] epoch {epoch+1:03d}/{num_epochs:03d} | MSE_z_scaled={avg:.3e}")

    return model, z_scaler