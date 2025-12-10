import joblib

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from Encoders.utils_data_gen import PrimalsDataset

from sklearn.preprocessing import StandardScaler
import numpy as np

from Encoders.utils import interpolate_primals_coarse_to_fine


class PrimalsVAE(nn.Module):
    """
    Simple VAE for flattened primal trajectories x in R^D.

    Encoder:  x -> h -> (mu, logvar)
    Decoder:  z -> x_hat
    """

    def __init__(
            self,
            in_dim: int,
            latent_dim: int = 64,
            hidden_dims=(512, 256, 128),
    ):
        super().__init__()
        # ----- encoder -----
        enc_layers = []
        last = in_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(last, h))
            enc_layers.append(nn.ReLU())
            last = h
        self.encoder = nn.Sequential(*enc_layers)

        self.fc_mu = nn.Linear(last, latent_dim)
        self.fc_logvar = nn.Linear(last, latent_dim)

        # ----- decoder -----
        dec_layers = []
        last = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(last, h))
            dec_layers.append(nn.ReLU())
            last = h
        dec_layers.append(nn.Linear(last, in_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        # z = mu + std * eps
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def train_vae_on_primals(
        X: np.ndarray,
        latent_dim: int = 64,
        beta: float = 1.0,  # weight on KL term (beta-VAE, normal one now)
        batch_size: int = 128,
        num_epochs: int = 50,
        lr: float = 1e-3,
        device: str = None,
):
    """
    Train a VAE on primal trajectories X (num_samples x n_primals).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # ---------- scale inputs ----------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.astype(np.float32))
    n_samples, in_dim = X_scaled.shape

    X_tensor = torch.from_numpy(X_scaled).float()
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # ---------- build model ----------
    vae = PrimalsVAE(in_dim=in_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # ---------- training loop ----------
    for epoch in range(num_epochs):
        vae.train()
        total_rec = 0.0
        total_kl = 0.0
        total_n = 0

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)

            x_hat, mu, logvar = vae(batch_x)

            # reconstruction loss (MSE)
            rec_loss = F.mse_loss(x_hat, batch_x, reduction="mean")

            # KL(q(z|x) || N(0,I)) averaged over batch
            # -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            kl_loss = -0.5 * torch.mean(
                1.0 + logvar - mu.pow(2) - logvar.exp()
            )

            loss = rec_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = batch_x.size(0)
            total_rec += rec_loss.item() * bs
            total_kl += kl_loss.item() * bs
            total_n += bs

        if (epoch + 1) % max(1, num_epochs // 10) == 0:
            avg_rec = total_rec / total_n
            avg_kl = total_kl / total_n
            print(
                f"[VAE] epoch {epoch + 1:03d}/{num_epochs:03d} | "
                f"rec={avg_rec:.4e}, kl={avg_kl:.4e}, beta={beta:.2f}"
            )

    # ---------- full recon + latent codes ----------
    vae.eval()
    with torch.no_grad():
        X_full = X_tensor.to(device)
        mu_full, logvar_full = vae.encode(X_full)
        z_full = mu_full  # deterministic latent = mean
        X_recon_scaled = vae.decode(z_full).cpu().numpy()
        Z_mu = mu_full.cpu().numpy()

    # back to original physical units
    X_recon = scaler.inverse_transform(X_recon_scaled)

    return scaler, vae, X_scaled, X_recon, Z_mu


class VaeWarmLibrary:
    """
    Warm-start library based on a VAE.
    Can use:
      - nearest neighbour in feature space (feats -> X_recon[idx]),
      - OR regressor + decoder: feats -> z -> x_coarse.
    """

    def __init__(
            self,
            feats,
            X_recon,
            Z_mu,
            scaler,
            coarse_env,
            feat_scaler=None,
            z_scaler=None,
            regressor=None,
            regressor_device="cpu",
            decoder=None,
    ):
        self.feats = np.asarray(feats, dtype=float)
        self.X_recon = np.asarray(X_recon, dtype=float)
        self.Z_mu = np.asarray(Z_mu, dtype=float)
        self.scaler = scaler
        self.coarse_env = coarse_env

        assert self.feats.shape[0] == self.X_recon.shape[0], \
            "feats and X_recon must have same number of samples."
        assert self.Z_mu.shape[0] == self.feats.shape[0], \
            "Z_mu must have same number of samples as feats."

        # feature scaler
        self.feat_scaler = feat_scaler
        if feat_scaler is not None:
            self.feats_scaled = feat_scaler.transform(self.feats)
        else:
            self.feats_scaled = None

        # latent scaler + regressor + decoder
        self.z_scaler = z_scaler
        self.regressor = regressor
        self.regressor_device = torch.device(regressor_device)
        self.decoder = decoder  # expects z -> x_scaled

    # attach helpers
    def attach_feat_scaler(self, feat_scaler):
        self.feat_scaler = feat_scaler
        if feat_scaler is not None:
            self.feats_scaled = feat_scaler.transform(self.feats)
        else:
            self.feats_scaled = None

    def attach_z_scaler(self, z_scaler):
        self.z_scaler = z_scaler

    def attach_regressor(self, regressor, device="cpu"):
        self.regressor = regressor
        self.regressor_device = torch.device(device)

    def attach_decoder(self, decoder):
        self.decoder = decoder

    # NN helper
    def _nearest_index(self, feat, feat_scaled=None):
        if self.feats_scaled is not None and feat_scaled is not None:
            diffs = self.feats_scaled - feat_scaled
        else:
            diffs = self.feats - feat
        dists = np.linalg.norm(diffs, axis=1)
        idx = int(np.argmin(dists))
        return idx

    # decode latent z -> x (physical)
    def _decode_from_z(self, z):
        if self.decoder is None:
            raise RuntimeError("VaeWarmLibrary: decoder is None but regressor path was requested.")

        z = np.asarray(z, dtype=np.float32).reshape(1, -1)

        try:
            dec_device = next(self.decoder.parameters()).device
        except StopIteration:
            dec_device = torch.device("cpu")

        with torch.no_grad():
            z_t = torch.from_numpy(z).to(dec_device)
            x_scaled_t = self.decoder(z_t)
            x_scaled = x_scaled_t.detach().cpu().numpy()

        x = self.scaler.inverse_transform(x_scaled)[0]
        return x

    def build_fine_primals(self, fine_env, s_prev, s_curr):
        feat = np.array([s_prev, s_curr, s_curr - s_prev], dtype=float).reshape(1, -1)
        if self.feat_scaler is not None:
            feat_scaled = self.feat_scaler.transform(feat)
        else:
            feat_scaled = feat

        use_regression = (
            (self.regressor is not None) and
            (self.z_scaler is not None) and
            (self.decoder is not None)
        )

        if use_regression:
            # feats_scaled -> z_scaled -> z -> x via decoder
            with torch.no_grad():
                x_in = torch.as_tensor(
                    feat_scaled, dtype=torch.float32, device=self.regressor_device
                )
                z_scaled_pred = self.regressor(x_in).cpu().numpy()[0]
            z_pred = self.z_scaler.inverse_transform(
                z_scaled_pred.reshape(1, -1)
            )[0]
            x_coarse = self._decode_from_z(z_pred)
        else:
            idx = self._nearest_index(feat, feat_scaled)
            x_coarse = self.X_recon[idx]

        x_fine = interpolate_primals_coarse_to_fine(
            x_coarse_vec=x_coarse,
            coarse_env=self.coarse_env,
            fine_env=fine_env,
        )
        return x_fine

    def save(self, path: str):
        payload = {
            "feats": self.feats,
            "X_recon": self.X_recon,
            "Z_mu": self.Z_mu,
            "scaler": self.scaler,
            "feat_scaler": self.feat_scaler,
            "z_scaler": self.z_scaler,
        }
        joblib.dump(payload, path)
        print(f"[VAE LIB] Saved to {path}")

    @classmethod
    def load(cls, path: str, coarse_env,
             regressor=None, regressor_device="cpu", decoder=None):
        payload = joblib.load(path)
        feats = payload["feats"]
        X_recon = payload["X_recon"]
        Z_mu = payload["Z_mu"]
        scaler = payload["scaler"]
        feat_scaler = payload.get("feat_scaler", None)
        z_scaler = payload.get("z_scaler", None)
        print(f"[VAE LIB] Loaded from {path}, feats shape={feats.shape}, X_recon shape={X_recon.shape}")
        return cls(
            feats=feats,
            X_recon=X_recon,
            Z_mu=Z_mu,
            scaler=scaler,
            coarse_env=coarse_env,
            feat_scaler=feat_scaler,
            z_scaler=z_scaler,
            regressor=regressor,
            regressor_device=regressor_device,
            decoder=decoder,
        )



