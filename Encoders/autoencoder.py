import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Encoders.utils_data_gen import PrimalsDataset

from sklearn.preprocessing import StandardScaler
import joblib
from Encoders.utils import interpolate_primals_coarse_to_fine


# ---------- Autoencoder ----------
class PrimalsAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def train_autoencoder_on_primals(
        X: np.ndarray,
        latent_dim: int = 32,
        batch_size: int = 128,
        num_epochs: int = 50,
        lr: float = 1e-3,
        device: str = "cpu",
):
    """
    Train an autoencoder on primal trajectories X (N_samples x D).

    Returns:
      scaler      : StandardScaler fitted on X
      ae_model    : trained PrimalsAutoencoder (on CPU)
      X_scaled    : scaled X
      X_recon     : reconstructed X (unscaled) from the AE
    """
    X = np.asarray(X, dtype=float)
    N, D = X.shape
    print(f"[AE] Training autoencoder on X with shape {X.shape}, latent_dim={latent_dim}")

    # scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dataset = PrimalsDataset(X_scaled)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = PrimalsAutoencoder(input_dim=D, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat = model(batch)
            loss = criterion(x_hat, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= N
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[AE] Epoch {epoch + 1:3d}/{num_epochs}, MSE={epoch_loss:.4e}")

    # reconstruct all samples once (for fast lookup later)
    model.eval()
    with torch.no_grad():
        X_scaled_t = torch.from_numpy(X_scaled.astype(np.float32)).to(device)
        X_rec_scaled = model(X_scaled_t).cpu().numpy()

    X_recon = scaler.inverse_transform(X_rec_scaled)

    # move model to CPU for saving; we won't use it in the hot loop
    model_cpu = model.to("cpu")
    return scaler, model_cpu, X_scaled, X_recon


class AeWarmLibrary:

    def __init__(
            self,
            feats,
            X_recon,
            scaler,
            coarse_env,
            Z_latent=None,
            feat_scaler=None,
            z_scaler=None,
            regressor=None,
            regressor_device="cpu",
            decoder=None,
    ):
        """
        feats        : (N,3)  [s_prev, s_curr, Δs]
        X_recon      : (N,D)  reconstructed primals (physical units)
        scaler       : StandardScaler for primals (X)
        coarse_env   : coarse PfrIpoptEnv
        Z_latent     : (N,r)  AE latent codes (optional, useful for reload)
        feat_scaler  : StandardScaler for feats (optional)
        z_scaler     : StandardScaler for Z (optional)
        regressor    : NN mapping scaled feats -> scaled Z (optional)
        decoder      : AE decoder, mapping z -> x_scaled (optional)
        """
        self.feats = np.asarray(feats, dtype=float)
        self.X_recon = np.asarray(X_recon, dtype=float)
        self.scaler = scaler
        self.coarse_env = coarse_env

        if self.feats.shape[0] != self.X_recon.shape[0]:
            raise ValueError("AeWarmLibrary: feats and X_recon must have same #samples.")

        # latent codes (for reload / debugging)
        self.Z_latent = None if Z_latent is None else np.asarray(Z_latent, dtype=float)

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

    # ---- attachment helpers (for reload) ----
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

    # ---- nearest neighbour in feature space (raw or scaled) ----
    def _nearest_idx(self, feat, feat_scaled=None):
        if self.feats_scaled is not None and feat_scaled is not None:
            diffs = self.feats_scaled - feat_scaled
        else:
            diffs = self.feats - feat
        d2 = np.sum(diffs ** 2, axis=1)
        idx = int(np.argmin(d2))
        return idx

    def _decode_from_z(self, z):
        if self.decoder is None:
            raise RuntimeError("AeWarmLibrary: decoder is None but regressor path was requested.")

        # ensure numpy float32 with shape (1, latent_dim)
        z = np.asarray(z, dtype=np.float32).reshape(1, -1)

        # figure out which device the decoder lives on
        try:
            dec_device = next(self.decoder.parameters()).device
        except StopIteration:
            dec_device = torch.device("cpu")

        with torch.no_grad():
            z_t = torch.from_numpy(z).to(dec_device)   # <-- move to decoder device
            x_scaled_t = self.decoder(z_t)             # (1, D_scaled)
            x_scaled = x_scaled_t.detach().cpu().numpy()

        x = self.scaler.inverse_transform(x_scaled)[0]  # (D,)
        return x

    def build_fine_primals(self, fine_env, s_prev, s_curr):

        feat = np.array([s_prev, s_curr, s_curr - s_prev], dtype=float).reshape(1, -1)
        # scale feature if we have a scaler
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
            # ----- feats_scaled -> z_scaled -> z -> x_coarse via decoder -----
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
            # ----- fallback: NN in feature space, then use stored X_recon -----
            idx = self._nearest_idx(feat, feat_scaled)
            x_coarse = self.X_recon[idx, :]

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
            "scaler": self.scaler,
            "Z_latent": self.Z_latent,
            "feat_scaler": self.feat_scaler,
            "z_scaler": self.z_scaler,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str, coarse_env,
             regressor=None, regressor_device="cpu", decoder=None):
        payload = joblib.load(path)
        return cls(
            feats=payload["feats"],
            X_recon=payload["X_recon"],
            scaler=payload["scaler"],
            coarse_env=coarse_env,
            Z_latent=payload.get("Z_latent", None),
            feat_scaler=payload.get("feat_scaler", None),
            z_scaler=payload.get("z_scaler", None),
            regressor=regressor,
            regressor_device=regressor_device,
            decoder=decoder,
        )
