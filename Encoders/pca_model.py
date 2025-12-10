from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
from Encoders.utils import interpolate_primals_coarse_to_fine
import torch


def train_pca_on_primals(X: np.ndarray, var_threshold: float = 0.99):
    """
    Fit PCA to the primal-trajectory matrix X (samples x n_primals).

    - Standardize X (zero mean, unit variance).
    - Compute full PCA to see the variance spectrum.
    - Pick smallest r s.t. cumulative variance >= var_threshold.
    - Fit a reduced PCA with r components.
    """
    if X.shape[0] == 0:
        raise ValueError("X is empty; no samples to train PCA.")

    print(f"[PCA] Fitting PCA on X with shape {X.shape} ...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # full PCA for spectrum
    pca_full = PCA()
    pca_full.fit(X_scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    r = int(np.searchsorted(cumvar, var_threshold) + 1)
    print(f"[PCA] Components needed for {var_threshold * 100:.1f}% variance: r={r}")

    # reduced PCA
    pca = PCA(n_components=r)
    Z = pca.fit_transform(X_scaled)
    recon_scaled = pca.inverse_transform(Z)
    mse = np.mean((X_scaled - recon_scaled) ** 2)
    print(f"[PCA] Reconstruction MSE in scaled space: {mse:.3e}")

    return scaler, pca, Z


class PcaWarmLibrary:
    def __init__(self, feats, Z_latent, pca, scaler, coarse_env,
                 feat_scaler=None, z_scaler=None,
                 regressor=None, regressor_device="cpu"):

        self.feats = np.asarray(feats, dtype=float)
        self.Z_latent = np.asarray(Z_latent, dtype=float)
        self.pca = pca
        self.scaler = scaler
        self.coarse_env = coarse_env

        # feature scaler
        self.feat_scaler = feat_scaler
        if feat_scaler is not None:
            self.feats_scaled = feat_scaler.transform(self.feats)
        else:
            self.feats_scaled = None

        # NEW: z_scaler (targets)
        self.z_scaler = z_scaler

        self.regressor = regressor
        self.regressor_device = torch.device(regressor_device)

    def attach_regressor(self, regressor, device="cpu"):
        self.regressor = regressor
        self.regressor_device = torch.device(device)

    def attach_z_scaler(self, z_scaler):
        self.z_scaler = z_scaler

    def nearest_pca_code(self, feat):
        """
        feat: raw [s_prev, s_curr, Δs]
        """
        feat = np.asarray(feat, dtype=float).reshape(1, -1)

        # scale features if possible
        if self.feat_scaler is not None:
            feat_scaled = self.feat_scaler.transform(feat)
        else:
            feat_scaled = feat

        # ---------- regressor path ----------
        if self.regressor is not None:
            import torch
            with torch.no_grad():
                x = torch.as_tensor(feat_scaled, dtype=torch.float32,
                                    device=self.regressor_device)
                z_scaled_pred = self.regressor(x).cpu().numpy()[0]

            # unscale in Z-space
            if self.z_scaler is not None:
                z_pred = self.z_scaler.inverse_transform(
                    z_scaled_pred.reshape(1, -1)
                )[0]
            else:
                z_pred = z_scaled_pred

            return z_pred, -1

        # ---------- NN fallback ----------
        if self.feats_scaled is not None:
            diffs = self.feats_scaled - feat_scaled
        else:
            diffs = self.feats - feat

        d2 = np.sum(diffs**2, axis=1)
        idx = int(np.argmin(d2))
        z = self.Z_latent[idx].copy()
        return z, idx

    def decode_to_coarse_primals(self, z):

        z = np.asarray(z, dtype=float).reshape(1, -1)
        x_scaled = self.pca.inverse_transform(z)  # (1,D)
        x = self.scaler.inverse_transform(x_scaled)[0]
        return x

    def build_fine_primals(self, fine_env, s_prev, s_curr):

        feat = np.array([s_prev, s_curr, s_curr - s_prev], dtype=float)
        z, idx_nn = self.nearest_pca_code(feat)
        x_coarse = self.decode_to_coarse_primals(z)

        x_fine = interpolate_primals_coarse_to_fine(
            x_coarse_vec=x_coarse,
            coarse_env=self.coarse_env,
            fine_env=fine_env,
        )
        return x_fine

    def save(self, path):
        payload = {
            "feats": self.feats,
            "Z_latent": self.Z_latent,
            "pca": self.pca,
            "scaler": self.scaler,
            "feat_scaler": self.feat_scaler,
            "z_scaler": self.z_scaler,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path, coarse_env, regressor=None, regressor_device="cpu"):
        payload = joblib.load(path)
        return cls(
            feats=payload["feats"],
            Z_latent=payload["Z_latent"],
            pca=payload["pca"],
            scaler=payload["scaler"],
            coarse_env=coarse_env,
            feat_scaler=payload.get("feat_scaler", None),
            z_scaler=payload.get("z_scaler", None),
            regressor=regressor,
            regressor_device=regressor_device,
        )
