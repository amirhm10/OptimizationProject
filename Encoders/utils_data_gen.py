from PDESystems.pfr_pde import PfrIpoptEnv
import numpy as np
from Encoders.utils import flatten_primals_structured
import torch
from torch.utils.data import Dataset


def generate_pca_dataset(
        env: PfrIpoptEnv,
        num_samples: int = 500,
        sp_low: float = 20.0,
        sp_high: float = 55.0,
        lam_prim: float = 1.0,
        lam_dual: float = 1.0,
        tee: bool = False,
):
    """
    Generate a dataset of primal solutions for different setpoint transitions,
    in the structured layout [cA(:), cB(:), cC(:), Tr(:), F(:), Tj(:)].

    We:
      - Sample a random sequence of setpoints s_k.
      - Do env.reset(s0).
      - For each transition s_{k-1} -> s_k, solve with fixed lam_prim, lam_dual.
      - After each solve, flatten primals with flatten_primals_structured.
      - Also store a simple feature vector f_k = [s_prev, s_curr, s_curr - s_prev].
    """
    setpoints = np.random.uniform(sp_low, sp_high, size=num_samples).astype(float)

    # First solve: cold reset at s0
    s0 = float(setpoints[0])
    env.reset(s0, tee=tee)

    X_list = []
    feat_list = []

    # treat the first solution as a "transition" s0->s0
    X_list.append(flatten_primals_structured(env))
    feat_list.append(np.array([s0, s0, 0.0], dtype=float))

    for k in range(1, num_samples):
        s_prev = float(setpoints[k - 1])
        s_curr = float(setpoints[k])

        obs, info = env.step_no_rl(
            s_new=s_curr,
            lam_prim=lam_prim,
            lam_dual=lam_dual,
            tee=tee,
        )

        if not info["solved"]:
            print(f"[PCA DATA] Skipping k={k}: IPOPT failed at s={s_curr:.2f}")
            continue

        x_k = flatten_primals_structured(env)
        X_list.append(x_k)
        feat_list.append(np.array([s_prev, s_curr, s_curr - s_prev], dtype=float))

        if (k % 200) == 0 or (k == num_samples - 1):
            print(
                f"[PCA DATA] k={k}/{num_samples - 1}, "
                f"s_prev={s_prev:.2f}, s_curr={s_curr:.2f}, "
                f"iters={info['iters']}, time={info['solve_time']:.3f}s"
            )

    X = np.vstack(X_list)
    feats = np.vstack(feat_list)
    print(f"[PCA DATA] Collected {X.shape[0]} samples, each with {X.shape[1]} primals.")
    return feats, X, setpoints


# ---------- Dataset ----------
class PrimalsDataset(Dataset):
    def __init__(self, X_scaled: np.ndarray):
        # X_scaled: (N_samples, D)
        self.X = torch.from_numpy(X_scaled.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]