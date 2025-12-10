import numpy as np
from PDESystems.pfr_pde import PfrIpoptEnv


def _regularization_term(action, prev_action, cfg):
    """Small penalty on action magnitude and jitter."""
    if action is None:
        return 0.0
    a = np.asarray(action, dtype=float).ravel()
    norm2 = float(np.mean(a**2))

    if prev_action is None:
        dnorm2 = 0.0
    else:
        a_prev = np.asarray(prev_action, dtype=float).ravel()
        if a_prev.shape != a.shape:
            dnorm2 = 0.0
        else:
            dnorm2 = float(np.mean((a - a_prev)**2))

    return -(cfg["w_act"] * norm2 + cfg["w_act_delta"] * dnorm2)


def compute_reward_rl_vs_baseline(
    info_rl,
    info_base,
    env_rl: "PfrIpoptEnv",
    env_base: "PfrIpoptEnv",
    action,
    cfg,
    prev_action=None,
):
    """
    Compare RL vs baseline on the same setpoint and build a shaped reward.

    Cases:
      - baseline solves, RL fails -> big negative
      - baseline fails, RL solves -> big positive (rescue)
      - both fail -> modest negative
      - both solve -> use efficiency + quality + safety + KKT + reg
    """
    solved_rl = bool(info_rl.get("solved", False))
    solved_base = bool(info_base.get("solved", False))

    # ---------- feasibility cases ----------
    if solved_base and (not solved_rl):
        # RL broke a case the baseline could solve
        r_feas = -cfg["R_fail_vs_base"]
        r_reg = _regularization_term(action, prev_action, cfg)
        return float(r_feas + r_reg)

    if (not solved_base) and solved_rl:
        # RL rescued a failing baseline
        I_rl = float(info_rl.get("iters", env_rl.max_iter))
        r_feas = cfg["R_rescue"] - 0.1 * I_rl  # small penalty for taking many iters
        r_reg = _regularization_term(action, prev_action, cfg)
        return float(r_feas + r_reg)

    if (not solved_base) and (not solved_rl):
        # everyone fails, small negative and move on
        r_feas = -cfg["R_both_fail"]
        r_reg = _regularization_term(action, prev_action, cfg)
        return float(r_feas + r_reg)

    # If we're here: BOTH solved. We can use the full shaped reward.
    eps = 1e-8

    # ---------- efficiency: iterations + time (relative gains) ----------
    I_rl = float(info_rl["iters"])
    I_base = float(info_base["iters"])
    T_rl = float(info_rl["solve_time"])
    T_base = float(info_base["solve_time"])

    g_I = (I_base - I_rl) / max(I_base, eps)   # >0 if RL uses fewer iters
    g_T = (T_base - T_rl) / max(T_base, eps)   # >0 if RL is faster

    R_eff = cfg["w_iter"] * g_I + cfg["w_time"] * g_T

    # ---------- solution quality: tracking / objective (l2) ----------
    L_rl = float(info_rl["l2"])
    L_base = float(info_base["l2"])

    rel_L = (L_rl - L_base) / (abs(L_base) + eps)  # >0 if RL is worse
    R_qual = -cfg["w_L"] * max(0.0, rel_L)         # only penalize if worse than baseline

    # ---------- safety: distance to T_hi ----------
    T_hi = float(env_rl.params["T_hi"])  # same for both envs

    Tr_rl = float(info_rl["raw_results"]["Tr"].max())    # max reactor T RL
    Tr_base = float(info_base["raw_results"]["Tr"].max())  # max reactor T baseline

    margin_rl = T_hi - Tr_rl
    margin_base = T_hi - Tr_base

    # If RL eats more of the safety margin than baseline, penalize that difference.
    # Normalize by 50 K just to keep numbers sane.
    margin_diff = margin_base - margin_rl   # >0 if RL is closer to limit
    R_safe = -cfg["w_Tsafe"] * max(0.0, margin_diff / 50.0)

    # ---------- KKT-ish: active bounds and multipliers ----------
    # RL env
    z_mag_rl = np.maximum(np.abs(env_rl.warm_zL), np.abs(env_rl.warm_zU))
    active_frac_rl = float(np.mean(z_mag_rl > 1e-4))
    dual_mean_rl = float(np.mean(np.abs(env_rl.warm_duals))) if env_rl.warm_duals.size > 0 else 0.0

    # baseline env
    z_mag_base = np.maximum(np.abs(env_base.warm_zL), np.abs(env_base.warm_zU))
    active_frac_base = float(np.mean(z_mag_base > 1e-4))
    dual_mean_base = float(np.mean(np.abs(env_base.warm_duals))) if env_base.warm_duals.size > 0 else 0.0

    d_active = active_frac_rl - active_frac_base
    d_dual = dual_mean_rl - dual_mean_base

    # squash dual difference a bit
    d_dual_n = float(np.tanh(d_dual))

    R_kkt = -cfg["w_kkt"] * max(0.0, d_active) - cfg["w_kkt"] * max(0.0, d_dual_n)

    # ---------- action regularization ----------
    R_reg = _regularization_term(action, prev_action, cfg)

    R_total = R_eff + R_qual + R_safe + R_kkt + R_reg
    return float(R_total)
