import numpy as np
from PDESystems.pfr_pde import PfrIpoptEnv, map_log_range
from rl_helper.reward_fn import compute_reward_rl_vs_baseline


def decode_action_debug(a_vec, env: "PfrIpoptEnv"):
    """
    a_vec: [a_warm, a_pca, a_mu, a_tol, a_atol] in [-1,1]
    env : PfrIpoptEnv (for mu/tol ranges)

    Returns a dict with:
      - w_cold, w_warm, w_pca  (convex weights, matching env.step)
      - mu_init, tol, acc_tol  (actual IPOPT settings)
    """
    a = np.clip(np.asarray(a_vec, dtype=float).ravel(), -1.0, 1.0)
    if a.size < 5:
        raise ValueError("Expected action of length >=5: [a_warm, a_pca, a_mu, a_tol, a_atol].")

    a_warm, a_pca, a_mu, a_tol, a_atol = a[:5]

    if env.pca_lib is None:
        # same 2-way mixing as env.step
        w_warm = 0.5 * (a_warm + 1.0)
        w_warm = float(np.clip(w_warm, 0.0, 1.0))
        w_cold = 1.0 - w_warm
        w_pca = 0.0
    else:
        # same 3-way softmax as env.step
        logits = np.array([0.0, a_warm, a_pca], dtype=float)
        logits -= logits.max()
        w = np.exp(logits)
        w /= w.sum()
        w_cold, w_warm, w_pca = w

    mu_init = map_log_range(a_mu, env.mu_min, env.mu_max)
    tol = map_log_range(a_tol, env.tol_min, env.tol_max)
    acc_tol = map_log_range(a_atol, env.acc_tol_min, env.acc_tol_max)

    return {
        "w_cold": float(w_cold),
        "w_warm": float(w_warm),
        "w_pca": float(w_pca),
        "mu_init": float(mu_init),
        "tol": float(tol),
        "acc_tol": float(acc_tol),
    }


def run_td3_with_baseline(
    env_rl: PfrIpoptEnv,
    env_base: PfrIpoptEnv,
    agent,
    setpoints,
    cfg,
    warm_start_steps=250,
    tee=False,
):
    """
    Train TD3 oe SAC on env_rl, while env_base runs a fixed baseline strategy.
    Reward at each step compares RL vs baseline on that setpoint.

    - env_rl.step(...) uses RL actions [a_warm, a_pca, a_mu, a_tol, a_atol]
    - env_base.step_no_rl(...) uses a fixed lam_prim/lam_dual and default IPOPT options.
    """
    setpoints = np.asarray(setpoints, dtype=float)
    num_sp = len(setpoints)
    assert num_sp >= 2, "Need at least 2 setpoints."

    total_steps = 0

    rewards_hist = []
    it_rl_hist = []
    it_base_hist = []
    time_rl_hist = []
    time_base_hist = []

    # ---- initial solve at first setpoint on both envs ----
    s0 = float(setpoints[0])
    obs_rl = env_rl.reset(s0, tee=tee)
    _ = env_base.reset(s0, tee=False)   # baseline just for reference

    state = obs_rl["state_vec"]
    prev_action = None

    print(f"Initial setpoint s0={s0:.2f} solved; starting RL stream with baseline...")

    for k in range(1, num_sp):
        s_new = float(setpoints[k])

        # ---------- 1) Baseline solve on env_base ----------
        obs_base, info_base = env_base.step_no_rl(
            s_new=s_new,
            lam_prim=1.0,  # full warm primals
            lam_dual=0.0,
            tee=tee,
        )

        # ---------- 2) RL solve on env_rl ----------
        action = agent.take_action(state, explore=True)
        obs_rl_next, info_rl = env_rl.step(s_new=s_new, action=action, tee=tee)

        # ---------- 3) Reward from RL vs baseline ----------
        r = compute_reward_rl_vs_baseline(
            info_rl=info_rl,
            info_base=info_base,
            env_rl=env_rl,
            env_base=env_base,
            action=action,
            cfg=cfg,
            prev_action=prev_action
        )

        next_state = obs_rl_next["state_vec"]
        done = 0.0  # no terminal state in this streaming setup

        # ---------- 4) Store transition ----------
        agent.push(
            state,
            action.astype(np.float32),
            float(r),
            next_state,
            float(done),
        )

        # ---------- 5) Train TD3 after warm_start_steps ----------
        if total_steps >= warm_start_steps:
            _ = agent.train_step()

        total_steps += 1
        rewards_hist.append(r)
        it_rl_hist.append(info_rl["iters"])
        it_base_hist.append(info_base["iters"])
        time_rl_hist.append(info_rl["solve_time"])
        time_base_hist.append(info_base["solve_time"])

        if (k % 100) == 0 or (k == num_sp - 1):
            try:
                decoded = decode_action_debug(action, env_rl)
                extra_str = (
                    f" | w_warm={decoded['w_warm']:.2f}, w_pca={decoded['w_pca']:.2f}, "
                    f"mu={decoded['mu_init']:.2e}, tol={decoded['tol']:.1e}"
                )
            except Exception:
                extra_str = ""

            print(
                f"[step {k}/{num_sp-1}] s={s_new:.2f}, "
                f"r={r:.1f}, "
                f"iters_rl={info_rl['iters']}, iters_base={info_base['iters']}, "
                f"time_rl={info_rl['solve_time']:.3f}s, time_base={info_base['solve_time']:.3f}s"
                + extra_str
            )

        state = next_state
        prev_action = action

    print(f"\nStreaming training with baseline finished. Total RL steps = {total_steps}")
    return {
        "rewards": np.array(rewards_hist),
        "iters_rl": np.array(it_rl_hist),
        "iters_base": np.array(it_base_hist),
        "time_rl": np.array(time_rl_hist),
        "time_base": np.array(time_base_hist),
    }
