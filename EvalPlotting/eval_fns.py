import numpy as np


def eval_policy_on_setpoints_full(
        env,
        agent,
        setpoints,
        mode="rl",  # "rl", "cold", or "warm"
        tee=False,
):
    setpoints = np.asarray(setpoints, dtype=float)
    assert len(setpoints) >= 2

    # initial solve at first setpoint
    s0 = float(setpoints[0])
    obs0 = env.reset(s0, tee=tee)

    # RL needs a state; baselines ignore it
    if mode == "rl":
        state = obs0["state_vec"]

    it_list = []
    time_list = []
    lam_list = []
    mu_list = []
    tol_list = []
    acc_list = []
    act_list = []
    sp_list = []
    l2_list = []
    w_cold_list = []
    w_warm_list = []
    w_pca_list = []

    solved_list = []  # bool: did IPOPT declare success?
    Tr_max_list = []  # max reactor temperature per solve (if available)

    for k in range(1, len(setpoints)):
        s_new = float(setpoints[k])

        # -------- choose how to step the environment --------
        if mode == "rl":
            # deterministic evaluation: no exploration noise
            try:
                action = agent.act_eval(state)  # shape (5,)
            except AttributeError:
                action = agent.take_action(state, explore=False)

            obs_next, info = env.step(s_new=s_new, action=action, tee=tee)
            state = obs_next["state_vec"]

        elif mode == "cold":
            # full cold start: lam_prim = 0, lam_dual = 0
            action = np.array([-1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            obs_next, info = env.step_no_rl(
                s_new=s_new,
                lam_prim=0.0,
                lam_dual=0.0,
                tee=tee,
            )

        elif mode == "warm":
            # full warm start: lam_prim = 1, lam_dual = 1
            action = np.array([+1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            obs_next, info = env.step_no_rl(
                s_new=s_new,
                lam_prim=1.0,
                lam_dual=0.0,
                tee=tee,
            )

        else:
            raise ValueError(f"Unknown mode={mode}")

        # -------- log stats --------
        it_list.append(info["iters"])
        time_list.append(info["solve_time"])
        lam_list.append(info["lambda_prim"])  # = w_warm + w_pca in RL mode
        mu_list.append(info.get("mu_init", np.nan))
        tol_list.append(info.get("tol", np.nan))
        acc_list.append(info.get("acceptable_tol", np.nan))
        act_list.append(np.array(action, dtype=float))
        sp_list.append(s_new)
        l2_list.append(info["l2"])

        w_cold_list.append(info.get("w_cold", np.nan))
        w_warm_list.append(info.get("w_warm", np.nan))
        w_pca_list.append(info.get("w_pca", np.nan))

        # -------- NEW: fields for the global metric --------
        solved_list.append(bool(info.get("solved", True)))

        raw = info.get("raw_results", None)
        if raw is not None and ("Tr" in raw):
            Tr_max = float(np.max(raw["Tr"]))
        else:
            Tr_max = np.nan
        Tr_max_list.append(Tr_max)

    return {
        "iters": np.array(it_list),
        "solve_time": np.array(time_list),
        "lam_prim": np.array(lam_list),
        "mu_init": np.array(mu_list),
        "tol": np.array(tol_list),
        "acc_tol": np.array(acc_list),
        "actions": np.vstack(act_list),
        "setpoints": np.array(sp_list),
        "l2": np.array(l2_list),
        "w_cold": np.array(w_cold_list),
        "w_warm": np.array(w_warm_list),
        "w_pca": np.array(w_pca_list),

        # NEW: for metric
        "solved": np.array(solved_list, dtype=bool),
        "Tr_max": np.array(Tr_max_list, dtype=float),
    }

