import numpy as np
import time
import matplotlib.pyplot as plt


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
        if agent is None:
            raise ValueError("agent cannot be None when mode='rl'")
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
            action = np.array([-1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            obs_next, info = env.step_no_rl(
                s_new=s_new,
                lam_prim=0.0,
                lam_dual=0.0,
                tee=tee,
            )

        elif mode == "warm":
            action = np.array([1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            obs_next, info = env.step_no_rl(
                s_new=s_new,
                lam_prim=1.0,
                lam_dual=1.0,
                tee=tee,
            )

        else:
            raise ValueError(f"Unknown mode={mode}")

        # -------- log stats --------
        it_list.append(info["iters"])
        time_list.append(info["solve_time"])
        lam_list.append(info["lambda_prim"])
        mu_list.append(info.get("mu_init", np.nan))
        tol_list.append(info.get("tol", np.nan))
        acc_list.append(info.get("acceptable_tol", np.nan))
        act_list.append(np.array(action, dtype=float))
        sp_list.append(s_new)
        l2_list.append(info["l2"])

        w_cold_list.append(info.get("w_cold", np.nan))
        w_warm_list.append(info.get("w_warm", np.nan))
        w_pca_list.append(info.get("w_pca", np.nan))

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
    }


def eval_policy_on_setpoints_meta(
    env,
    agent,
    setpoints,
    mode="rl",
    tee=False,
):
    """
    Run eval_policy_on_setpoints_full and pack results into a meta_list
    similar to run_setpoint_sequence.

    Returns
    -------
    eval_out : dict
        Raw arrays from eval_policy_on_setpoints_full.
    meta_list : list of dict
        One entry per setpoint change (i.e., for setpoints[1:]).
        Each dict contains setpoint, iters, solve_time, l2, and action weights.
    """
    setpoints = np.asarray(setpoints, dtype=float)
    assert len(setpoints) >= 2

    eval_out = eval_policy_on_setpoints_full(
        env=env,
        agent=agent,
        setpoints=setpoints,
        mode=mode,
        tee=tee,
    )

    iters = eval_out["iters"]
    solve_time = eval_out["solve_time"]
    l2 = eval_out["l2"]
    w_cold = eval_out["w_cold"]
    w_warm = eval_out["w_warm"]
    w_pca = eval_out["w_pca"]

    meta_list = []
    for k in range(1, len(setpoints)):
        meta_list.append({
            "setpoint": float(setpoints[k]),
            "iters": int(iters[k - 1]),
            "solve_time": float(solve_time[k - 1]),
            "l2": float(l2[k - 1]),
            "w_cold": float(w_cold[k - 1]),
            "w_warm": float(w_warm[k - 1]),
            "w_pca": float(w_pca[k - 1]),
        })

    total_time = float(np.sum(solve_time))
    total_iters = int(np.sum(iters))

    print(
        f"[{mode}] total run time = {total_time:.3f} seconds, "
        f"total iterations = {total_iters:d}"
    )

    return eval_out, meta_list


def plot_meta_over_solves(meta_list):
    idx = np.arange(len(meta_list))
    l2_vals = np.array([d["l2"] for d in meta_list])
    t_vals  = np.array([d["solve_time"] for d in meta_list])
    s_vals  = np.array([d["setpoint"] for d in meta_list])

    plt.figure(figsize=(6, 3))
    plt.step(idx, l2_vals, where="mid")
    plt.xlabel("solve index")
    plt.ylabel("L2 tracking error")
    plt.title("NLP L2 error vs solve index")
    plt.tight_layout()

    plt.figure(figsize=(6, 3))
    plt.step(idx, t_vals, where="mid")
    plt.xlabel("solve index")
    plt.ylabel("solve time (s)")
    plt.title("NLP solve time vs solve index")
    plt.tight_layout()

    plt.figure(figsize=(6, 3))
    plt.step(idx, s_vals, where="mid")
    plt.xlabel("solve index")
    plt.ylabel("setpoint")
    plt.title("Setpoint sequence")
    plt.tight_layout()


def stitch_results_sequence(all_results):
    """
    Take list of per-solve results and build one global-time trajectory.
    Avoids duplicating the end point between runs.
    """
    first = all_results[0]
    z = first["z"]
    n_zones = first["Tj"].shape[0]

    t_list = []
    F_list = []
    Tj_list = []
    cA_list = []
    cB_list = []
    cC_list = []
    Tr_list = []

    T_block = first["t"][-1]

    for j, res in enumerate(all_results):
        t_local = res["t"]
        F_local = res["F"]
        Tj_local = res["Tj"]
        cA_local = res["cA"]
        cB_local = res["cB"]
        cC_local = res["cC"]
        Tr_local = res["Tr"]

        # shift time for this run
        t_shift = t_local + j * T_block

        # avoid repeating t=0 for subsequent runs
        if j == 0:
            sl = slice(None)
        else:
            sl = slice(1, None)

        t_list.append(t_shift[sl])
        F_list.append(F_local[sl])
        Tj_list.append(Tj_local[:, sl])
        cA_list.append(cA_local[:, sl])
        cB_list.append(cB_local[:, sl])
        cC_list.append(cC_local[:, sl])
        Tr_list.append(Tr_local[:, sl])

    t = np.concatenate(t_list)
    F = np.concatenate(F_list)
    Tj = np.concatenate(Tj_list, axis=1)
    cA = np.concatenate(cA_list, axis=1)
    cB = np.concatenate(cB_list, axis=1)
    cC = np.concatenate(cC_list, axis=1)
    Tr = np.concatenate(Tr_list, axis=1)

    return dict(
        z=z, t=t,
        F=F, Tj=Tj,
        cA=cA, cB=cB, cC=cC, Tr=Tr,
    )


def plot_outlet_sequence(env, all_results):
    """Outlet B(L,t) (or C) vs global time + setpoint."""
    seq = stitch_results_sequence(all_results)
    t = seq["t"]
    cB = seq["cB"]
    cC = seq["cC"]

    which = "B" if env.target_species.upper() == "B" else "C"
    y_out = cB[-1, :] if which == "B" else cC[-1, :]

    # build piecewise-constant setpoint in global time
    first = all_results[0]
    T_block = first["t"][-1]
    s_glob = []
    t_glob = []
    for j, res in enumerate(all_results):
        t_loc = res["t"]
        s_val = res["setpoint"]
        t_shift = t_loc + j * T_block
        sl = slice(None) if j == 0 else slice(1, None)
        t_glob.append(t_shift[sl])
        s_glob.append(np.full_like(t_shift[sl], s_val))
    t_s = np.concatenate(t_glob)
    s_s = np.concatenate(s_glob)

    plt.figure(figsize=(9, 3.2))
    plt.plot(t, y_out, label=f"Outlet {which}(L,t)")
    plt.step(t_s, s_s, "--", label="Setpoint")
    plt.xlabel("global time (stitched horizons)")
    plt.ylabel(f"{which}(L,t)")
    plt.title("Outlet trajectories over setpoint sequence")
    plt.legend()
    plt.tight_layout()


def plot_MVs_sequence(env, all_results):
    """Inlet flow F(t) and jacket temps by zone over global time."""
    seq = stitch_results_sequence(all_results)
    t = seq["t"]
    F = seq["F"]
    Tj = seq["Tj"]
    n_zones = Tj.shape[0]

    plt.figure(figsize=(9, 3.2))
    plt.plot(t, F)
    plt.xlabel("global time")
    plt.ylabel("F (m^3/h)")
    plt.title("Inlet flow over setpoint sequence")
    plt.tight_layout()

    plt.figure(figsize=(9, 3.2))
    for i in range(n_zones):
        plt.plot(t, Tj[i, :], label=f"Zone {i+1}")
    plt.xlabel("global time")
    plt.ylabel("Tj (K)")
    plt.title("Jacket temperatures by zone")
    plt.legend(ncol=min(5, n_zones))
    plt.tight_layout()


def plot_species_probes_sequence(env, all_results, probes=("inlet", "middle", "outlet")):
    """C_A, C_B, C_C at selected axial positions over global time."""
    seq = stitch_results_sequence(all_results)
    z = seq["z"]
    t = seq["t"]
    cA = seq["cA"]
    cB = seq["cB"]
    cC = seq["cC"]

    N = len(z)
    sel_idx = []
    for p in probes:
        if p == "inlet":
            sel_idx.append(0)
        elif p == "middle":
            sel_idx.append(N // 2)
        elif p == "outlet":
            sel_idx.append(N - 1)
        elif isinstance(p, (int, np.integer)):
            sel_idx.append(int(p))
    # unique
    sel_idx = list(dict.fromkeys(sel_idx))

    # C_A
    plt.figure(figsize=(8, 3))
    for i in sel_idx:
        plt.plot(t, cA[i, :], label=f"z={z[i]:.2f}")
    plt.xlabel("global time")
    plt.ylabel("C_A (mol/m^3)")
    plt.title("C_A at selected axial positions")
    plt.legend(ncol=min(3, len(sel_idx)))
    plt.tight_layout()

    # C_B
    plt.figure(figsize=(8, 3))
    for i in sel_idx:
        plt.plot(t, cB[i, :], label=f"z={z[i]:.2f}")
    plt.xlabel("global time")
    plt.ylabel("C_B (mol/m^3)")
    plt.title("C_B at selected axial positions")
    plt.legend(ncol=min(3, len(sel_idx)))
    plt.tight_layout()

    # C_C
    plt.figure(figsize=(8, 3))
    for i in sel_idx:
        plt.plot(t, cC[i, :], label=f"z={z[i]:.2f}")
    plt.xlabel("global time")
    plt.ylabel("C_C (mol/m^3)")
    plt.title("C_C at selected axial positions")
    plt.legend(ncol=min(3, len(sel_idx)))
    plt.tight_layout()


def plot_outlet_metrics_sequence(env, all_results):
    """
    Outlet conversion of A, selectivity to B, and production rates vs global time.
    """
    seq = stitch_results_sequence(all_results)
    t = seq["t"]
    cA = seq["cA"]
    cB = seq["cB"]
    cC = seq["cC"]
    F = seq["F"]

    yA = cA[-1, :]
    yB = cB[-1, :]
    yC = cC[-1, :]

    cA_in = env.params["cA_in"]
    eps = 1e-9

    X_out = 1.0 - yA / max(cA_in, eps)           # conversion of A at outlet
    Sel_B = yB / np.maximum(yB + yC, eps)        # selectivity to B
    prodB_rate = F * yB                          # mol/h
    prodC_rate = F * yC

    dt_glob = np.mean(np.diff(t))
    cumB = np.cumsum(prodB_rate) * dt_glob
    cumC = np.cumsum(prodC_rate) * dt_glob

    plt.figure(figsize=(8, 3))
    plt.plot(t, X_out)
    plt.xlabel("global time")
    plt.ylabel("X_out")
    plt.title("Outlet conversion of A")
    plt.tight_layout()

    plt.figure(figsize=(8, 3))
    plt.plot(t, Sel_B)
    plt.xlabel("global time")
    plt.ylabel("Selectivity to B")
    plt.title("Outlet selectivity: B / (B + C)")
    plt.tight_layout()

    plt.figure(figsize=(8, 3))
    plt.plot(t, prodB_rate, label="B (mol/h)")
    plt.plot(t, prodC_rate, label="C (mol/h)")
    plt.xlabel("global time")
    plt.ylabel("Production rate (mol/h)")
    plt.title("Outlet production rates")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 3))
    plt.plot(t, cumB, label="B (mol)")
    plt.plot(t, cumC, label="C (mol)")
    plt.xlabel("global time")
    plt.ylabel("Cumulative production (mol)")
    plt.title("Cumulative outlet production")
    plt.legend()
    plt.tight_layout()