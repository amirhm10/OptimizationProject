import numpy as np
import time
import math

from pyomo.environ import (
    ConcreteModel, Var, Param, Constraint, Objective, RangeSet,
    minimize, SolverFactory, value, exp, Suffix
)
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import TerminationCondition


def unflatten_primals_fields(vec, N, K, n_zones):
    """
    Inverse of flatten_primals_fields.
    Given a 1D vec, recover (cA,cB,cC,Tr,F,Tj) with shapes:
      - cA,cB,cC,Tr: (N, K+1)
      - F: (K+1, )
      - Tj: (n_zones, K+1)
    """
    vec = np.asarray(vec, dtype=float).ravel()
    n_t = K + 1

    n_state_block = N * n_t

    idx = 0
    cA = vec[idx:idx + n_state_block].reshape(N, n_t)
    idx += n_state_block
    cB = vec[idx:idx + n_state_block].reshape(N, n_t)
    idx += n_state_block
    cC = vec[idx:idx + n_state_block].reshape(N, n_t)
    idx += n_state_block
    Tr = vec[idx:idx + n_state_block].reshape(N, n_t)
    idx += n_state_block

    F = vec[idx:idx + n_t]
    idx += n_t

    n_tj = n_zones * n_t
    Tj = vec[idx:idx + n_tj].reshape(n_zones, n_t)
    idx += n_tj

    # Just to make sure if the size of the vector completely has been used
    assert idx == vec.size, "unflatten_primals_fields: dimension mismatch."

    return cA, cB, cC, Tr, F, Tj


# ----- helper functions -----
# This function is from IPOPT documentation with a few changes
def ipopt_solve_with_stats(model, solver, max_iter=500, max_cpu_time=120, tee=False):
    """
    Solve with IPOPT and parse its output file to get:
      - status_obj: full Pyomo result
      - solved: bool (termination_condition == optimal)
      - iters: IPOPT "Number of Iterations"
      - cpu_time: IPOPT CPU time (internal + function evals), if parsed
      - wall_time: Python wall-clock time for solver.solve(...)
    """
    TempfileManager.push()
    tempfile = TempfileManager.create_tempfile(suffix='ipopt_out', text=True)

    opts = {
        'output_file': tempfile,
        'max_iter': max_iter,
        'max_cpu_time': max_cpu_time,
    }

    t0 = time.perf_counter()
    try:
        status_obj = solver.solve(model, options=opts, tee=tee)
    except ValueError as e:
        # IPOPT crashed / Pyomo refused to load solution
        t1 = time.perf_counter()
        wall_time = t1 - t0
        print(f"[IPOPT] solve failed with ValueError: {e}")
        TempfileManager.pop(remove=True)
        # Return a "failed" solve
        return None, False, 0, 0.0, None, wall_time

    t1 = time.perf_counter()
    wall_time = t1 - t0

    solved = (status_obj.solver.termination_condition == TerminationCondition.optimal)

    iters = 0
    cpu_time = 0.0
    regu = None

    try:
        with open(tempfile, 'r') as f:
            line_m_2 = None
            line_m_1 = None
            for line in f:
                line_stripped = line.strip()

                if line_stripped.startswith('Number of Iterations'):
                    tokens = line_stripped.split()
                    iters = int(tokens[-1])
                    if line_m_2 is not None:
                        regu = line_m_2.strip()

                elif 'Total CPU secs in IPOPT (w/o function evaluations)' in line_stripped:
                    tokens = line_stripped.split()
                    try:
                        cpu_time += float(tokens[-1])
                    except ValueError:
                        pass

                elif 'Total CPU secs in NLP function evaluations' in line_stripped:
                    tokens = line_stripped.split()
                    try:
                        cpu_time += float(tokens[-1])
                    except ValueError:
                        pass

                line_m_2 = line_m_1
                line_m_1 = line
    except OSError:
        # e.g., IPOPT never wrote the output file
        pass

    TempfileManager.pop(remove=True)
    return status_obj, solved, iters, cpu_time, regu, wall_time


# ---- Zone mapping along the reactors PDEs ----
def _build_zone_sampling(L, N, n_zones):
    """Return (zone_bounds, z_nodes, node_zone) given reactor length L and N nodes."""
    zone_bounds = np.linspace(0.0, L, n_zones + 1)
    dz = L / (N - 1)
    z_nodes = np.array([i * dz for i in range(N)])

    def zone_of_z(z):
        for zi in range(n_zones):
            if (z >= zone_bounds[zi]) and (z <= zone_bounds[zi + 1] + 1e-12):
                return zi + 1
        return n_zones

    node_zone = {i: zone_of_z(z_nodes[i]) for i in range(N)}
    return zone_bounds, z_nodes, node_zone


# ---- mapping from action outputs to real outputs ----
def map_log_range(a, low, high):
    """
    Map a in [-1,1] to [low, high] on a log10 scale.
    """
    a = float(np.clip(a, -1.0, 1.0))
    log_lo = math.log10(low)
    log_hi = math.log10(high)
    log_val = 0.5 * (a + 1.0) * (log_hi - log_lo) + log_lo
    return 10.0 ** log_val

def inv_map_log_range(val, low, high):
    """
    Approximate inverse of map_log_range:
      val in [low, high] -> a in [-1,1]
    """
    val = float(np.clip(val, low, high))
    log_lo = math.log10(low)
    log_hi = math.log10(high)
    alpha = (math.log10(val) - log_lo) / (log_hi - log_lo + 1e-16)
    alpha = np.clip(alpha, 0.0, 1.0)
    return float(2.0 * alpha - 1.0)


# ---- PFR environment ----
class PfrIpoptEnv:
    def __init__(
            self,
            params,
            N=10,
            K=10,
            L=20.0,
            dt=0.02,
            target_species="B",
            n_zones=5,
            ipopt_options=None,
            use_warm_start=True,
            max_iter=500,
            pca_lib=None
    ):
        self.params = params
        self.N = N
        self.K = K
        self.L = L
        self.dt = dt
        self.dz = L / (N - 1)
        self.target_species = target_species
        self.n_zones = n_zones
        self.use_warm_start = use_warm_start
        self.max_iter = max_iter

        self.pca_lib = pca_lib

        # grid + zones
        self.zone_bounds, self.z_nodes, self.node_zone = _build_zone_sampling(L, N, n_zones)

        # build model
        self.m = self._build_model()

        # IPOPT solver
        self.solver = SolverFactory('ipopt')
        default_opts = {
            'tol': 1e-6,
            'linear_solver': 'mumps',
            'mu_init': 1e-2,
            'print_level': 5,
            'max_iter': self.max_iter,
            'acceptable_tol': 1e-4,
            'acceptable_iter': 10,
        }
        if ipopt_options:
            default_opts.update(ipopt_options)
        self.solver.options.update(default_opts)

        if self.use_warm_start:
            self.solver.options["warm_start_init_point"] = "yes"
            # self.solver.options["warm_start_bound_push"] = 1e-6
            # self.solver.options["warm_start_mult_bound_push"] = 1e-6

        # bookkeeping
        self.prev_results = None
        self.prev_setpoint = None
        self.current_setpoint = None

        # for convex-combo warm start: store primals + duals
        self._init_warm_cold_storage()

        # last actual convex weights on [cold, warm, pca]
        self.last_w_cold = 1.0
        self.last_w_warm = 0.0
        self.last_w_pca  = 0.0

        # RL-tuned IPOPT parameter ranges (log-scale)
        self.mu_min, self.mu_max = 1e-4, 1e-1
        self.tol_min, self.tol_max = 1e-8, 1e-4
        self.acc_tol_min, self.acc_tol_max = 1e-6, 1e-3

        # store last chosen solver params (start from defaults)
        self.last_mu_init = self.solver.options['mu_init']
        self.last_tol = self.solver.options['tol']
        self.last_acc_tol = self.solver.options['acceptable_tol']

    # ---------------- Model builder ----------------
    def _build_model(self):
        N = self.N
        K = self.K
        dt = self.dt
        target_species = self.target_species

        m = ConcreteModel()

        m.I = RangeSet(0, N - 1)
        m.K = RangeSet(0, K)
        m.Z = RangeSet(1, self.n_zones)

        m.s = Param(initialize=40.0, mutable=True) # This will initialize setpoint of the desired product at 40.0

        cA0_init = {i: (self.params["cA_in"] if i == 0 else 0.0) for i in range(N)}
        cB0_init = {i: self.params["cB_in"] for i in range(N)}
        cC0_init = {i: self.params["cC_in"] for i in range(N)}
        Tr0_init = {i: self.params["T_in"] for i in range(N)}

        m.cA0 = Param(m.I, initialize=cA0_init, mutable=True)
        m.cB0 = Param(m.I, initialize=cB0_init, mutable=True)
        m.cC0 = Param(m.I, initialize=cC0_init, mutable=True)
        m.Tr0 = Param(m.I, initialize=Tr0_init, mutable=True)

        # states
        m.cA = Var(m.I, m.K, bounds=(0.0, None), initialize=self.params["cA_in"])
        m.cB = Var(m.I, m.K, bounds=(0.0, None), initialize=0.0)
        m.cC = Var(m.I, m.K, bounds=(0.0, None), initialize=0.0)
        m.Tr = Var(
            m.I, m.K,
            bounds=(self.params["T_lo"], self.params["T_hi"]),
            initialize=self.params["T_in"]
        )

        # MVs
        F_min = self.params["F_min"]
        F_max = self.params["F_max"]

        m.F = Var(
            m.K,
            bounds=(F_min, F_max),
            initialize=1.2
        )
        Tj_min = self.params["Tj_min"]
        Tj_max = self.params["Tj_max"]

        m.Tj = Var(
            m.Z, m.K,
            bounds=(Tj_min, Tj_max),
            initialize=350.0
        )

        def Tj_at(m, i, k):
            return m.Tj[self.node_zone[int(i)], k]

        # ICs
        def ic_cA(m, i):
            return m.cA[i, 0] == m.cA0[i]

        def ic_cB(m, i):
            return m.cB[i, 0] == m.cB0[i]

        def ic_cC(m, i):
            return m.cC[i, 0] == m.cC0[i]

        def ic_Tr(m, i):
            return m.Tr[i, 0] == m.Tr0[i]

        m.icA = Constraint(m.I, rule=ic_cA)
        m.icB = Constraint(m.I, rule=ic_cB)
        m.icC = Constraint(m.I, rule=ic_cC)
        m.icTr = Constraint(m.I, rule=ic_Tr)

        # inlet BCs
        def bcA_in(m, k):
            return m.cA[0, k] == self.params["cA_in"]

        def bcB_in(m, k):
            return m.cB[0, k] == self.params["cB_in"]

        def bcC_in(m, k):
            return m.cC[0, k] == self.params["cC_in"]

        def bcT_in(m, k):
            return m.Tr[0, k] == self.params["T_in"]

        m.bcA = Constraint(m.K, rule=bcA_in)
        m.bcB = Constraint(m.K, rule=bcB_in)
        m.bcC = Constraint(m.K, rule=bcC_in)
        m.bcT = Constraint(m.K, rule=bcT_in)

        # kinetics
        def R1(m, i, k1):
            T = m.Tr[i, k1]
            return (self.params["k01"] *
                    exp((self.params["E1"] / self.params["Rgas"]) *
                        (1.0 / 300.0 - 1.0 / T)) *
                    m.cA[i, k1])

        def R2(m, i, k1):
            T = m.Tr[i, k1]
            return (self.params["k02"] *
                    exp((self.params["E2"] / self.params["Rgas"]) *
                        (1.0 / 300.0 - 1.0 / T)) *
                    (m.cA[i, k1] ** 2))

        def u_at(m, k1):
            return m.F[k1] / self.params["A_t"]

        # PDEs
        def pde_cA(m, i, k):
            if i == 0 or k == self.K:
                return Constraint.Skip
            k1 = k + 1
            u = u_at(m, k1)
            conv = u * (m.cA[i, k1] - m.cA[i - 1, k1]) / self.dz
            return (m.cA[i, k1] - m.cA[i, k]) / self.dt + conv + (R1(m, i, k1) + R2(m, i, k1)) == 0.0

        def pde_cB(m, i, k):
            if i == 0 or k == self.K:
                return Constraint.Skip
            k1 = k + 1
            u = u_at(m, k1)
            conv = u * (m.cB[i, k1] - m.cB[i - 1, k1]) / self.dz
            return (m.cB[i, k1] - m.cB[i, k]) / self.dt + conv - R1(m, i, k1) == 0.0

        def pde_cC(m, i, k):
            if i == 0 or k == self.K:
                return Constraint.Skip
            k1 = k + 1
            u = u_at(m, k1)
            conv = u * (m.cC[i, k1] - m.cC[i - 1, k1]) / self.dz
            return (m.cC[i, k1] - m.cC[i, k]) / self.dt + conv - 0.5 * R2(m, i, k1) == 0.0

        def pde_T(m, i, k):
            if i == 0 or k == self.K:
                return Constraint.Skip
            k1 = k + 1
            u = u_at(m, k1)
            conv = u * (m.Tr[i, k1] - m.Tr[i - 1, k1]) / self.dz
            q_rxn = (-self.params["dH1"]) * R1(m, i, k1) + (-self.params["dH2"]) * R2(m, i, k1)
            q_ht = self.params["Ua"] * (Tj_at(m, i, k1) - m.Tr[i, k1])
            return (m.Tr[i, k1] - m.Tr[i, k]) / self.dt + conv - (q_rxn + q_ht) / self.params["rhoCp"] == 0.0

        m.pdeA = Constraint(m.I, m.K, rule=pde_cA)
        m.pdeB = Constraint(m.I, m.K, rule=pde_cB)
        m.pdeC = Constraint(m.I, m.K, rule=pde_cC)
        m.pdeT = Constraint(m.I, m.K, rule=pde_T)

        # optional outlet T limit
        def outlet_T_limit(m, k):
            return m.Tr[self.N - 1, k] <= 640.0

        m.outlet_T_limit = Constraint(m.K, rule=outlet_T_limit)

        # outlet objective
        out_idx = N - 1

        def y_of_k(m, k):
            if target_species.upper() == 'B':
                return m.cB[out_idx, k]
            else:
                return m.cC[out_idx, k]

        def obj_rule(m):
            return sum((y_of_k(m, k) - m.s) ** 2 for k in m.K) * self.dt

        m.obj = Objective(rule=obj_rule, sense=minimize)

        # IPOPT suffixes for primals + duals warm-start
        m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)

        return m

    # ------------ warm/cold storage ------------
    def _init_warm_cold_storage(self):
        m = self.m
        # fixed order of vars/constraints
        self.var_list = list(m.component_data_objects(Var, active=True))
        self.con_list = list(m.component_data_objects(Constraint, active=True))

        self.cold_primals = np.array([float(v.value) for v in self.var_list], dtype=float)
        self.warm_primals = self.cold_primals.copy()
        self.pca_primals = self.cold_primals.copy() # naming is pca but will actually work with others as well

        self.cold_duals = np.zeros(len(self.con_list), dtype=float)
        self.warm_duals = np.zeros(len(self.con_list), dtype=float)

        self.cold_zL = np.zeros(len(self.var_list), dtype=float)
        self.cold_zU = np.zeros(len(self.var_list), dtype=float)
        self.warm_zL = np.zeros(len(self.var_list), dtype=float)
        self.warm_zU = np.zeros(len(self.var_list), dtype=float)


    def _set_mixed_initial_guess(self, w_cold, w_warm, w_pca, lam_dual=0.0):
        """
        Use a convex combination of:
          - cold_primals
          - warm_primals (from previous solve)
          - pca_primals  (from library)

        Weights must satisfy w_cold + w_warm + w_pca = 1.
        """
        m = self.m

        # Normalize just in case
        w = np.array([w_cold, w_warm, w_pca], dtype=float)
        w = np.clip(w, 0.0, None) # clip to have no negative mixing
        s = w.sum()
        if s <= 0:
            w[:] = np.array([1.0, 0.0, 0.0])
        else:
            w /= s
        w_cold, w_warm, w_pca = w

        # --- store them for the next state ---
        self.last_w_cold = float(w_cold)
        self.last_w_warm = float(w_warm)
        self.last_w_pca = float(w_pca)
        self.last_lambda_prim = float(w_warm + w_pca)
        self.last_lambda_dual = float(np.clip(lam_dual, 0.0, 1.0))

        # build initial primals
        x_init = (
                w_cold * self.cold_primals +
                w_warm * self.warm_primals +
                w_pca * self.pca_primals
        )

        # pushing values to pyomo model
        for val, v in zip(x_init, self.var_list):
            v.set_value(float(val), skip_validation=True)

        # if the RL agent is also tuning the duals as well
        m.dual.clear()
        m.ipopt_zL_in.clear()
        m.ipopt_zU_in.clear()

        lam_dual = self.last_lambda_dual
        if self.use_warm_start and lam_dual > 0.0:
            lambda_init = lam_dual * self.warm_duals
            for val, c in zip(lambda_init, self.con_list):
                if abs(val) > 1e-14:
                    m.dual[c] = float(val)
            zL_init = lam_dual * self.warm_zL
            zU_init = lam_dual * self.warm_zU
            for valL, valU, v in zip(zL_init, zU_init, self.var_list):
                if (not v.is_fixed()) and (abs(valL) > 1e-14 or abs(valU) > 1e-14):
                    m.ipopt_zL_in[v] = float(valL)
                    m.ipopt_zU_in[v] = float(valU)

    def _set_convex_initial_guess(self, lam_prim, lam_dual):
        """
        Warm-start both primals and (optionally) duals / bound multipliers.

        lam_prim ∈ [0,1]: convex combo of cold vs warm primals.
        lam_dual ∈ [0,1]: scaling of warm duals/bound multipliers
                          (0 → no dual warm-start; 1 → full last duals).
        """
        m = self.m
        lam_prim = float(np.clip(lam_prim, 0.0, 1.0))
        lam_dual = float(np.clip(lam_dual, 0.0, 1.0))

        # ---------- primals: convex combo of cold & warm ----------
        x_init = (1.0 - lam_prim) * self.cold_primals + lam_prim * self.warm_primals
        for val, v in zip(x_init, self.var_list):
            v.set_value(float(val), skip_validation=True)

        # ---------- DUALS + BOUND MULTIPLIERS ----------
        # clear old suffix values
        m.dual.clear()
        m.ipopt_zL_in.clear()
        m.ipopt_zU_in.clear()

        if self.use_warm_start and lam_dual > 0.0:
            # constraint multipliers lambdas
            lambda_init = lam_dual * self.warm_duals
            for val, c in zip(lambda_init, self.con_list):
                # skip tiny values – no point in writing exact zeros
                if abs(val) > 1e-14:
                    m.dual[c] = float(val)

            # bound multipliers zL, zU
            zL_init = lam_dual * self.warm_zL
            zU_init = lam_dual * self.warm_zU
            for valL, valU, v in zip(zL_init, zU_init, self.var_list):
                # only write for variables that:
                #  - are not fixed, and
                #  - had nonzero multipliers last time (so they were in the NLP)
                if (not v.is_fixed()) and (abs(valL) > 1e-14 or abs(valU) > 1e-14):
                    m.ipopt_zL_in[v] = float(valL)
                    m.ipopt_zU_in[v] = float(valU)

        self.last_lambda_prim = lam_prim
        self.last_lambda_dual = lam_dual

    def _update_warm_from_solution(self):
        """After a solve, copy current solution into warm_* buffers."""
        m = self.m
        self.warm_primals = np.array([float(v.value) for v in self.var_list], dtype=float)

        # duals
        warm_duals = []
        for c in self.con_list:
            val = 0.0
            if c in m.dual:
                val = float(m.dual[c])
            warm_duals.append(val)
        self.warm_duals = np.array(warm_duals, dtype=float)

        # bound multipliers
        warm_zL = []
        warm_zU = []
        for v in self.var_list:
            valL = float(m.ipopt_zL_out.get(v, 0.0))
            valU = float(m.ipopt_zU_out.get(v, 0.0))
            warm_zL.append(valL)
            warm_zU.append(valU)
        self.warm_zL = np.array(warm_zL, dtype=float)
        self.warm_zU = np.array(warm_zU, dtype=float)

    # ------------ solve wrapper ------------
    def _solve_current_nlp(self, tee=False, print_solver=False):
        m = self.m
        K = self.K
        dt = self.dt
        n_zones = self.n_zones
        target_species = self.target_species
        s_val = float(m.s.value)

        # run nlp with full details parsed
        status_obj, solved, iters, cpu_time, regu, wall_time = ipopt_solve_with_stats(
            m, self.solver, max_iter=self.max_iter, max_cpu_time=120, tee=tee
        )

        if not solved:
            print(f"[PFR ENV] IPOPT failed or returned non-optimal status at s={s_val:.2f}")

        # collect solution
        z = self.z_nodes
        t = np.array([k * dt for k in range(K + 1)])

        cA = np.array([[value(m.cA[i, k]) for k in m.K] for i in m.I])
        cB = np.array([[value(m.cB[i, k]) for k in m.K] for i in m.I])
        cC = np.array([[value(m.cC[i, k]) for k in m.K] for i in m.I])
        Tr = np.array([[value(m.Tr[i, k]) for k in m.K] for i in m.I])

        F = np.array([value(m.F[k]) for k in m.K])
        Tj = np.zeros((n_zones, K + 1))
        for zi in range(1, n_zones + 1):
            Tj[zi - 1, :] = np.array([value(m.Tj[zi, k]) for k in m.K])

        if target_species.upper() == 'B':
            y_out = cB[-1, :]
        else:
            y_out = cC[-1, :]

        l2 = math.sqrt(np.sum((y_out - s_val) ** 2) * dt)

        results = dict(
            z=z, t=t,
            cA=cA, cB=cB, cC=cC, Tr=Tr,
            F=F, Tj=Tj,
            y_out=y_out,
            l2=l2,
            solve_time=wall_time,
            iters=iters,
            solved=solved,
            solver_status=status_obj,
            setpoint=s_val,
        )

        if print_solver:
            print(
                f"[PFR ENV] s={s_val:.2f}, L2={l2:.3e}, "
                f"iters={iters:d}, solved={solved}, "
                f"time={wall_time:.3f} s"
            )

        # Only update warm-start buffers if the solve actually succeeded
        if solved:
            self._update_warm_from_solution()

        return results

    # -------- IC update between solves --------
    def _update_initial_state_from_results(self, results):
        m = self.m
        cA_end = results["cA"][:, -1]
        cB_end = results["cB"][:, -1]
        cC_end = results["cC"][:, -1]
        Tr_end = results["Tr"][:, -1]

        for i in m.I:
            idx = int(i)
            m.cA0[i].set_value(float(cA_end[idx]))
            m.cB0[i].set_value(float(cB_end[idx]))
            m.cC0[i].set_value(float(cC_end[idx]))
            m.Tr0[i].set_value(float(Tr_end[idx]))

    # -------- reset --------
    def reset(self, s0, tee=False):
        m = self.m

        # reset ICs
        for i in m.I:
            idx = int(i)
            m.cA0[i].set_value(self.params["cA_in"] if idx == 0 else 0.0)
            m.cB0[i].set_value(self.params["cB_in"])
            m.cC0[i].set_value(self.params["cC_in"])
            m.Tr0[i].set_value(self.params["T_in"])

        m.s.set_value(s0)
        self.current_setpoint = s0
        self.prev_setpoint = None
        self.prev_results = None

        # first solve: pure cold start (lambda_p = 0, lambda_d = 0)
        self._set_convex_initial_guess(lam_prim=0.0, lam_dual=0.0)
        results = self._solve_current_nlp(tee=tee)

        self.prev_results = results
        self.prev_setpoint = s0
        self._update_initial_state_from_results(results)

        obs = self._make_observation(results, prev_results=None, s_prev=s0, s_curr=s0)
        return obs

    def step(self, s_new, action=None, tee=False):
        m = self.m
        s_new = float(s_new)
        m.s.set_value(s_new)
        self.current_setpoint = s_new

        # previous setpoint to extract optimal solution from the features
        if self.prev_setpoint is None:
            s_prev = s_new
        else:
            s_prev = self.prev_setpoint

        # building primal optimal warm start
        if self.pca_lib is not None:
            x_pca_vec = self.pca_lib.build_fine_primals(
                fine_env=self,
                s_prev=s_prev,
                s_curr=s_new,
            )
            # store into env.pca_primals
            if x_pca_vec.shape[0] != self.cold_primals.shape[0]:
                raise ValueError("PCA primals length mismatch with env.var_list.")
            self.pca_primals = np.asarray(x_pca_vec, dtype=float)
        else:
            # in case of not having optimal solution library
            self.pca_primals = self.warm_primals.copy()

        # taking rl actions and decoding it back to the real outputs range
        if action is not None:
            a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
            if a.shape[0] < 5:
                raise ValueError("action must have length >=5: [a_warm, a_pca, a_mu, a_tol, a_atol]")
            a_warm, a_pca, a_mu, a_tol, a_atol = a[:5]

            # mixing weights
            if self.pca_lib is None:
                # no PCA/AE/VAE library → ignore a_pca, only cold vs warm
                w_warm = 0.5 * (a_warm + 1.0)  # map [-1,1] → [0,1]
                w_warm = float(np.clip(w_warm, 0.0, 1.0))
                w_cold = 1.0 - w_warm
                w_pca = 0.0
            else:
                # library available → 3-way softmax on [cold, warm, pca]
                logits = np.array([0.0, a_warm, a_pca], dtype=float)
                logits -= logits.max()
                exp_logits = np.exp(logits)
                weights = exp_logits / exp_logits.sum()
                w_cold, w_warm, w_pca = weights

            # solver hyperparameters (log-scale)
            mu_init = map_log_range(a_mu, self.mu_min, self.mu_max)
            tol = map_log_range(a_tol, self.tol_min, self.tol_max)
            acc_tol = map_log_range(a_atol, self.acc_tol_min, self.acc_tol_max)

            self.solver.options["mu_init"] = mu_init
            self.solver.options["tol"] = tol
            self.solver.options["acceptable_tol"] = acc_tol

            self.last_mu_init = mu_init
            self.last_tol = tol
            self.last_acc_tol = acc_tol

        else:
            # only warm-start
            w_cold, w_warm, w_pca = 0.0, 1.0, 0.0
            mu_init = self.solver.options["mu_init"]
            tol = self.solver.options["tol"]
            acc_tol = self.solver.options["acceptable_tol"]

        # drt thr initial primals and duals
        self._set_mixed_initial_guess(
            w_cold=w_cold,
            w_warm=w_warm,
            w_pca=w_pca,
            lam_dual=w_warm,
        )

        # solve
        results = self._solve_current_nlp(tee=tee)

        # observation
        if self.prev_setpoint is None:
            s_prev = s_new
            prev_results = None
        else:
            s_prev = self.prev_setpoint
            prev_results = self.prev_results

        obs = self._make_observation(
            results,
            prev_results=prev_results,
            s_prev=s_prev,
            s_curr=s_new
        )

        info = {
            "solve_time": results["solve_time"],
            "l2": results["l2"],
            "iters": results["iters"],
            "setpoint": s_new,
            "y_out": results["y_out"],
            "F": results["F"],
            "Tj": results["Tj"],
            "raw_results": results,
            "solved": results["solved"],
            "lambda_prim": self.last_lambda_prim,
            "lambda_dual": self.last_lambda_dual,
            "mu_init": self.last_mu_init,
            "tol": self.last_tol,
            "acceptable_tol": self.last_acc_tol,
            "w_cold": self.last_w_cold,
            "w_warm": self.last_w_warm,
            "w_pca":  self.last_w_pca,
        }

        # update for next call
        self.prev_results = results
        self.prev_setpoint = s_new
        self._update_initial_state_from_results(results)

        return obs, info

    def step_no_rl(
            self,
            s_new,
            lam_prim=1.0,
            lam_dual=0.0,
            tee=False,
            mu_init=None,
            tol=None,
            acc_tol=None,
    ):

        # this is just using the warm-strat with a fixed lambda
        m = self.m
        m.s.set_value(float(s_new))
        self.current_setpoint = float(s_new)

        # to override the solver options if provided
        if mu_init is not None:
            self.solver.options["mu_init"] = float(mu_init)
        if tol is not None:
            self.solver.options["tol"] = float(tol)
        if acc_tol is not None:
            self.solver.options["acceptable_tol"] = float(acc_tol)

        # safety bounds
        lam_prim = float(np.clip(lam_prim, 0.0, 1.0))
        lam_dual = float(np.clip(lam_dual, 0.0, 1.0))

        self._set_convex_initial_guess(lam_prim=lam_prim, lam_dual=lam_dual)

        # Solve NLP
        results = self._solve_current_nlp(tee=tee)

        # Build observation
        if self.prev_setpoint is None:
            s_prev = float(s_new)
            prev_results = None
        else:
            s_prev = self.prev_setpoint
            prev_results = self.prev_results

        obs = self._make_observation(
            results,
            prev_results=prev_results,
            s_prev=s_prev,
            s_curr=float(s_new),
        )

        info = {
            "solve_time": results["solve_time"],
            "l2": results["l2"],
            "iters": results["iters"],
            "setpoint": float(s_new),
            "y_out": results["y_out"],
            "F": results["F"],
            "Tj": results["Tj"],
            "raw_results": results,
            "solved": results["solved"],
            "lambda_prim": self.last_lambda_prim,
            "lambda_dual": self.last_lambda_dual,
            "mu_init": self.last_mu_init,
            "tol": self.last_tol,
            "acceptable_tol": self.last_acc_tol,
        }

        # Update for next call
        self.prev_results = results
        self.prev_setpoint = float(s_new)
        self._update_initial_state_from_results(results)

        return obs, info

    def step_with_pca_primals(self, s_new, x_vec, lam_dual=0.0, tee=False):
        m = self.m
        s_new = float(s_new)
        m.s.set_value(s_new)
        self.current_setpoint = s_new

        # --- 1) unpack x_vec into fields ---
        x_vec = np.asarray(x_vec, dtype=float).ravel()
        N, K, Z = self.N, self.K, self.n_zones

        cA, cB, cC, Tr, F, Tj = unflatten_primals_fields(
            x_vec, N=N, K=K, n_zones=Z
        )

        n_t = K + 1

        # --- 2) write these fields into the model as initial guesses ---
        # states
        for i in m.I:
            ii = int(i)
            for k in m.K:
                kk = int(k)
                m.cA[i, k].set_value(float(cA[ii, kk]), skip_validation=True)
                m.cB[i, k].set_value(float(cB[ii, kk]), skip_validation=True)
                m.cC[i, k].set_value(float(cC[ii, kk]), skip_validation=True)
                m.Tr[i, k].set_value(float(Tr[ii, kk]), skip_validation=True)

        # F
        for k in m.K:
            kk = int(k)
            m.F[k].set_value(float(F[kk]), skip_validation=True)

        # Tj
        for zi in m.Z:
            z_idx = int(zi) - 1
            for k in m.K:
                kk = int(k)
                m.Tj[zi, k].set_value(float(Tj[z_idx, kk]), skip_validation=True)

        # --- 3) duals: clear, or optionally reuse warm_duals ---
        m.dual.clear()
        m.ipopt_zL_in.clear()
        m.ipopt_zU_in.clear()

        lam_dual = float(np.clip(lam_dual, 0.0, 1.0))
        if self.use_warm_start and lam_dual > 0.0:
            # reuse last duals / bound multipliers if desired
            lambda_init = lam_dual * self.warm_duals
            for val, c in zip(lambda_init, self.con_list):
                if abs(val) > 1e-14:
                    m.dual[c] = float(val)

            zL_init = lam_dual * self.warm_zL
            zU_init = lam_dual * self.warm_zU
            for valL, valU, v in zip(zL_init, zU_init, self.var_list):
                if (not v.is_fixed()) and (abs(valL) > 1e-14 or abs(valU) > 1e-14):
                    m.ipopt_zL_in[v] = float(valL)
                    m.ipopt_zU_in[v] = float(valU)

        # --- 4) logging for RL state: "100% PCA" primals ---
        self.last_w_cold = 0.0
        self.last_w_warm = 0.0
        self.last_w_pca = 1.0
        self.last_lambda_prim = 1.0
        self.last_lambda_dual = lam_dual

        # --- 5) solve ---
        results = self._solve_current_nlp(tee=tee)

        # --- 6) build observation, like in step() / step_no_rl ---
        if self.prev_setpoint is None:
            s_prev = float(s_new)
            prev_results = None
        else:
            s_prev = self.prev_setpoint
            prev_results = self.prev_results

        obs = self._make_observation(
            results,
            prev_results=prev_results,
            s_prev=s_prev,
            s_curr=float(s_new),
        )

        info = {
            "solve_time": results["solve_time"],
            "l2": results["l2"],
            "iters": results["iters"],
            "setpoint": float(s_new),
            "y_out": results["y_out"],
            "F": results["F"],
            "Tj": results["Tj"],
            "raw_results": results,
            "solved": results["solved"],
            "lambda_prim": self.last_lambda_prim,
            "lambda_dual": self.last_lambda_dual,
        }

        # update warm buffers from the *actual* solution
        self.prev_results = results
        self.prev_setpoint = float(s_new)
        self._update_initial_state_from_results(results)

        return obs, info

    def _make_observation(self, results, prev_results, s_prev, s_curr):
        """
        Build a normalized feature vector for RL.
        """
        it_curr = results["iters"]
        t_curr = results["solve_time"]
        solved_curr = 1.0 if results["solved"] else 0.0

        # previous-step diagnostics
        if prev_results is None:
            it_prev = 0.0
            t_prev = t_curr
            solved_prev = solved_curr
        else:
            it_prev = float(prev_results["iters"])
            t_prev = float(prev_results["solve_time"])
            solved_prev = 1.0 if prev_results["solved"] else 0.0

        # --- setpoint normalization (assuming [20,60]) ---
        def norm_s(s):
            return (s - 40.0) / 20.0

        s_prev_n = norm_s(s_prev)
        s_curr_n = norm_s(s_curr)
        ds_n = norm_s(s_curr - s_prev)

        # --- iterations + time normalization ---
        it_prev_n = min(it_prev / self.max_iter, 1.0)
        # scaling time
        t_scale = 0.5
        t_prev_n = min(t_prev / t_scale, 10.0)

        # --- last mixing weights and lambdas ---
        a_prim_last = 2.0 * self.last_lambda_prim - 1.0
        a_dual_last = 2.0 * self.last_lambda_dual - 1.0

        w_cold_last = self.last_w_cold
        w_warm_last = self.last_w_warm
        w_pca_last = self.last_w_pca

        # --- last solver settings (log-normalized to [-1,1]) ---
        mu_last = float(self.last_mu_init)
        tol_last = float(self.last_tol)
        acc_last = float(self.last_acc_tol)

        mu_last_n = inv_map_log_range(mu_last, self.mu_min, self.mu_max)
        tol_last_n = inv_map_log_range(tol_last, self.tol_min, self.tol_max)
        acc_last_n = inv_map_log_range(acc_last, self.acc_tol_min, self.acc_tol_max)

        # physical info
        # outlet product at final time
        y_out_final = float(results["y_out"][-1])
        # range [0, 100] mol/m^3 then scale to [-1,1]
        y_out_final_n = (y_out_final - 50.0) / 50.0

        Tr_arr = results["Tr"]  # shape (N, K+1)
        Tr_max = float(Tr_arr.max())
        # temperature in [280, 700]
        Tr_max_n = (Tr_max - 490.0) / 210.0

        F_arr = results["F"]  # shape (K+1,)
        F_mean = float(F_arr.mean())
        F_min = self.params["F_min"]
        F_max = self.params["F_max"]
        F_mean_n = 2.0 * (F_mean - F_min) / (F_max - F_min) - 1.0

        Tj_arr = results["Tj"]  # shape (n_zones, K+1)
        Tj_mean = float(Tj_arr.mean())
        Tj_min = self.params["Tj_min"]
        Tj_max = self.params["Tj_max"]
        Tj_mean_n = 2.0 * (Tj_mean - Tj_min) / (Tj_max - Tj_min) - 1.0

        # KKT like info
        # warm_zL / warm_zU / warm_duals
        z_mag = np.maximum(np.abs(self.warm_zL), np.abs(self.warm_zU))
        active_frac = float(np.mean(z_mag > 1e-4))  # in [0,1]

        dual_mean = float(np.mean(np.abs(self.warm_duals))) if self.warm_duals.size > 0 else 0.0
        dual_mean_n = np.tanh(dual_mean)  # squash to [-1,1]

        state_vec = np.array([
            # setpoint info
            s_prev_n,
            s_curr_n,
            ds_n,

            # previous difficulty
            it_prev_n,
            t_prev_n,
            solved_prev,

            # last mixing weights / lambdas
            a_prim_last,
            a_dual_last,
            w_warm_last,
            w_pca_last,

            # last solver options (log-normalized)
            mu_last_n,
            tol_last_n,
            acc_last_n,

            # physical info
            y_out_final_n,
            Tr_max_n,
            F_mean_n,
            Tj_mean_n,

            # KKT-ish summaries
            active_frac,
            dual_mean_n,
        ], dtype=np.float32)

        obs = {
            "state_vec": state_vec,
            "s_prev": s_prev,
            "s_curr": s_curr,
            "iters_curr": it_curr,
        }
        if prev_results is not None:
            obs.update({
                "iters_prev": it_prev,
                "time_prev": t_prev,
            })
        return obs

    def get_last_results(self):
        return self.prev_results

