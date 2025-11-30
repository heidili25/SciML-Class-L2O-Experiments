# figures.py
"""
Compare runtime and OPF-style accuracy of:
- A1: direct NN mapping to AC-OPF labels
- A2: physics-informed NN mapping to AC-OPF labels
- B2: B1 DC-hot-start + B2 residual refinement

Metrics:
- Runtime per batch (forward pass only, no PF/OPF inside timing)
- Constraint violations (Pg, V, line loading)
- L2 errors vs AC-OPF: Pg, Vm, theta
- Optimality gap J_model - J_ACOPF
"""

import time
import copy
import numpy as np
import torch
import pandapower as pp
import pandapower.networks as pn
import matplotlib.pyplot as plt

# --- Import model architectures / helpers ---
from model_a1 import DirectMap, setup_opf_costs   # A1/A2 architecture + cost setup
from model_b1 import HotStartMap, DCFeasibilityLayer
from model_b2 import B2Refine, inference_b2

torch.serialization.add_safe_globals([HotStartMap, DCFeasibilityLayer, B2Refine])


# -----------------------------------------------------
# 1. Small timing helper
# -----------------------------------------------------
def timing(fn, X, n_runs: int = 30):
    """
    Measure average runtime [seconds] of fn(X) over n_runs repetitions.
    We assume fn does pure forward inference (no printing, no PF/OPF).
    """
    t0 = time.time()
    for _ in range(n_runs):
        fn(X)
    t1 = time.time()
    return (t1 - t0) / n_runs


# -----------------------------------------------------
# 2. Constraint violation helpers
# -----------------------------------------------------
def compute_gen_violation(net) -> float:
    """
    Sum of generator active power violations (MW).
    Uses net.gen[min_p_mw, max_p_mw] and net.res_gen.p_mw.
    """
    p = net.res_gen["p_mw"].values
    pmin = net.gen["min_p_mw"].values
    pmax = net.gen["max_p_mw"].values
    lower = np.maximum(0.0, pmin - p)
    upper = np.maximum(0.0, p - pmax)
    return float(np.sum(lower + upper))


def compute_voltage_violation(net) -> float:
    """
    Sum of voltage magnitude violations (pu).
    Uses net.bus[min_vm_pu, max_vm_pu] and net.res_bus.vm_pu.
    """
    v = net.res_bus["vm_pu"].values
    vmin = net.bus["min_vm_pu"].values
    vmax = net.bus["max_vm_pu"].values
    lower = np.maximum(0.0, vmin - v)
    upper = np.maximum(0.0, v - vmax)
    return float(np.sum(lower + upper))


def compute_line_violation(net) -> float:
    """
    Sum of line loading violations (% over 100%).
    Uses net.res_line.loading_percent.
    """
    if len(net.line) == 0:
        return 0.0
    loading = net.res_line["loading_percent"].values
    over = np.maximum(0.0, loading - 100.0)
    return float(np.sum(over))


def l2_norm(a: np.ndarray, b: np.ndarray) -> float:
    """L2 norm ||a - b||_2."""
    return float(np.linalg.norm(a - b))


# -----------------------------------------------------
# OPF objective function: J(Pg) = sum cp1 * Pg
# -----------------------------------------------------
def opf_objective(net, Pg):
    """
    Computes the OPF objective J(Pg) = c^T Pg using net.poly_cost 
    which stores cp1_eur_per_mw = linear cost coefficients.
    """
    cp1 = net.poly_cost.cp1_eur_per_mw.values
    return float(np.sum(cp1 * Pg))


# -----------------------------------------------------
# 3. Helpers to (re)build network states
# -----------------------------------------------------
def set_loads_from_X(net, x_vec: np.ndarray):
    """
    Given a pandapower net and an input vector x = [P_load, Q_load],
    write loads into net.load.
    """
    n_load = len(net.load)
    p_load = x_vec[:n_load]
    q_load = x_vec[n_load:2*n_load]
    for i, idx in enumerate(net.load.index):
        net.load.at[idx, "p_mw"] = float(p_load[i])
        net.load.at[idx, "q_mvar"] = float(q_load[i])


def run_acopf_for_sample(base_net, x_vec: np.ndarray):
    """
    For a given load scenario x_vec, run AC-OPF and return:
    Pg*, Vm*, Va*, and a copy of the solved net.
    """
    net = copy.deepcopy(base_net)
    set_loads_from_X(net, x_vec)

    try:
        pp.runopp(net)
        if not net["OPF_converged"]:
            raise RuntimeError("OPF did not converge")
    except Exception as e:
        print("[AC-OPF] Did not converge for this sample:", e)
        return None, None, None, None

    Pg_star = net.res_gen["p_mw"].values.copy()
    Vm_star = net.res_bus["vm_pu"].values.copy()
    Va_star = net.res_bus["va_degree"].values.copy()
    return Pg_star, Vm_star, Va_star, net


def run_pf_from_prediction(base_net,
                           x_vec: np.ndarray,
                           p_gen_pred: np.ndarray,
                           vm_setpoints: np.ndarray | None = None):
    """
    Given predicted Pg (and optional generator Vm setpoints),
    update net, run AC PF, and return solved net or None if PF fails.
    """
    net = copy.deepcopy(base_net)
    set_loads_from_X(net, x_vec)

    # Set generator active powers
    for j, g_idx in enumerate(net.gen.index):
        net.gen.at[g_idx, "p_mw"] = float(p_gen_pred[j])

    # Optional: set generator vm setpoints (A1/A2 case)
    if vm_setpoints is not None:
        for j, g_idx in enumerate(net.gen.index):
            net.gen.at[g_idx, "vm_pu"] = float(vm_setpoints[j])

    try:
        pp.runpp(net, calculate_voltage_angles=True)
    except Exception as e:
        print("[PF] Did not converge for this sample:", e)
        return None

    return net


# -----------------------------------------------------
# 4. Main evaluation
# -----------------------------------------------------
if __name__ == "__main__":
    # ------------------------------
    # 4.1 Load network + AC-OPF dataset
    # ------------------------------
    print("=== Loading network + AC-OPF dataset ===")
    base_net = pn.case14()
    # make sure OPF has a cost (same as A1)
    setup_opf_costs(base_net, linear_cost=10.0)

    data = torch.load("acopf_dataset.pt")
    X_all = data["X"]   # (N, 2*n_load) torch tensor
    Y_all = data["Y"]   # (N, 2*n_gen) torch tensor: [Pg*, Vm*]

    # Held-out test slice (adjust indices as you like)
    X_test = X_all[60:80].clone()
    Y_test = Y_all[60:80].clone()

    X_test_np = X_test.cpu().numpy()
    Y_test_np = Y_test.cpu().numpy()

    n_gen = Y_test_np.shape[1] // 2
    n_in = X_test.shape[1]
    n_out = Y_test.shape[1]

    # ------------------------------
    # 4.2 Load A1 and A2 models
    # ------------------------------
    print("=== Loading A1 model ===")
    a1_model = DirectMap(n_in, n_out)
    a1_model.load_state_dict(torch.load("a1_model.pt"))
    a1_model.eval()

    print("=== Loading A2 model ===")
    a2_model = DirectMap(n_in, n_out)
    a2_model.load_state_dict(torch.load("a2_model.pt"))
    a2_model.eval()

    # ------------------------------
    # 4.3 Load B1 + B2
    # ------------------------------
    print("=== Loading B1 + B2 models ===")
    trained_b1 = torch.load("train_hotstart_b1.pth", weights_only=False)
    trained_b2 = torch.load("trained_b2.pth", weights_only=False)

    # ------------------------------
    # 4.4 Define run_* wrappers ONLY for runtime timing
    # ------------------------------
    def run_a1_pg(X_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            out = a1_model(X_tensor)
        return out[:, :n_gen].cpu().numpy()

    def run_a2_pg(X_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            out = a2_model(X_tensor)
        return out[:, :n_gen].cpu().numpy()

    def run_b2_pg(X_np: np.ndarray) -> np.ndarray:
        out = inference_b2(
            base_net,
            trained_b1,
            trained_b2,
            X_np,
            run_ac_pf=False,   # <- no AC-PF inside timing
            verbose=False,
        )
        return out["p_b2_refined"]    # (N, n_gen)

    # ------------------------------
    # 4.5 Measure runtimes (forward pass only)
    # ------------------------------
    print("\n=== Measuring runtimes (forward pass only) ===")
    runtime_a1 = timing(lambda X: run_a1_pg(X_test), X_test, n_runs=50)
    runtime_a2 = timing(lambda X: run_a2_pg(X_test), X_test, n_runs=50)
    runtime_b2 = timing(lambda X: run_b2_pg(X_test_np), X_test, n_runs=50)

    print(f"A1 runtime per call: {runtime_a1:.6e} s")
    print(f"A2 runtime per call: {runtime_a2:.6e} s")
    print(f"B2 runtime per call: {runtime_b2:.6e} s")

    # ------------------------------
    # 4.6 Run NN predictions (full outputs for PF evaluation)
    # ------------------------------
    with torch.no_grad():
        Y_a1 = a1_model(X_test)      # (N, 2*n_gen)
        Y_a2 = a2_model(X_test)      # (N, 2*n_gen)
    Pg_b2 = run_b2_pg(X_test_np)     # (N, n_gen)

    Pg_a1 = Y_a1[:, :n_gen].cpu().numpy()
    Vm_a1 = Y_a1[:, n_gen:].cpu().numpy()

    Pg_a2 = Y_a2[:, :n_gen].cpu().numpy()
    Vm_a2 = Y_a2[:, n_gen:].cpu().numpy()

    # ------------------------------
    # 4.7 Evaluate per-sample OPF metrics
    # ------------------------------
    print("\n=== Evaluating constraint violations and errors vs AC-OPF ===")

    n_test = X_test_np.shape[0]

    # Accumulators (averaged later)
    metrics = {
        "A1": {
            "V_Pg": 0.0, "V_V": 0.0, "V_line": 0.0,
            "e_Pg": 0.0, "e_V": 0.0, "e_theta": 0.0,
            "J_gap": 0.0,
        },
        "A2": {
            "V_Pg": 0.0, "V_V": 0.0, "V_line": 0.0,
            "e_Pg": 0.0, "e_V": 0.0, "e_theta": 0.0,
            "J_gap": 0.0,
        },
        "B2": {
            "V_Pg": 0.0, "V_V": 0.0, "V_line": 0.0,
            "e_Pg": 0.0, "e_V": 0.0, "e_theta": 0.0,
            "J_gap": 0.0,
        },
    }

    for i in range(n_test):
        x_i = X_test_np[i, :]

        # --- Ground-truth AC-OPF for this sample ---
        Pg_star, Vm_star, Va_star, net_opf = run_acopf_for_sample(base_net, x_i)
        if Pg_star is None:
            print(f"Skipping sample {i}: OPF did not converge.")
            continue
        J_star = opf_objective(net_opf, Pg_star)

        # ============ A1 ============
        net_a1 = run_pf_from_prediction(
            base_net,
            x_i,
            p_gen_pred=Pg_a1[i, :],
            vm_setpoints=Vm_a1[i, :],
        )

        if net_a1 is not None:
            metrics["A1"]["V_Pg"]   += compute_gen_violation(net_a1)
            metrics["A1"]["V_V"]    += compute_voltage_violation(net_a1)
            metrics["A1"]["V_line"] += compute_line_violation(net_a1)

            Pg_a1_pf = net_a1.res_gen["p_mw"].values
            Vm_a1_pf = net_a1.res_bus["vm_pu"].values
            Va_a1_pf = net_a1.res_bus["va_degree"].values

            metrics["A1"]["e_Pg"]    += l2_norm(Pg_a1_pf, Pg_star)
            metrics["A1"]["e_V"]     += l2_norm(Vm_a1_pf, Vm_star)
            metrics["A1"]["e_theta"] += l2_norm(Va_a1_pf, Va_star)

            J_model = opf_objective(net_a1, Pg_a1_pf)
            metrics["A1"]["J_gap"] += (J_model - J_star)

        # ============ A2 ============
        net_a2 = run_pf_from_prediction(
            base_net,
            x_i,
            p_gen_pred=Pg_a2[i, :],
            vm_setpoints=Vm_a2[i, :],
        )

        if net_a2 is not None:
            metrics["A2"]["V_Pg"]   += compute_gen_violation(net_a2)
            metrics["A2"]["V_V"]    += compute_voltage_violation(net_a2)
            metrics["A2"]["V_line"] += compute_line_violation(net_a2)

            Pg_a2_pf = net_a2.res_gen["p_mw"].values
            Vm_a2_pf = net_a2.res_bus["vm_pu"].values
            Va_a2_pf = net_a2.res_bus["va_degree"].values

            metrics["A2"]["e_Pg"]    += l2_norm(Pg_a2_pf, Pg_star)
            metrics["A2"]["e_V"]     += l2_norm(Vm_a2_pf, Vm_star)
            metrics["A2"]["e_theta"] += l2_norm(Va_a2_pf, Va_star)

            J_model = opf_objective(net_a2, Pg_a2_pf)
            metrics["A2"]["J_gap"] += (J_model - J_star)

        # ============ B2 ============
        net_b2 = run_pf_from_prediction(
            base_net,
            x_i,
            p_gen_pred=Pg_b2[i, :],
            vm_setpoints=None,   # use generator vm_pu already in the case
        )

        if net_b2 is not None:
            metrics["B2"]["V_Pg"]   += compute_gen_violation(net_b2)
            metrics["B2"]["V_V"]    += compute_voltage_violation(net_b2)
            metrics["B2"]["V_line"] += compute_line_violation(net_b2)

            Pg_b2_pf = net_b2.res_gen["p_mw"].values
            Vm_b2_pf = net_b2.res_bus["vm_pu"].values
            Va_b2_pf = net_b2.res_bus["va_degree"].values

            metrics["B2"]["e_Pg"]    += l2_norm(Pg_b2_pf, Pg_star)
            metrics["B2"]["e_V"]     += l2_norm(Vm_b2_pf, Vm_star)
            metrics["B2"]["e_theta"] += l2_norm(Va_b2_pf, Va_star)

            J_model = opf_objective(net_b2, Pg_b2_pf)
            metrics["B2"]["J_gap"] += (J_model - J_star)

    # Normalize by number of test samples
    for key in metrics.keys():
        for met in metrics[key]:
            metrics[key][met] /= n_test

    # ------------------------------
    # 4.8 Print summary
    # ------------------------------
    print("\n=== Summary: Runtime & Constraint-based Accuracy (averaged over test set) ===")
    print("{:<5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12}".format(
        "Model", "V_Pg", "V_V", "V_line", "e_Pg", "e_V", "e_theta", "Runtime [s]"
    ))
    print("-" * 80)
    for name, rt in zip(["A1", "A2", "B2"], [runtime_a1, runtime_a2, runtime_b2]):
        print("{:<5} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>12.3e}".format(
            name,
            metrics[name]["V_Pg"],
            metrics[name]["V_V"],
            metrics[name]["V_line"],
            metrics[name]["e_Pg"],
            metrics[name]["e_V"],
            metrics[name]["e_theta"],
            rt,
        ))
 # -----------------------------------------------------
# 5. Generate comparison figures
# -----------------------------------------------------
models = ["A1", "A2", "B2"]
x = np.arange(len(models))
width = 0.25

# Collect averaged metrics
V_Pg   = [metrics[m]["V_Pg"]   for m in models] 
V_V    = [metrics[m]["V_V"]    for m in models]
V_line = [metrics[m]["V_line"] for m in models]

e_Pg    = [metrics[m]["e_Pg"]    for m in models]
e_V     = [metrics[m]["e_V"]     for m in models]
e_theta = [metrics[m]["e_theta"] for m in models]

J_gap   = [metrics[m]["J_gap"]  for m in models]
runtime = [runtime_a1, runtime_a2, runtime_b2]

# =====================================================
# Figure 1: Constraint violations + solution errors
# =====================================================
# Match Figure 2 aspect ratio (7.5 × 3.2)
fig1, axes1 = plt.subplots(1, 2, figsize=(7.5, 3.2))
fig1.subplots_adjust(wspace=0.35, bottom=0.25, top=0.88)

TITLE_FONTSIZE  = 12
LABEL_FONTSIZE  = 10
TICK_FONTSIZE   = 9
LEGEND_FONTSIZE = 8

# ===========================
# (a) Constraint violations
# ===========================
ax = axes1[0]
ax.bar(x - width, V_Pg,   width, label="Gen limits [MW]")
ax.bar(x,         V_V,    width, label="Voltage [p.u.]")
ax.bar(x + width, V_line, width, label="Lines [% overlimit]")

ax.set_title("Constraint violations (avg)", fontsize=TITLE_FONTSIZE, pad=6)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=TICK_FONTSIZE)
ax.set_ylabel("Sum of violations per sample", fontsize=LABEL_FONTSIZE)
ax.set_yscale("log")

# Legend INSIDE panel (top-left)
ax.legend(loc="upper left",
          fontsize=LEGEND_FONTSIZE,
          frameon=True)


# ===========================
# (b) Solution error vs AC-OPF
# ===========================
ax = axes1[1]
ax.bar(x - width, e_Pg,    width, label="Pg error [MW]")
ax.bar(x,         e_V,     width, label="Voltage error [p.u.]")
ax.bar(x + width, e_theta, width, label="Angle error [deg]")

ax.set_title("Solution error vs AC-OPF", fontsize=TITLE_FONTSIZE, pad=6)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=TICK_FONTSIZE)
ax.set_ylabel(r"L2 error $||x - x^*||_2$", fontsize=LABEL_FONTSIZE)
ax.set_yscale("log")

# Legend INSIDE panel (top-left)
ax.legend(loc="upper left",
          fontsize=LEGEND_FONTSIZE,
          frameon=True)

# Save
fig1.tight_layout()
fig1.savefig("comparison_constraints_errors.png",
             dpi=300, bbox_inches="tight")


# =====================================================
# Figure 2: Optimality gap + runtime
# =====================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(7.5, 3.2))
fig2.subplots_adjust(wspace=0.35, bottom=0.30)

# --- (c) Optimality gap ---
ax = axes2[0]
ax.bar(x, J_gap, width)

ax.set_title("Optimality gap (avg)")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel(r"$J_{\mathrm{model}} - J_{\mathrm{AC-OPF}}$")  # FIXED

ax.axhline(0, color="black", linewidth=0.8)
ax.set_yscale("symlog", linthresh=1e-3)


# --- (d) Runtime ---
ax = axes2[1]
ax.bar(x, runtime, width)

ax.set_title("Runtime: forward pass only")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Seconds per batch")
ax.set_yscale("log")

fig2.tight_layout()
fig2.savefig("comparison_gap_runtime.png",
             dpi=300, bbox_inches="tight")

plt.show()
