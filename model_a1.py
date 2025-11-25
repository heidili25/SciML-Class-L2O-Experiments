# model_a1.py
#A1 is a direct supervised surrogate that maps load (P,Q) to the AC-OPF optimal generation (Pg*, Vm*).
#We generate training data by randomizing the loads, running AC-OPF with linear cost 10×Pg, and collecting the optimal generator dispatch from pp.runopp.
#The network learns a regression model against these OPF outputs.
#There is no physics layer or feasibility module inside A1 — it purely regresses to OPF solutions.

"""
A1: Direct Supervised Surrogate for AC-OPF
------------------------------------------
A1 learns a neural surrogate that maps load (P_load, Q_load)
to the optimal AC-OPF generator outputs (Pg*, Vm*).

Pipeline:
1. Generate randomized load scenarios.
2. Solve AC-OPF with linear cost 10 × Pg per generator.
3. Store OPF-optimal Pg*, Vm*.
4. Train DirectMap MLP: load → (Pg*, Vm*), MSE loss only.
5. Save:
    - Dataset (acopf_dataset.pt)
    - Model weights (a1_model.pt)
6. Print sample truth and prediction for inspection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# (0) Cost function for AC-OPF
# ============================================================

def setup_opf_costs(net, linear_cost=10.0):
    """
    Attach linear generation costs so that pp.runopp() has an objective.

    This matches the baseline AC-OPF script used for A0:
        cost = 10 × Pg  [€/MW]
    """
    if len(net.poly_cost) > 0:
        net.poly_cost.drop(net.poly_cost.index, inplace=True)

    for gen_idx in net.gen.index:
        pp.create_poly_cost(net, gen_idx, 'gen',
                            cp1_eur_per_mw=linear_cost)


# ============================================================
# (1) A1 network: Direct mapping
# ============================================================

class DirectMap(nn.Module):
    """
    Direct mapping MLP:
        [P_load, Q_load] → [P_gen*, Vm_gen*]

    No physics constraints, no feasibility layer.
    Pure supervised regression to OPF labels.
    """
    def __init__(self, n_load, n_out, hidden=[64, 64]):
        super().__init__()
        layers = []
        last = n_load
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# (2) Dataset generator: Random loads → AC-OPF → (Pg*, Vm*)
# ============================================================

def generate_dummy_dataset_opf(net, n_samples=80):
    """
    Randomized load sampling for IEEE-14 + AC-OPF solver.

    Returns:
        X (torch.Tensor): shape [N, 2*n_load]
        Y (torch.Tensor): shape [N, n_gen + n_gen] = [Pg*, Vm*]
    """
    setup_opf_costs(net)

    X_list, Y_list = [], []
    n_load = len(net.load)

    for s in range(n_samples):

        # (a) Randomly perturb loads
        pload = net.load.p_mw.values * (0.8 + 0.4*np.random.rand(n_load))
        qload = net.load.q_mvar.values * (0.8 + 0.4*np.random.rand(n_load))
        X_list.append(np.concatenate([pload, qload]))

        # Write loads to network
        for i, idx in enumerate(net.load.index):
            net.load.at[idx, "p_mw"] = pload[i]
            net.load.at[idx, "q_mvar"] = qload[i]

        # (b) Solve AC-OPF
        try:
            pp.runopp(net)
            if not net["OPF_converged"]:
                print(f"[WARN] OPF didn't converge on sample {s}")
                continue
        except Exception as e:
            print(f"[ERROR] OPF solve failed on sample {s}: {e}")
            continue

        # (c) Extract OPF solution
        Pg = net.res_gen.p_mw.values
        Vm = net.res_bus.vm_pu.values[net.gen.bus.values]   # Vm at generator buses
        Y_list.append(np.concatenate([Pg, Vm]))

    # Convert to tensors
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)

    print(f"\nGenerated {X.shape[0]} OPF samples (requested {n_samples}).")
    return X, Y


# ============================================================
# (3) A1 training loop (KEEP THIS)
# ============================================================

def train_direct(model, X, Y, n_epochs=20, lr=1e-3):
    """
    Supervised training for A1:
        minimize || DirectMap(load) − OPF_labels ||²
    """
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    for ep in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)

        print(f"[A1] Epoch {ep}/{n_epochs} | Loss={epoch_loss/len(dataset):.5e}")

    return model


# ============================================================
# (4) Main Script: Generate dataset, train A1, save artifacts
# ============================================================

if __name__ == "__main__":
    print("\n=== Loading IEEE-14 network ===")
    net = pn.case14()

    print("\n=== Generating AC-OPF dataset for A1 ===")
    X_train, Y_train = generate_dummy_dataset_opf(net, n_samples=80)

    n_input  = X_train.shape[1]    # 2*n_load
    n_output = Y_train.shape[1]    # n_gen + n_gen

    # Create and train model
    print("\n=== Training A1 DirectMap model ===")
    a1_model = DirectMap(n_input, n_output)
    a1_model = train_direct(a1_model, X_train, Y_train, n_epochs=25)

    # ---- SAVE DATASET + MODEL ----
    torch.save({"X": X_train, "Y": Y_train}, "acopf_dataset.pt")
    torch.save(a1_model.state_dict(), "a1_model.pt")

    print("\nSaved:")
    print(" • acopf_dataset.pt (training data for A2/B1/B2)")
    print(" • a1_model.pt      (trained A1 baseline model)")

    # ========================================================
    # Print sample truth & prediction
    # ========================================================
    print("\n=== Example: Sample 0 Truth vs A1 Prediction ===")
    with torch.no_grad():
        y_pred = a1_model(X_train)

    n_gen = len(net.gen)

    print("AC-OPF Truth (sample 0):")
    print(" Pg* =", np.round(Y_train[0, :n_gen].numpy(), 4))
    print(" Vm* =", np.round(Y_train[0, n_gen:].numpy(), 4))

    print("\nA1 Prediction (sample 0):")
    print(" Pg^ =", np.round(y_pred[0, :n_gen].numpy(), 4))
    print(" Vm^ =", np.round(y_pred[0, n_gen:].numpy(), 4))
