# model_b2.py
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandapower as pp

from model_b1 import inference_b1, HotStartMap, DCFeasibilityLayer


# ---------------------------------------------------------
# 1. B2 Residual Refinement Model
# ---------------------------------------------------------
class B2Refine(nn.Module):
    """
    B2 refinement network:
    Takes p_dc (from B1) and optionally x_load, outputs residual r2.
    """
    def __init__(self, n_load, n_gen, hidden_sizes=[256,256], use_load_inputs=True):
        super().__init__()
        self.use_load_inputs = use_load_inputs
        input_dim = n_gen + (2*n_load if use_load_inputs else 0)

        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, n_gen))
        self.model = nn.Sequential(*layers)

    def forward(self, p_dc, x_load=None):
        if self.use_load_inputs:
            x = torch.cat([x_load, p_dc], dim=1)
        else:
            x = p_dc
        return self.model(x)


# ---------------------------------------------------------
# 2. B2 Training (using AC-OPF dataset from A1)
# ---------------------------------------------------------
def train_b2(net,
             trained_b1,
             X_np_a1,
             Y_np_a1,
             n_epochs=40,
             batch_size=32,
             lr=1e-3,
             use_gpu=False,
             use_load_inputs=True):

    """
    X_np_a1: (N, 2*n_load) AC-load dataset from A1
    Y_np_a1: (N, n_gen + n_vm) AC-OPF labels from A1
    """

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print("Training B2 on AC-OPF labels…  Device:", device)

    n_load = trained_b1["n_load"]
    n_gen = trained_b1["n_gen"]

    # AC OPF targets (only Pgen)
    Y_pgen_np = Y_np_a1[:, :n_gen]

    # Convert to torch
    X = torch.tensor(X_np_a1, dtype=torch.float32, device=device)
    Y = torch.tensor(Y_pgen_np, dtype=torch.float32, device=device)

    # -----------------------------------------
    # Step 1: Compute B1 DC-feasible p_dc for all samples
    # -----------------------------------------
    print("Running B1 to generate p_dc hot-starts…")

    p_dc_batches = []
    N = X.shape[0]

    for i in range(0, N, batch_size):
        xb = X_np_a1[i:i+batch_size]
        out = inference_b1(net, trained_b1, xb, run_ac_pf=False, verbose=False)
        p_dc_batches.append(out["p_dc_corrected"])

    p_dc_all = np.vstack(p_dc_batches)
    P_dc = torch.tensor(p_dc_all, dtype=torch.float32, device=device)

    # -----------------------------------------
    # Step 2: Train B2 residual model
    # -----------------------------------------
    model = B2Refine(n_load, n_gen, hidden_sizes=[256,256],
                     use_load_inputs=use_load_inputs).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    perm = np.arange(N)

    for epoch in range(1, n_epochs+1):
        np.random.shuffle(perm)
        total_loss = 0.0

        model.train()
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]

            x_batch = X[idx]
            y_batch = Y[idx]
            p_dc_batch = P_dc[idx]

            optimizer.zero_grad()

            r2 = model(p_dc_batch, x_batch if use_load_inputs else None)
            p_ref = p_dc_batch + r2

            loss = loss_fn(p_ref, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)

        avg_loss = total_loss / N
        if epoch % 5 == 0 or epoch == 1:
            print(f"[B2] Epoch {epoch}/{n_epochs}  Loss={avg_loss:.6e}")

    # Save only CPU version
    trained = {
        "model": model.cpu(),
        "use_load_inputs": use_load_inputs,
        "n_load": n_load,
        "n_gen": n_gen,
    }
    return trained


# ---------------------------------------------------------
# 3. B2 Inference
# ---------------------------------------------------------
def inference_b2(net, trained_b1, trained_b2, x_np, run_ac_pf=True, verbose=True):
    """
    x_np: (M, 2*n_load)
    """
    model = trained_b2["model"]
    model.eval()
    use_load_inputs = trained_b2["use_load_inputs"]

    # Step 1: Get p_dc from B1
    out_b1 = inference_b1(net, trained_b1, x_np, run_ac_pf=False, verbose=verbose)
    p_dc = torch.tensor(out_b1["p_dc_corrected"], dtype=torch.float32)

    # Step 2: B2 refinement
    X_tensor = torch.tensor(x_np, dtype=torch.float32)
    with torch.no_grad():
        r2 = model(p_dc, X_tensor if use_load_inputs else None)
        p_refined = p_dc + r2

    out = {"p_b2_refined": p_refined.numpy()}

    # Step 3: Optional AC PF projection
    if run_ac_pf:
        p_pf_list = []
        vm_list = []
        n_load = trained_b2["n_load"]
        n_gen = trained_b2["n_gen"]

        for i in range(p_refined.shape[0]):
            net_local = copy.deepcopy(net)

            # Set loads
            p_load_i = x_np[i, :n_load]
            q_load_i = x_np[i, n_load:]
            for j, l in enumerate(net_local.load.index):
                net_local.load.at[l, "p_mw"] = float(p_load_i[j])
                net_local.load.at[l, "q_mvar"] = float(q_load_i[j])

            # Set generator dispatch
            for g_idx, g in enumerate(net_local.gen.index):
                net_local.gen.at[g, "p_mw"] = float(p_refined[i, g_idx])

            try:
                pp.runpp(net_local)
                p_pf_list.append(net_local.res_gen.p_mw.values.copy())
                vm_all = net_local.res_bus.vm_pu.values.copy()
                vm_list.append([vm_all[bus] for bus in net_local.gen.bus.values])
            except:
                if verbose:
                    print("AC PF failed on sample", i)
                p_pf_list.append(np.full(n_gen, np.nan))
                vm_list.append(np.full(n_gen, np.nan))

        out["p_pf"] = np.vstack(p_pf_list)
        out["vm_pf"] = np.vstack(vm_list)

    return out


# ---------------------------------------------------------
# 4. Standalone execution
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Running B2 training...")

    net = pp.networks.case14()

    # Load B1
    torch.serialization.add_safe_globals([HotStartMap, DCFeasibilityLayer])
    trained_b1 = torch.load("train_hotstart_b1.pth", weights_only=False)

    # Load A1 dataset
    data = torch.load("acopf_dataset.pt")
    X_np_a1 = data["X"].numpy()
    Y_np_a1 = data["Y"].numpy()


    trained_b2 = train_b2(net, trained_b1, X_np_a1, Y_np_a1)
    torch.save(trained_b2, "trained_b2.pth")

    print("B2 training complete.")

