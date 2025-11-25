# model_b2.py
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandapower as pp
import numpy as np
from model_b1 import generate_dummy_dataset, inference_b1, HotStartMap, DCFeasibilityLayer


# ------------------------
# 1) B2 Residual/Refinement Network
# ------------------------
class B2Refine(nn.Module):
    """
    Optional refinement model for B2.
    Takes DC-corrected p from B1 (p_dc_corrected) and optionally load inputs,
    outputs residual correction r2.
    """
    def __init__(self, n_load, n_gen, hidden_sizes=[128,128], use_load_inputs=True):
        super().__init__()
        self.use_load_inputs = use_load_inputs
        input_size = n_gen + (2*n_load if use_load_inputs else 0)
        layers = []
        last_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(last_size, h))
            layers.append(nn.ReLU())
            last_size = h
        layers.append(nn.Linear(last_size, n_gen))  # residual output
        self.model = nn.Sequential(*layers)

    def forward(self, p_dc, x_load=None):
        if self.use_load_inputs:
            if x_load is None:
                raise ValueError("x_load required if use_load_inputs=True")
            x = torch.cat([x_load, p_dc], dim=-1)
        else:
            x = p_dc
        return self.model(x)  # residual correction r2

# ------------------------
# 2) B2 Training (raw units)
# ------------------------
def train_b2(net, trained_b1, n_epochs=30, batch_size=32, lr=1e-3, use_gpu=False, use_load_inputs=True):
    """
    Trains B2 model to refine DC-corrected outputs from B1 (raw units, no normalization).
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print("Training device:", device)

    # Generate dataset
    X_np, Y_np = generate_dummy_dataset(net, n_samples=2000)  # reuse from B1
    n_load = trained_b1['n_load']
    n_gen = trained_b1['n_gen']

    # Run B1 inference to get p_dc_corrected (hot start)
    p_dc_list = []
    for i in range(0, X_np.shape[0], batch_size):
        batch = X_np[i:i+batch_size]
        out = inference_b1(net, trained_b1, batch, run_ac_pf=False, verbose=False)
        p_dc_list.append(out['p_dc_corrected'])
    p_dc_all = np.vstack(p_dc_list)

    # Convert to tensors
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    Y = torch.tensor(Y_np[:, :n_gen], dtype=torch.float32, device=device)
    P_dc = torch.tensor(p_dc_all, dtype=torch.float32, device=device)

    # Initialize B2 model
    model = B2Refine(n_load, n_gen, hidden_sizes=[256,256], use_load_inputs=use_load_inputs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    N = X.shape[0]
    perm = np.arange(N)

    for epoch in range(1, n_epochs+1):
        np.random.shuffle(perm)
        epoch_loss = 0.0
        model.train()
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            x_batch = X[idx]
            y_batch = Y[idx]
            p_dc_batch = P_dc[idx]

            optimizer.zero_grad()
            r2 = model(p_dc_batch, x_batch if use_load_inputs else None)
            p_refined = p_dc_batch + r2

            # Constraint penalty (optional)
            total_load = torch.sum(x_batch[:, :n_load], dim=1, keepdim=True)
            load_penalty = ((torch.sum(p_refined, dim=1, keepdim=True) - total_load)**2).mean()
            alpha = 0.1
            loss = loss_fn(p_refined, y_batch) + alpha * load_penalty

            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * p_refined.shape[0]

        epoch_loss /= N
        if epoch % 5 == 0 or epoch == 1:
            print(f"[B2] Epoch {epoch}/{n_epochs} avg_loss={epoch_loss:.6e}")

    trained = {'model': model.cpu(), 'use_load_inputs': use_load_inputs}
    return trained


# ------------------------
# 3) B2 Inference (raw units)
# ------------------------
def inference_b2_refine(net, trained_b1, trained_b2, x_load_np, run_ac_pf=True, verbose=True):
    """
    Full B2 inference: run B1 -> DCFeasibilityLayer -> optional B2 refinement -> optional AC PF
    """
    # B1 DC-corrected output
    out_b1 = inference_b1(net, trained_b1, x_load_np, run_ac_pf=False, verbose=verbose)
    p_dc = torch.tensor(out_b1['p_dc_corrected'], dtype=torch.float32)

    # B2 refinement
    model = trained_b2['model']
    model.eval()
    use_load_inputs = trained_b2['use_load_inputs']
    X_tensor = torch.tensor(x_load_np, dtype=torch.float32)
    with torch.no_grad():
        r2 = model(p_dc, X_tensor if use_load_inputs else None)
        p_refined = p_dc + r2
    out = {'p_b2_refined': p_refined.numpy()}

    # Optional AC PF correction (same as before)
    if run_ac_pf:
        p_pf_list, vm_pf_list = [], []
        n_load = trained_b1['n_load']
        n_gen = trained_b1['n_gen']
        for i in range(p_refined.shape[0]):
            net_local = copy.deepcopy(net)
            # set loads
            p_load_i = x_load_np[i, :n_load]
            q_load_i = x_load_np[i, n_load:]
            for j, l in enumerate(net_local.load.index):
                net_local.load.at[l, 'p_mw'] = float(p_load_i[j])
                net_local.load.at[l, 'q_mvar'] = float(q_load_i[j])
            # set generator dispatch
            for idx_g, g in enumerate(net_local.gen.index):
                net_local.gen.at[g, 'p_mw'] = float(p_refined[i, idx_g])
                try:
                    net_local.gen.at[g, 'vm_pu'] = 1.0
                except: pass
            try:
                pp.runpp(net_local, calculate_voltage_angles=True)
            except Exception as e:
                if verbose: print("PF failed:", e)
                p_pf_list.append(np.full(n_gen, np.nan))
                vm_pf_list.append(np.full(n_gen, np.nan))
                continue
            p_pf = net_local.res_gen['p_mw'].values.copy()
            vm_all = net_local.res_bus['vm_pu'].values.copy()
            gen_buses = net_local.gen['bus'].values
            vm_pf = np.array([vm_all[bus] for bus in gen_buses])
            p_pf_list.append(p_pf)
            vm_pf_list.append(vm_pf)
        out['p_pf'] = np.stack(p_pf_list, axis=0)
        out['vm_pf'] = np.stack(vm_pf_list, axis=0)

    return out


# ------------------------
# 4) Example: Run B2 training when executing file directly
# ------------------------
if __name__ == "__main__":
    print("Running B2 training...")

    # Load your network
    net = pp.networks.case14()  

    # Load trained B1
    torch.serialization.add_safe_globals([HotStartMap, DCFeasibilityLayer])
    trained_b1 = torch.load("train_hotstart_b1.pth", weights_only=False)
    

    # Train B2
    trained_b2 = train_b2(net, trained_b1, n_epochs=30, batch_size=32)
    
    # Save B2
    torch.save(trained_b2, "trained_b2.pth")

    print("B2 training complete.")


