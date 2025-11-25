# model_b1.py
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandapower as pp
import pandapower.networks as pn
import numpy as np

# ------------------------
# 1) Helper: Build Ybus for physics layer
# ------------------------
#Extracts the bus admittance matrix (Ybus) from Pandapower.
#Ybus is essential if you want to build physics-based layers later (e.g., AC power flow constraints).

def build_ybus_numpy(net):
    # ensure power flow is run first to populate Ybus
    pp.runpp(net)
    Ybus_sparse = net["_ppc"]["internal"]["Ybus"]  # scipy.sparse matrix
    return Ybus_sparse.toarray()  # convert to dense NumPy array

# ------------------------
# 2) Neural Networks (Modalities)
# ------------------------

class DirectMap(nn.Module):
    #a1 network (baseline direct mapping without residual correction)
    """Direct mapping MLP: pload/qload -> generator p/vm"""
    def __init__(self, n_load, n_gen, hidden_sizes=[64,64]):
        super().__init__()
        layers = []
        last_size = n_load
        for h in hidden_sizes:
            layers.append(nn.Linear(last_size,h))
            layers.append(nn.ReLU())
            last_size = h
        layers.append(nn.Linear(last_size,n_gen))
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)

class HotStartMap(nn.Module):
    """Hot-start mapping: pload/qload + prior feasible p0, output is the residual to correct hot-start"""
    def __init__(self, n_load, n_state, n_gen, hidden_sizes=[128,128]):
        super().__init__()
        self.n_load = n_load
        self.n_state = n_state
        self.n_gen = n_gen
        layers = []
        last_size = n_load + n_state
        for h in hidden_sizes:
            layers.append(nn.Linear(last_size,h))
            layers.append(nn.ReLU())
            last_size = h
        layers.append(nn.Linear(last_size,n_gen))
        self.model = nn.Sequential(*layers)

    def forward(self, x_load, x_hot):
        # x_hot is p0 (prior feasible dispatch) or other state
        x = torch.cat([x_load, x_hot], dim=-1)
        return self.model(x)

# ------------------------
# 3) Feasibility Layers
# ------------------------

def postprocess_pf(net, p_gen_pred):
    """
    Legacy post-process: write predicted p_gen into net.gen and run pp.runpp to produce final feasible result.
    Note: this is non-differentiable (Pandapower is a black box).
    """
    for i, g in enumerate(net.gen.index):
        net.gen.at[g,'p_mw'] = float(p_gen_pred[i])
    pp.runpp(net)
    p_gen = net.res_gen['p_mw'].values
    vm = net.res_bus['vm_pu'].values
    return p_gen, vm

class DCFeasibilityLayer(nn.Module):
    """
    Differentiable DC feasibility restoration layer (PyTorch).
    - Enforces total active power balance (sum p_gen = total_load)
    - Enforces generator box limits via projection (hard clip)
    - Implemented as unrolled small iterative primal-dual updates (differentiable)
    #Uses unrolled primal-dual iterations for constraint satisfaction like in https://arxiv.org/pdf/1909.10461
    allows end to end training of b1 residuals + DC feasibility
    """
    def __init__(self, n_gen, p_min, p_max, n_iter=12, lr_primal=0.5, lr_dual=0.8):
        super().__init__()
        self.n_gen = n_gen
        self.n_iter = n_iter
        self.lr_primal = lr_primal
        self.lr_dual = lr_dual
        # store as buffers so they move with the module's device
        self.register_buffer('p_min', torch.tensor(p_min, dtype=torch.float32))
        self.register_buffer('p_max', torch.tensor(p_max, dtype=torch.float32))

    def forward(self, p_pred, total_load):
        """
        p_pred: (B, n_gen) predicted dispatch (p0 + residual)
        total_load: (B,) or (B,1) total active load per sample
        returns: p_corrected (B, n_gen)
        """
        if p_pred.dim() == 1:
            p = p_pred.unsqueeze(0)
        else:
            p = p_pred.clone()
        B = p.shape[0]
        device = p.device

        lam = torch.zeros((B,1), dtype=p.dtype, device=device)  # dual for sum constraint

        # unrolled primal-dual iterations
        for _ in range(self.n_iter):
            grad = (p - p_pred)  # grad of 0.5||p-p_pred||^2
            grad = grad + lam  # lam has shape (B,1) broadcasted
            p = p - self.lr_primal * grad
            # project box constraints
            p = torch.max(torch.min(p, self.p_max.unsqueeze(0)), self.p_min.unsqueeze(0))
            # residual of equality
            res = torch.sum(p, dim=1, keepdim=True) - total_load.view(-1,1)
            lam = lam + self.lr_dual * res
            lam = torch.clamp(lam, -1e6, 1e6)

        # final small correction: if equality not exact, distribute residual among non-saturated gens
        res = torch.sum(p, dim=1, keepdim=True) - total_load.view(-1,1)  # (B,1)
        slack_mask = ((p > (self.p_min.unsqueeze(0) + 1e-8)) & (p < (self.p_max.unsqueeze(0) - 1e-8))).float()
        slack_sum = slack_mask.sum(dim=1, keepdim=True)
        slack_sum = torch.where(slack_sum == 0, torch.ones_like(slack_sum), slack_sum)
        p = p - (res / slack_sum) * slack_mask

        return p.squeeze(0) if p_pred.dim() == 1 else p

# ------------------------
# 4) Dataset generation (randomized IEEE-14 loads)
# ------------------------
def generate_dummy_dataset(net_base, n_samples=50, max_tries=20):
    """
    Safe dataset generator:
    - copies the net each iteration (never corrupts base)
    - retries random load samples until PF converges
    - guarantees valid OPF outputs
    """
    n_load = len(net_base.load)
    n_gen = len(net_base.gen)

    X_list = []
    Y_list = []

    for _ in range(n_samples):
        success = False
        tries = 0

        while not success and tries < max_tries:
            tries += 1

            # Copy network for safe editing
            net = copy.deepcopy(net_base)

            # Conservative perturbation (±10%)
            pload_orig = net_base.load.p_mw.values
            qload_orig = net_base.load.q_mvar.values
            scale = 0.9 + 0.2 * np.random.rand(n_load)

            pload = pload_orig * scale
            qload = qload_orig * scale

            # Apply loads
            for i, idx in enumerate(net.load.index):
                net.load.at[idx, 'p_mw'] = float(pload[i])
                net.load.at[idx, 'q_mvar'] = float(qload[i])

            try:
                pp.runpp(net, max_iteration=30)
                success = True
            except:
                success = False

        if not success:
            print("Warning: skipping sample, no convergence")
            continue

        # Extract generator outputs
        p_gen = net.res_gen.p_mw.values
        vm_gen = net.res_bus.vm_pu.values[net.gen.bus.values]

        X_list.append(np.concatenate([pload, qload]))
        Y_list.append(np.concatenate([p_gen, vm_gen]))

    return (
        torch.tensor(np.array(X_list), dtype=torch.float32),
        torch.tensor(np.array(Y_list), dtype=torch.float32),
    )

# ------------------------
# 5) Training: Direct (A1) and Hot-Start (B1)
# ------------------------

def train_direct(net, n_epochs=20, lr=1e-3):
    """
    Original direct mapping (A1) training: supervised MSE from loads -> p_gen + vm labels
    """
    X_np, Y_np = generate_dummy_dataset(net, n_samples=300)
    
    # Convert to tensors
    X = torch.tensor(X_np, dtype=torch.float32)
    Y = torch.tensor(Y_np, dtype=torch.float32)  # full p_gen + vm, no slicing needed
    
    n_load = X.shape[1]
    n_gen = Y.shape[1]

    model = DirectMap(n_load, n_gen)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = loss_fn(Y_pred, Y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0 or epoch==0:
            print(f"[A1] Epoch {epoch+1}/{n_epochs}  Loss={loss.item():.6f}")
    return model

def create_base_hotstart(net, n_samples):
    """
    Construct base-case hot starts p0 (use network base case PF)
    """
    net_copy = copy.deepcopy(net)
    try:
        pp.runpp(net_copy)
    except Exception:
        pass
    base_p = net_copy.res_gen['p_mw'].values.copy()
    return np.tile(base_p.reshape(1,-1), (n_samples,1))


def train_hotstart_b1(net, n_epochs=60, batch_size=32, lr=1e-3, n_samples=2000, use_gpu=False):

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print("Training device:", device)

    # Load dataset
    X_np, Y_np = generate_dummy_dataset(net, n_samples=n_samples)
    n_gen = len(net.gen)

    # Only use p_gen targets
    Y_pgen_np = Y_np[:, :n_gen]

    N = X_np.shape[0]
    n_load = X_np.shape[1] // 2

    p0 = create_base_hotstart(net, N)

    X = X_np.float().to(device)
    Y = Y_pgen_np.float().to(device)
    p0_tensor = torch.tensor(p0, dtype=torch.float32, device=device)

    # Model + DC layer
    model = HotStartMap(n_load*2, n_gen, n_gen, hidden_sizes=[256,256]).to(device)

    p_min = net.gen['min_p_mw'].values.copy()
    p_max = net.gen['max_p_mw'].values.copy()
    dc_layer = DCFeasibilityLayer(n_gen=n_gen, p_min=p_min, p_max=p_max,
                                  n_iter=12, lr_primal=0.5, lr_dual=0.8).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    perm = np.arange(N)

    for epoch in range(1, n_epochs+1):
        np.random.shuffle(perm)
        epoch_loss = 0.0
        model.train()

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            x_batch = X[idx]
            y_batch = Y[idx, :n_gen]
            p0_batch = p0_tensor[idx]

            optimizer.zero_grad()

            residual = model(x_batch, p0_batch)
            p_hat = p0_batch + residual

            # compute total load per sample
            p_load_batch = x_batch[:, :n_load]
            total_load = torch.sum(p_load_batch, dim=1)

            p_corrected = dc_layer(p_hat, total_load)
            loss = loss_fn(p_corrected, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * p_corrected.shape[0]

        # finalize epoch metrics
        epoch_loss /= N

        if epoch % 5 == 0 or epoch == 1:
            print(f"[B1] Epoch {epoch}/{n_epochs}  loss={epoch_loss:.6e}")

    # SAVE CHECKPOINT FOR B2
    torch.save(model.state_dict(), "train_hotstart_b1.pth")
    print("Saved hot-start checkpoint: train_hotstart_b1.pth")

    trained = {
    'model': model.cpu(),
    'dc_layer': dc_layer.cpu(),
    'p0_template': p0,
    'n_load': n_load,
    'n_gen': n_gen,
    }

    torch.save(trained, "train_hotstart_b1.pth")  # save the full dict
    print("Saved hot-start checkpoint: train_hotstart_b1.pth")



    return trained


# ------------------------
# 6) Inference: B2 warm-start + optional AC PF (Pandapower). DC-corrected generator outputs
# ------------------------
def inference_b1(net, trained, x_load_np, run_ac_pf=True, verbose=True):
    """
    x_load_np: shape (M, 2*n_load) or (2*n_load,)
    returns dict with DC-corrected p and optionally final PF-corrected outputs
    """

    model = trained['model']
    dc_layer = trained['dc_layer']
    p0_template = trained['p0_template']     # numpy array or tensor from training
    n_load = trained['n_load']
    n_gen = trained['n_gen']

    model.eval()

    # Ensure batch dimension
    if x_load_np.ndim == 1:
        x_load_np = x_load_np.reshape(1, -1)

    # Convert load input to tensor
    X = torch.tensor(x_load_np, dtype=torch.float32)

    # --- FIX #1: Convert p0_template to tensor ---
    if isinstance(p0_template, np.ndarray):
        p0 = torch.tensor(p0_template[:X.shape[0]], dtype=torch.float32)
    else:
        # already a tensor (rare)
        p0 = p0_template[:X.shape[0]].clone().float()

    # Move to same device as model
    device = next(model.parameters()).device
    X = X.to(device)
    p0 = p0.to(device)

    # Forward pass
    with torch.no_grad():
        residual = model(X, p0)
        p_hat = p0 + residual
        total_load = torch.sum(X[:, :n_load], dim=1)

        # DC feasibility projection
        p_dc_tensor = dc_layer(p_hat, total_load)
        p_dc = p_dc_tensor.cpu().numpy()     # convert after computation

    out = {'p_dc_corrected': p_dc}

    # ----- Optional AC PF refinement -----
    if run_ac_pf:
        p_pf_list = []
        vm_pf_list = []

        for i in range(p_dc.shape[0]):
            net_local = copy.deepcopy(net)

            # Set loads
            p_load_i = x_load_np[i, :n_load]
            q_load_i = x_load_np[i, n_load:]
            for j, l in enumerate(net_local.load.index):
                net_local.load.at[l, 'p_mw'] = float(p_load_i[j])
                net_local.load.at[l, 'q_mvar'] = float(q_load_i[j])

            # Set generator dispatch to DC-corrected warm start
            for idx_g, g in enumerate(net_local.gen.index):
                net_local.gen.at[g, 'p_mw'] = float(p_dc[i, idx_g])
                try:
                    net_local.gen.at[g, 'vm_pu'] = 1.0
                except Exception:
                    pass

            # Try AC PF
            try:
                pp.runpp(net_local, calculate_voltage_angles=True)
            except Exception as e:
                if verbose:
                    print("PF failed (fallback):", e)
                try:
                    pp.runpp(net_local)
                except Exception as e2:
                    if verbose:
                        print("PF fallback failed:", e2)
                    p_pf_list.append(np.full(n_gen, np.nan))
                    vm_pf_list.append(np.full(n_gen, np.nan))
                    continue

            # Extract PF outputs
            p_pf_list.append(net_local.res_gen['p_mw'].values.copy())

            vm_all = net_local.res_bus['vm_pu'].values.copy()
            gen_buses = net_local.gen['bus'].values
            vm_pf = np.array([vm_all[bus] for bus in gen_buses])
            vm_pf_list.append(vm_pf)

        out['p_pf'] = np.stack(p_pf_list, axis=0)
        out['vm_pf'] = np.stack(vm_pf_list, axis=0)

    return out


# ------------------------
# 7) Main execution (demo)
# ------------------------
if __name__ == "__main__":
    net = pn.case14()
    Ybus = build_ybus_numpy(net)  # optional AC layer prep

    # Train B1 hot-start residual model
    print("Training B1 Hot-Start model with DC feasibility layer...")
    trained = train_hotstart_b1(net, n_epochs=50, batch_size=64, lr=5e-3, n_samples=2000, use_gpu=False)

    # Demo inference on a few samples
    X_test_np, Y_test_np = generate_dummy_dataset(net, n_samples=4)
    out = inference_b1(net, trained, X_test_np, run_ac_pf=True, verbose=True)
    print("DC-corrected p (per sample):")
    print(out['p_dc_corrected'])
    if 'p_pf' in out:
        print("Final PF-corrected p (per sample):")
        print(out['p_pf'])
