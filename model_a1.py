# ieee14_neuromancer_demo.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandapower as pp
import pandapower.networks as pn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def setup_opf_costs(net, linear_cost=10.0):
    """
    Attach linear generation costs so that pp.runopp() has an objective.
    Matches the AC OPF 14-bus script: cost = linear_cost * Pg.
    """
    # Clear any existing cost data
    if len(net.poly_cost) > 0:
        net.poly_cost.drop(net.poly_cost.index, inplace=True)

    # Add linear cost for each generator
    for gen_idx in net.gen.index:
        pp.create_poly_cost(net, gen_idx, 'gen', cp1_eur_per_mw=linear_cost)


# ------------------------
# 1) Helper: Build Ybus for physics layer
# ------------------------
def build_ybus_numpy(net):
    # ensure power flow is run first to populate Ybus
    pp.runpp(net)
    # access Ybus from net["_ppc"] using pandapower’s internal API
    Ybus_sparse = net["_ppc"]["internal"]["Ybus"]  # scipy.sparse matrix
    return Ybus_sparse.toarray()  # convert to dense NumPy array

# ------------------------
# 2) Neural Networks (Modalities)
# ------------------------

class DirectMap(nn.Module):
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
    """Hot-start mapping: pload/qload + previous solution -> generator p/vm"""
    def __init__(self, n_load, n_state, n_gen, hidden_sizes=[64,64]):
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
        x = torch.cat([x_load, x_hot], dim=-1)
        return self.model(x)

# ------------------------
# 3) Feasibility Layers
# ------------------------

# 3a) Post-processing: run Pandapower PF to enforce constraints
def postprocess_pf(net, p_gen_pred, vm_pred):
    for i, g in enumerate(net.gen.index):
        net.gen.at[g,'p_mw'] = float(p_gen_pred[i])
        net.gen.at[g,'vm_pu'] = float(vm_pred[i])
    pp.runpp(net)
    # returns feasible solution
    p_gen = net.res_gen['p_mw'].values
    vm = net.res_gen['vm_pu'].values
    return p_gen, vm

# 3b) DC3 feasibility layer skeleton (differentiable)
class DC3Layer(nn.Module):
    """Differentiable feasibility correction (skeleton)"""
    def __init__(self, Ybus_np):
        super().__init__()
        self.Ybus = torch.tensor(Ybus_np, dtype=torch.complex64)

    def forward(self, v_mag, theta, p_gen, q_gen, p_load, q_load):
        # Placeholder: implement constraint completion and correction here
        # Returns corrected v_mag, theta, p_gen, q_gen
        # For demo, just return input
        return v_mag, theta, p_gen, q_gen

# ------------------------
# 4) Example Dataset (randomized IEEE-14 loads)
# ------------------------

#AC OPF 
def generate_dummy_dataset_opf(net, n_samples=50, linear_cost=10.0):
    setup_opf_costs(net, linear_cost=linear_cost)

    n_load = len(net.load)
    X_load = []
    Y_gen  = []

    for _ in range(n_samples):
        pload = net.load['p_mw'].values * (0.8 + 0.4*np.random.rand(n_load))
        qload = net.load['q_mvar'].values * (0.8 + 0.4*np.random.rand(n_load))
        X_load.append(np.concatenate([pload, qload]))

        # update loads
        for i, l in enumerate(net.load.index):
            net.load.at[l,'p_mw'] = pload[i]
            net.load.at[l,'q_mvar'] = qload[i]

        try:
            pp.runopp(net)
            if not net["OPF_converged"]:
                continue
        except:
            continue

        p_gen = net.res_gen['p_mw'].values
        vm    = net.res_gen['vm_pu'].values
        Y_gen.append(np.concatenate([p_gen, vm]))

    X_array = np.array(X_load, dtype=np.float32)
    Y_array = np.array(Y_gen,  dtype=np.float32)

    return torch.from_numpy(X_array), torch.from_numpy(Y_array)


# ------------------------
# 5) Training Loop
# ------------------------
def train_direct(model, X_tensor, Y_tensor, n_epochs=10):
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        avg = epoch_loss / len(dataset)
        print(f"Epoch {epoch}/{n_epochs}, Loss={avg:.5e}")

    return model


# ------------------------
# 6) Main Execution
# ------------------------
if __name__ == "__main__":
    # Load IEEE-14
    pp_net = pn.case14()

    # AC-OPF dataset
    X_tensor, Y_tensor = generate_dummy_dataset_opf(pp_net, n_samples=50)

    input_dim  = X_tensor.shape[1]
    output_dim = Y_tensor.shape[1]

    # Define A1 model
    model_direct = DirectMap(n_load=input_dim, n_gen=output_dim, hidden_sizes=[64,64])

    # Train
    model_direct = train_direct(model_direct, X_tensor, Y_tensor, n_epochs=10)

    # Save
    torch.save(model_direct.state_dict(), "a1_model.pt")

    # Evaluate a sample
    with torch.no_grad():
        y_pred = model_direct(X_tensor)

    n_gen = len(pp_net.gen)   # number of actual physical generators

    print("\nAC-OPF truth for sample 0:")
    print("Pg* =", Y_tensor[0,:n_gen].numpy())
    print("Vm* =", Y_tensor[0,n_gen:].numpy())

    print("\nA1 prediction for sample 0:")
    print("Pg^ =", y_pred[0,:n_gen].numpy())
    print("Vm^ =", y_pred[0,n_gen:].numpy())
