# model_a2_l2o_updated.py
import sys
sys.path.append("/Users/heidili/Desktop/Desktop/JHU PhD Year 1/SciML")

import torch
import pandapower as pp
import pandapower.networks as pn
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from model_a1 import DirectMap, build_ybus_numpy, generate_dummy_dataset, postprocess_pf, train_direct

# Neuromancer imports
from neuromancer.loss import PenaltyLoss, Constraint

# ------------------------
# 1) Setup network
# ------------------------
net = pn.case14()
Ybus = build_ybus_numpy(net)
n_load = len(net.load) * 2   # p_load + q_load
n_gen = len(net.gen)         # number of generators

# ------------------------
# 2) Load or train Direct model
# ------------------------
print("Training Direct Modality for A2...")
direct_model = train_direct(net, n_epochs=50, lr=1e-3)

# ------------------------
# 3) Generate new test samples
# ------------------------
X_test, _ = generate_dummy_dataset(net, n_samples=5)
X_test_tensor = torch.tensor(np.stack(X_test), dtype=torch.float32)

# ------------------------
# 4) Define comparators and constraints for PenaltyLoss
# ------------------------
def mse_comparator(y_pred, y_true):
    diff = (y_pred - y_true) ** 2
    if diff.ndim == 1:
        diff = diff.unsqueeze(0)
    loss = diff.mean(dim=1, keepdim=True)
    value = loss.detach()
    violation = torch.zeros_like(loss)
    return loss, value, violation

def gen_limit_comparator(y_pred, limits):
    lower, upper = limits
    lower_violation = torch.relu(lower - y_pred)
    upper_violation = torch.relu(y_pred - upper)
    loss = (lower_violation + upper_violation).mean(dim=1, keepdim=True)
    value = loss.detach()
    violation = loss.clone()
    return loss, value, violation

# Generator and voltage limits for IEEE 14 (example)
pgen_limits = (torch.zeros(n_gen), torch.ones(n_gen) * 10.0)    # MW
v_limits = (torch.ones(n_gen) * 0.9, torch.ones(n_gen) * 1.1)   # per-unit voltage

# ------------------------
# 5) Setup constraints
# ------------------------
constraints = [
    Constraint(
        left=torch.zeros((1, n_gen)),  # dummy placeholder
        right=torch.zeros((1, n_gen)), # must be a tensor
        comparator=lambda y, _: gen_limit_comparator(y, pgen_limits),
        name="p_gen"
    ),
    Constraint(
        left=torch.zeros((1, n_gen)),
        right=torch.zeros((1, n_gen)), # must be a tensor
        comparator=lambda y, _: gen_limit_comparator(y, v_limits),
        name="vm"
    ),
]


# No explicit objectives
objectives = []

loss_fn = PenaltyLoss(objectives, constraints)

# ------------------------
# 6) Predict and compute L2O loss
# ------------------------
Y_pred = direct_model(X_test_tensor)

# Build input_dict for Neuromancer
input_dict = {"p_gen": Y_pred[:, :n_gen], "vm": Y_pred[:, n_gen:]}

loss = loss_fn(input_dict)
print("L2O surrogate PenaltyLoss:", loss.items())

# ------------------------
# 7) Post-process with Pandapower PF
# ------------------------
pred_vs_pf = {"p_gen_pred": [], "p_gen_pf": [], "vm_pred": [], "vm_pf": []}

print("\nFeasible outputs after PF validation:")
for i in range(len(X_test)):
    pload = X_test[i, :len(net.load)]
    qload = X_test[i, len(net.load):]
    p_vm_pred = Y_pred[i].detach().numpy()
    p_gen_pred = p_vm_pred[:len(net.gen)]
    vm_pred = p_vm_pred[len(net.gen):]

    # update loads
    for j, l in enumerate(net.load.index):
        net.load.at[l, 'p_mw'] = float(pload[j])
        net.load.at[l, 'q_mvar'] = float(qload[j])

    # Run PF
    pp.runpp(net)
    p_gen_pf = net.res_gen.p_mw.values
    vm_pf = net.res_bus.vm_pu.values[:len(net.gen)]

    # store for plotting
    pred_vs_pf["p_gen_pred"].append(p_gen_pred)
    pred_vs_pf["p_gen_pf"].append(p_gen_pf)
    pred_vs_pf["vm_pred"].append(vm_pred)
    pred_vs_pf["vm_pf"].append(vm_pf)

    print(f"Sample {i+1}:")
    print("p_gen (predicted / PF):", np.round(p_gen_pred,3), "/", np.round(p_gen_pf,3))
    print("vm  (predicted / PF):", np.round(vm_pred,3), "/", np.round(vm_pf,3))

# ------------------------
# 8) Visualize training loss
# ------------------------
# If you want to track loss during Direct model training,
# store it inside `train_direct` and return it. Here we mock it.
pred_vs_pf_np = {k: np.array(v) for k,v in pred_vs_pf.items()}

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(pred_vs_pf_np["p_gen_pred"].T, "o-", label="Predicted")
plt.plot(pred_vs_pf_np["p_gen_pf"].T, "x--", label="PF-corrected")
plt.xlabel("Generator index")
plt.ylabel("P_gen [MW]")
plt.title("Predicted vs PF Generator Outputs")
plt.legend()

plt.subplot(1,2,2)
plt.plot(pred_vs_pf_np["vm_pred"].T, "o-", label="Predicted")
plt.plot(pred_vs_pf_np["vm_pf"].T, "x--", label="PF-corrected")
plt.xlabel("Generator index")
plt.ylabel("V_mag [p.u.]")
plt.title("Predicted vs PF Bus Voltages")
plt.legend()

plt.tight_layout()
plt.savefig("pred_vs_pf.png", dpi=300)  # Save figure for LaTeX
plt.show()





'''losses = [loss_fn(input_dict).get('loss').item() for _ in range(50)]  # dummy example

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Penalty Loss')
plt.title('Direct Model Training (L2O Surrogate)')
plt.show()

vm_pred = Y_pred.detach().numpy()[:, n_gen:]      # predicted
vm_feas_list = []                                 # PF corrected

for i in range(len(X_test)):
    p_gen_pred = Y_pred[i].detach().numpy()[:len(net.gen)]
    vm_pred_i = Y_pred[i].detach().numpy()[len(net.gen):]
    _, vm_feas = postprocess_pf(net, p_gen_pred, vm_pred_i)
    vm_feas_list.append(vm_feas)

vm_feas_array = np.array(vm_feas_list)

plt.scatter(range(len(vm_pred.flatten())), vm_pred.flatten(), label='Predicted')
plt.scatter(range(len(vm_feas_array.flatten())), vm_feas_array.flatten(), label='PF Corrected', marker='x')
plt.axhline(0.9, color='red', linestyle='--', alpha=0.5)
plt.axhline(1.1, color='red', linestyle='--', alpha=0.5)
plt.xlabel('Generator Index')
plt.ylabel('Voltage [pu]')
plt.title('Voltage Magnitudes Before and After PF')
plt.legend()
plt.show()

pp.runpp(net)
print("Bus voltages after PF:", net.res_bus.vm_pu.values)
print("Predicted vm:", vm_pred)
'''