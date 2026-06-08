# ============================================================
# CONTROLLED ELECTRICAL DUFFING SYSTEM:
# CLASSICAL NUMERICAL CONTROL vs 4 COMBINACIONES PINN
#
# GPU-ready for Google Colab
#
# q'' + delta q' + alpha q + beta q^3 = u(t)
#
# State:
#   q' = i
#   i' = -delta i - alpha q - beta q^3 + u
#
# PINN cases:
#   1. State MLP        + Control MLP
#   2. State MLP        + Control FourierKAN
#   3. State FourierKAN + Control MLP
#   4. State FourierKAN + Control FourierKAN
# ============================================================

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# ============================================================
# GPU / CPU
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True
else:
    print("No active GPU. In Colab: Runtime -> Change runtime type -> GPU")

# ============================================================
# SEEDS
# ============================================================

SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# DUFFING PARAMETERS
# ============================================================

T = 8.0

delta = 0.25
alpha = 1.0
beta = 1.0

q0, i0 = 1.0, 0.0
qT, iT = 0.0, 0.0

# ============================================================
# NUMERICAL PARAMETERS
# ============================================================

EPOCHS_PINN = 15000
EPOCHS_CLASSIC = 6000

N_col = 900
N_eval = 1600
N_ctrl = 300

# PINN weights
lambda_ode = 1.0
lambda_ic = 100.0
lambda_tc = 100.0
lambda_u = 1e-3
lambda_smooth = 1e-4

# ============================================================
# BASELINE CLASSICAL NUMERICAL CONTROL:
# direct optimization of the discretized control using RK4
# ============================================================

print("\n================================================")
print("CONTROL CLASSICAL NUMERICAL CONTROL POR OPTIMIZACIÓN DIRECTA")
print("================================================")

start_time = time.time()

t_ctrl = torch.linspace(0.0, T, N_ctrl, device=device)
dt = T / (N_ctrl - 1)

u_param = torch.zeros(N_ctrl, device=device, requires_grad=True)

opt_u = torch.optim.Adam([u_param], lr=5e-2)


def duffing_rhs(x, u):
    q = x[0]
    i = x[1]

    dq = i
    di = -delta * i - alpha * q - beta * q**3 + u

    return torch.stack([dq, di])


def simulate_discrete_control(u_vec):
    x = torch.tensor([q0, i0], device=device)

    xs = [x]

    for k in range(N_ctrl - 1):
        u_k = u_vec[k]
        u_k1 = u_vec[k + 1]
        u_mid = 0.5 * (u_k + u_k1)

        k1 = duffing_rhs(x, u_k)
        k2 = duffing_rhs(x + 0.5 * dt * k1, u_mid)
        k3 = duffing_rhs(x + 0.5 * dt * k2, u_mid)
        k4 = duffing_rhs(x + dt * k3, u_k1)

        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        xs.append(x)

    return torch.stack(xs, dim=0)


classic_loss_hist = []

for ep in range(EPOCHS_CLASSIC):

    opt_u.zero_grad()

    xs = simulate_discrete_control(u_param)

    q = xs[:, 0]
    i = xs[:, 1]

    terminal_loss = (q[-1] - qT)**2 + (i[-1] - iT)**2
    energy_loss = torch.mean(u_param**2)

    du = (u_param[1:] - u_param[:-1]) / dt
    smooth_loss = torch.mean(du**2)

    loss = (
        500.0 * terminal_loss
        + 1e-3 * energy_loss
        + 1e-5 * smooth_loss
    )

    loss.backward()
    opt_u.step()

    classic_loss_hist.append(loss.item())

    if ep % 1000 == 0:
        print(
            f"Classic ep={ep:5d} | "
            f"loss={loss.item():.3e} | "
            f"terminal={terminal_loss.item():.3e} | "
            f"energy={energy_loss.item():.3e} | "
            f"smooth={smooth_loss.item():.3e}"
        )

with torch.no_grad():
    xs_classic = simulate_discrete_control(u_param)

t_classic = t_ctrl.detach().cpu().numpy()
q_classic = xs_classic[:, 0].detach().cpu().numpy()
i_classic = xs_classic[:, 1].detach().cpu().numpy()
u_classic = u_param.detach().cpu().numpy()

u_classic_L2_sq = np.trapz(u_classic**2, t_classic)
u_classic_L2 = np.sqrt(u_classic_L2_sq)
u_classic_smooth = np.trapz(np.gradient(u_classic, t_classic)**2, t_classic)

print("\nRESULTADOS CONTROL CLASSICAL NUMERICAL CONTROL")
print(f"q(T) = {q_classic[-1]: .8f}")
print(f"i(T) = {i_classic[-1]: .8f}")
print(f"L2^2 = {u_classic_L2_sq:.8f}")
print(f"L2   = {u_classic_L2:.8f}")
print(f"S    = {u_classic_smooth:.8f}")
print(f"Classical optimization time: {(time.time() - start_time)/60:.2f} min")

# ============================================================
# ARCHITECTURES
# ============================================================

class MLPState(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),

            nn.Linear(hidden, hidden),
            nn.Tanh(),

            nn.Linear(hidden, hidden),
            nn.Tanh(),

            nn.Linear(hidden, 2)
        )

    def forward(self, t):
        return self.net(t)


class MLPControl(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),

            nn.Linear(hidden, hidden),
            nn.Tanh(),

            nn.Linear(hidden, hidden),
            nn.Tanh(),

            nn.Linear(hidden, 1)
        )

    def forward(self, t):
        return self.net(t)


class FourierKANState(nn.Module):
    def __init__(self, modes=16, hidden=64):
        super().__init__()

        self.modes = modes

        self.net = nn.Sequential(
            nn.Linear(2 * modes + 1, hidden),
            nn.Tanh(),

            nn.Linear(hidden, hidden),
            nn.Tanh(),

            nn.Linear(hidden, 2)
        )

    def features(self, t):
        tau = t / T

        feats = [tau]

        for k in range(1, self.modes + 1):
            feats.append(torch.sin(2 * torch.pi * k * tau))
            feats.append(torch.cos(2 * torch.pi * k * tau))

        return torch.cat(feats, dim=1)

    def forward(self, t):
        return self.net(self.features(t))


class FourierKANControl(nn.Module):
    def __init__(self, modes=16, hidden=64):
        super().__init__()

        self.modes = modes

        self.net = nn.Sequential(
            nn.Linear(2 * modes + 1, hidden),
            nn.Tanh(),

            nn.Linear(hidden, hidden),
            nn.Tanh(),

            nn.Linear(hidden, 1)
        )

    def features(self, t):
        tau = t / T

        feats = [tau]

        for k in range(1, self.modes + 1):
            feats.append(torch.sin(2 * torch.pi * k * tau))
            feats.append(torch.cos(2 * torch.pi * k * tau))

        return torch.cat(feats, dim=1)

    def forward(self, t):
        return self.net(self.features(t))


# ============================================================
# UTILITIES
# ============================================================

def grad(y, x):
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]


def compute_metrics(t, u):
    L2_sq = np.trapz(u**2, t)
    L2 = np.sqrt(L2_sq)

    du = np.gradient(u, t)
    smooth = np.trapz(du**2, t)

    U = np.fft.rfft(u - np.mean(u))
    freq = np.fft.rfftfreq(len(u), d=t[1] - t[0])
    amp = np.abs(U)

    centroid = np.sum(freq * amp) / (np.sum(amp) + 1e-12)

    return L2_sq, L2, smooth, freq, amp, centroid


# ============================================================
# PINN TRAINING
# ============================================================

def train_combo(
    state_net,
    control_net,
    name,
    epochs=15000,
    lr_state=1e-3,
    lr_control=1e-3,
    seed=1234
):

    print("\n================================================")
    print(f"TRAINING: {name}")
    print("================================================")

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    state_net = state_net.to(device)
    control_net = control_net.to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": state_net.parameters(), "lr": lr_state},
            {"params": control_net.parameters(), "lr": lr_control},
        ]
    )

    t_col = torch.linspace(0.0, T, N_col, device=device).view(-1, 1)
    t_col.requires_grad_(True)

    t0 = torch.tensor([[0.0]], device=device, requires_grad=True)
    tF = torch.tensor([[T]], device=device, requires_grad=True)

    loss_history = []
    ode_history = []
    tc_history = []
    u_history = []
    smooth_history = []

    start_time = time.time()

    for ep in range(epochs):

        optimizer.zero_grad()

        state = state_net(t_col)
        u = control_net(t_col)

        q = state[:, 0:1]
        i = state[:, 1:2]

        dq = grad(q, t_col)
        di = grad(i, t_col)
        du = grad(u, t_col)

        res1 = dq - i

        res2 = (
            di
            + delta * i
            + alpha * q
            + beta * q**3
            - u
        )

        loss_ode = torch.mean(res1**2) + torch.mean(res2**2)

        state0 = state_net(t0)

        loss_ic = (
            (state0[:, 0:1] - q0)**2
            + (state0[:, 1:2] - i0)**2
        ).mean()

        stateF = state_net(tF)

        loss_tc = (
            (stateF[:, 0:1] - qT)**2
            + (stateF[:, 1:2] - iT)**2
        ).mean()

        loss_u = torch.mean(u**2)
        loss_smooth = torch.mean(du**2)

        loss = (
            lambda_ode * loss_ode
            + lambda_ic * loss_ic
            + lambda_tc * loss_tc
            + lambda_u * loss_u
            + lambda_smooth * loss_smooth
        )

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        ode_history.append(loss_ode.item())
        tc_history.append(loss_tc.item())
        u_history.append(loss_u.item())
        smooth_history.append(loss_smooth.item())

        if ep % 1000 == 0:
            print(
                f"{name:36s} | ep={ep:5d} | "
                f"loss={loss.item():.3e} | "
                f"ODE={loss_ode.item():.3e} | "
                f"TC={loss_tc.item():.3e} | "
                f"U={loss_u.item():.3e} | "
                f"S={loss_smooth.item():.3e}"
            )

    train_minutes = (time.time() - start_time) / 60.0

    # Evaluation
    t_eval = torch.linspace(0.0, T, N_eval, device=device).view(-1, 1)
    t_eval.requires_grad_(True)

    state = state_net(t_eval)
    u = control_net(t_eval)

    q = state[:, 0:1]
    i = state[:, 1:2]

    dq = grad(q, t_eval)
    di = grad(i, t_eval)

    res1 = dq - i
    res2 = di + delta * i + alpha * q + beta * q**3 - u

    residual_eval = torch.mean(res1**2 + res2**2).item()

    t_np = t_eval.detach().cpu().numpy().flatten()
    q_np = q.detach().cpu().numpy().flatten()
    i_np = i.detach().cpu().numpy().flatten()
    u_np = u.detach().cpu().numpy().flatten()

    L2_sq, L2, smooth, freq, amp, centroid = compute_metrics(t_np, u_np)

    print(f"\nRESULTADOS: {name}")
    print(f"q(0) = {q_np[0]: .8f} | target {q0}")
    print(f"i(0) = {i_np[0]: .8f} | target {i0}")
    print(f"q(T) = {q_np[-1]: .8f} | target {qT}")
    print(f"i(T) = {i_np[-1]: .8f} | target {iT}")
    print(f"L2^2 control        = {L2_sq:.8f}")
    print(f"L2 control          = {L2:.8f}")
    print(f"Integral |u'|^2     = {smooth:.8f}")
    print(f"Spectral centroid = {centroid:.8f}")
    print(f"Mean evaluation residual = {residual_eval:.3e}")
    print(f"Training time = {train_minutes:.2f} min")

    return {
        "name": name,
        "state_net": state_net,
        "control_net": control_net,
        "loss": np.array(loss_history),
        "ode_loss": np.array(ode_history),
        "tc_loss": np.array(tc_history),
        "u_loss": np.array(u_history),
        "smooth_loss": np.array(smooth_history),
        "t": t_np,
        "q": q_np,
        "i": i_np,
        "u": u_np,
        "L2_sq": L2_sq,
        "L2": L2,
        "smooth": smooth,
        "freq": freq,
        "amp": amp,
        "centroid": centroid,
        "residual": residual_eval,
        "train_minutes": train_minutes,
    }


# ============================================================
# RUN THE FOUR COMBINATIONS
# ============================================================

results = []

results.append(
    train_combo(
        MLPState(hidden=64),
        MLPControl(hidden=64),
        name="State MLP + Control MLP",
        epochs=EPOCHS_PINN,
        lr_state=1e-3,
        lr_control=1e-3,
        seed=1234
    )
)

results.append(
    train_combo(
        MLPState(hidden=64),
        FourierKANControl(modes=16, hidden=64),
        name="State MLP + Control FourierKAN",
        epochs=EPOCHS_PINN,
        lr_state=1e-3,
        lr_control=5e-4,
        seed=1234
    )
)

results.append(
    train_combo(
        FourierKANState(modes=16, hidden=64),
        MLPControl(hidden=64),
        name="State FourierKAN + Control MLP",
        epochs=EPOCHS_PINN,
        lr_state=5e-4,
        lr_control=1e-3,
        seed=1234
    )
)

results.append(
    train_combo(
        FourierKANState(modes=16, hidden=64),
        FourierKANControl(modes=16, hidden=64),
        name="State FourierKAN + Control FourierKAN",
        epochs=EPOCHS_PINN,
        lr_state=5e-4,
        lr_control=5e-4,
        seed=1234
    )
)

# ============================================================
# FIGURES
# ============================================================

label_map = {
    "State MLP + Control MLP": "MLP / MLP",
    "State MLP + Control FourierKAN": "MLP / FourierKAN",
    "State FourierKAN + Control MLP": "FourierKAN / MLP",
    "State FourierKAN + Control FourierKAN": "FourierKAN / FourierKAN",
}

short_names = [label_map[r["name"]] for r in results]

# Classical Fourier spectrum
classic_L2_sq, classic_L2, classic_smooth, classic_freq, classic_amp, classic_centroid = compute_metrics(
    t_classic,
    u_classic
)

# ============================================================
# FIGURE 1: TRAINING LOSS
# ============================================================

plt.figure(figsize=(10, 4.5))

plt.semilogy(
    classic_loss_hist,
    linestyle="--",
    linewidth=2.2,
    label="Classical numerical control"
)

for r in results:
    plt.semilogy(
        r["loss"],
        linewidth=1.4,
        label=label_map[r["name"]]
    )

plt.xlabel("Epoch")
plt.ylabel("Total loss")
plt.legend(fontsize=8, frameon=True)
plt.grid(True, which="both", alpha=0.35)
plt.tight_layout()

plt.savefig("duffing_training_loss.pdf", bbox_inches="tight")
plt.savefig("duffing_training_loss.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# FIGURE 2: CONTROL PROFILES
# ============================================================

plt.figure(figsize=(11, 4.8))

plt.plot(
    t_classic,
    u_classic,
    "k--",
    label=fr"Classical numerical control, $E(u)={u_classic_L2_sq:.3f}$",
    linewidth=2.8
)

for r in results:
    plt.plot(
        r["t"],
        r["u"],
        linewidth=1.8,
        label=fr"{label_map[r['name']]}, $E(u)={r['L2_sq']:.3f}$"
    )

plt.xlabel(r"$t$")
plt.ylabel(r"$u(t)$")
plt.legend(fontsize=8, frameon=True)
plt.grid(True, alpha=0.35)
plt.tight_layout()

plt.savefig("duffing_control_profiles.pdf", bbox_inches="tight")
plt.savefig("duffing_control_profiles.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# FIGURE 3: STATE q(t)
# ============================================================

plt.figure(figsize=(11, 4.8))

plt.plot(
    t_classic,
    q_classic,
    "k--",
    label="Classical numerical control",
    linewidth=2.8
)

for r in results:
    plt.plot(
        r["t"],
        r["q"],
        linewidth=1.8,
        label=label_map[r["name"]]
    )

plt.xlabel(r"$t$")
plt.ylabel(r"$q(t)$")
plt.legend(fontsize=8, frameon=True)
plt.grid(True, alpha=0.35)
plt.tight_layout()

plt.savefig("duffing_state_q.pdf", bbox_inches="tight")
plt.savefig("duffing_state_q.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# FIGURE 4: STATE i(t)
# ============================================================

plt.figure(figsize=(11, 4.8))

plt.plot(
    t_classic,
    i_classic,
    "k--",
    label="Classical numerical control",
    linewidth=2.8
)

for r in results:
    plt.plot(
        r["t"],
        r["i"],
        linewidth=1.8,
        label=label_map[r["name"]]
    )

plt.xlabel(r"$t$")
plt.ylabel(r"$i(t)$")
plt.legend(fontsize=8, frameon=True)
plt.grid(True, alpha=0.35)
plt.tight_layout()

plt.savefig("duffing_state_i.pdf", bbox_inches="tight")
plt.savefig("duffing_state_i.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# FIGURE 5: PHASE PLANE
# ============================================================

plt.figure(figsize=(6.4, 6.4))

plt.plot(
    q_classic,
    i_classic,
    "k--",
    label="Classical numerical control",
    linewidth=2.8
)

for r in results:
    plt.plot(
        r["q"],
        r["i"],
        linewidth=1.8,
        label=label_map[r["name"]]
    )

plt.scatter([q0], [i0], s=70, label="Initial state", zorder=5)
plt.scatter([qT], [iT], s=70, label="Target state", zorder=5)

plt.xlabel(r"$q(t)$")
plt.ylabel(r"$i(t)$")
plt.legend(fontsize=8, frameon=True)
plt.grid(True, alpha=0.35)
plt.tight_layout()

plt.savefig("duffing_phase_plane.pdf", bbox_inches="tight")
plt.savefig("duffing_phase_plane.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# FIGURE 6: FOURIER SPECTRA OF THE CONTROLS
# ============================================================

plt.figure(figsize=(11, 4.8))

plt.semilogy(
    classic_freq,
    classic_amp + 1e-12,
    "k--",
    linewidth=2.8,
    label="Classical numerical control"
)

for r in results:
    plt.semilogy(
        r["freq"],
        r["amp"] + 1e-12,
        linewidth=1.8,
        label=label_map[r["name"]]
    )

plt.xlabel("Frequency")
plt.ylabel(r"$|\widehat{u}(f)|$")
plt.xlim(0, 8)
plt.legend(fontsize=8, frameon=True, loc="upper right")
plt.grid(True, which="both", alpha=0.35)
plt.tight_layout()

plt.savefig("duffing_fourier_spectra.pdf", bbox_inches="tight")
plt.savefig("duffing_fourier_spectra.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# FIGURE 7: CONTROL ENERGY
# ============================================================

energies = [r["L2_sq"] for r in results]
smooths = [r["smooth"] for r in results]
centroids = [r["centroid"] for r in results]
residuals = [r["residual"] for r in results]

plt.figure(figsize=(9.5, 4.3))

plt.bar(range(len(short_names)), energies)
plt.axhline(
    u_classic_L2_sq,
    linestyle="--",
    linewidth=1.8,
    label="Classical numerical control"
)

plt.xticks(range(len(short_names)), short_names, rotation=20, ha="right")
plt.ylabel(r"$E(u)=\int_0^T u(t)^2\,dt$")
plt.legend(fontsize=8, frameon=True)
plt.grid(True, axis="y", alpha=0.35)
plt.tight_layout()

plt.savefig("duffing_control_energy.pdf", bbox_inches="tight")
plt.savefig("duffing_control_energy.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# FIGURE 8: CONTROL SMOOTHNESS
# ============================================================

plt.figure(figsize=(9.5, 4.3))

plt.bar(range(len(short_names)), smooths)
plt.axhline(
    u_classic_smooth,
    linestyle="--",
    linewidth=1.8,
    label="Classical numerical control"
)

plt.xticks(range(len(short_names)), short_names, rotation=20, ha="right")
plt.ylabel(r"$\int_0^T |u'(t)|^2\,dt$")
plt.legend(fontsize=8, frameon=True)
plt.grid(True, axis="y", alpha=0.35)
plt.tight_layout()

plt.savefig("duffing_control_smoothness.pdf", bbox_inches="tight")
plt.savefig("duffing_control_smoothness.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# FIGURE 9: SPECTRAL CENTROID
# ============================================================

plt.figure(figsize=(9.5, 4.3))

plt.bar(range(len(short_names)), centroids)

plt.xticks(range(len(short_names)), short_names, rotation=20, ha="right")
plt.ylabel("Spectral centroid")
plt.grid(True, axis="y", alpha=0.35)
plt.tight_layout()

plt.savefig("duffing_spectral_centroid.pdf", bbox_inches="tight")
plt.savefig("duffing_spectral_centroid.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# FIGURE 10: DYNAMIC RESIDUAL
# ============================================================

plt.figure(figsize=(9.5, 4.3))

plt.bar(range(len(short_names)), residuals)
plt.yscale("log")

plt.xticks(range(len(short_names)), short_names, rotation=20, ha="right")
plt.ylabel("Mean dynamic residual")
plt.grid(True, axis="y", alpha=0.35)
plt.tight_layout()

plt.savefig("duffing_dynamic_residual.pdf", bbox_inches="tight")
plt.savefig("duffing_dynamic_residual.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n================================================")
print("DUFFING SUMMARY TABLE")
print("================================================")

print(
    f"{'Control strategy':40s} | {'E(u)':>12s} | {'L2':>10s} | "
    f"{'Smoothness':>12s} | {'Centroid':>10s} | {'Residual':>10s} | "
    f"{'q(T)':>10s} | {'i(T)':>10s} | {'min':>8s}"
)

print("-" * 150)

print(
    f"{'Classical numerical control':40s} | "
    f"{u_classic_L2_sq:12.6f} | "
    f"{u_classic_L2:10.6f} | "
    f"{u_classic_smooth:12.6f} | "
    f"{classic_centroid:10.6f} | "
    f"{'--':>10s} | "
    f"{q_classic[-1]:10.6f} | "
    f"{i_classic[-1]:10.6f} | "
    f"{'--':>8s}"
)

for r in results:
    print(
        f"{label_map[r['name']]:40s} | "
        f"{r['L2_sq']:12.6f} | "
        f"{r['L2']:10.6f} | "
        f"{r['smooth']:12.6f} | "
        f"{r['centroid']:10.6f} | "
        f"{r['residual']:10.3e} | "
        f"{r['q'][-1]:10.6f} | "
        f"{r['i'][-1]:10.6f} | "
        f"{r['train_minutes']:8.2f}"
    )
