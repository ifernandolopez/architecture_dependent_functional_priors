#
#rlc_paper_figures.py
#LINEAR CASE
# ============================================================
# RLC CONTROL:
# CLASSICAL CONTROL vs 4 ARCHITECTURAL COMBINATIONS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from scipy.linalg import expm
from scipy.integrate import solve_ivp

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

torch.manual_seed(1234)
np.random.seed(1234)

# ============================================================
# CIRCUIT PARAMETERS
# ============================================================

L = 1.0
R = 0.4
C = 1.0
T = 8.0

q0, i0 = 1.0, 0.0
qT, iT = 0.0, 0.0

# ============================================================
# CLASSICAL MINIMUM-ENERGY CONTROL
# ============================================================

A = np.array([[0.0, 1.0], [-1.0/(L*C), -R/L]])
B = np.array([[0.0], [1.0/L]])

x0 = np.array([[q0], [i0]])
xT = np.array([[qT], [iT]])

N_classic = 3000
t_classic = np.linspace(0, T, N_classic)
dt = t_classic[1] - t_classic[0]

W = np.zeros((2, 2))

for s in t_classic:
    E = expm(A * (T - s))
    W += E @ B @ B.T @ E.T * dt

W_inv = np.linalg.inv(W)
d = xT - expm(A*T) @ x0

u_classic = []

for t in t_classic:
    u_t = B.T @ expm(A.T * (T - t)) @ W_inv @ d
    u_classic.append(u_t.item())

u_classic = np.array(u_classic)

def u_classic_interp(t):
    return np.interp(t, t_classic, u_classic)

def rlc_system(t, x):
    q, i = x
    u = u_classic_interp(t)
    dq = i
    di = -(1.0/(L*C))*q - (R/L)*i + (1.0/L)*u
    return [dq, di]

sol = solve_ivp(
    rlc_system,
    [0, T],
    [q0, i0],
    t_eval=t_classic,
    rtol=1e-9,
    atol=1e-9
)

q_classic = sol.y[0]
i_classic = sol.y[1]

u_classic_L2_sq = np.trapz(u_classic**2, t_classic)
u_classic_L2 = np.sqrt(u_classic_L2_sq)

print("\nCLASSICAL CONTROL")
print(f"q(T) = {q_classic[-1]:.8f}")
print(f"i(T) = {i_classic[-1]:.8f}")
print(f"E(u) = {u_classic_L2_sq:.8f}")
print(f"L2   = {u_classic_L2:.8f}")

# ============================================================
# NEURAL ARCHITECTURES
# ============================================================

class MLPState(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2)
        )

    def forward(self, t):
        return self.net(t)


class MLPControl(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, t):
        return self.net(t)


class FourierKANState(nn.Module):
    def __init__(self, modes=16, hidden=64):
        super().__init__()
        self.modes = modes
        self.net = nn.Sequential(
            nn.Linear(2*modes + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2)
        )

    def features(self, t):
        tau = t / T
        feats = [tau]
        for k in range(1, self.modes + 1):
            feats.append(torch.sin(2*np.pi*k*tau))
            feats.append(torch.cos(2*np.pi*k*tau))
        return torch.cat(feats, dim=1)

    def forward(self, t):
        return self.net(self.features(t))


class FourierKANControl(nn.Module):
    def __init__(self, modes=16, hidden=64):
        super().__init__()
        self.modes = modes
        self.net = nn.Sequential(
            nn.Linear(2*modes + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def features(self, t):
        tau = t / T
        feats = [tau]
        for k in range(1, self.modes + 1):
            feats.append(torch.sin(2*np.pi*k*tau))
            feats.append(torch.cos(2*np.pi*k*tau))
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

    spectral_centroid = np.sum(freq * amp) / (np.sum(amp) + 1e-12)

    return L2_sq, L2, smooth, freq, amp, spectral_centroid


# ============================================================
# TRAINING
# ============================================================

def train_combo(
    state_net,
    control_net,
    name,
    epochs=15000,
    N_col=900,
    lr_state=5e-4,
    lr_control=5e-4,
    lambda_ode=1.0,
    lambda_ic=100.0,
    lambda_tc=100.0,
    lambda_u=1e-3,
    lambda_smooth=1e-4
):

    print("\n================================================")
    print(f"TRAINING: {name}")
    print("================================================")

    state_net = state_net.to(device)
    control_net = control_net.to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": state_net.parameters(), "lr": lr_state},
            {"params": control_net.parameters(), "lr": lr_control},
        ]
    )

    t_col = torch.linspace(0, T, N_col, device=device).view(-1, 1)
    t_col.requires_grad_(True)

    t0 = torch.tensor([[0.0]], device=device, requires_grad=True)
    tF = torch.tensor([[T]], device=device, requires_grad=True)

    loss_history = []
    ode_history = []
    tc_history = []
    u_history = []
    smooth_history = []

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
        res2 = di + (1.0/(L*C))*q + (R/L)*i - (1.0/L)*u

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
                f"{name:36s} | "
                f"epoch {ep:5d} | "
                f"loss={loss.item():.3e} | "
                f"ODE={loss_ode.item():.3e} | "
                f"TC={loss_tc.item():.3e} | "
                f"U={loss_u.item():.3e} | "
                f"S={loss_smooth.item():.3e}"
            )

    # Evaluation
    t_eval = torch.linspace(0, T, 1600, device=device).view(-1, 1)
    t_eval.requires_grad_(True)

    state = state_net(t_eval)
    u = control_net(t_eval)

    q = state[:, 0:1]
    i = state[:, 1:2]

    t_np = t_eval.detach().cpu().numpy().flatten()
    q_np = q.detach().cpu().numpy().flatten()
    i_np = i.detach().cpu().numpy().flatten()
    u_np = u.detach().cpu().numpy().flatten()

    L2_sq, L2, smooth, freq, amp, centroid = compute_metrics(t_np, u_np)

    print(f"\nRESULTS: {name}")
    print(f"q(0) = {q_np[0]: .8f} | target {q0}")
    print(f"i(0) = {i_np[0]: .8f} | target {i0}")
    print(f"q(T) = {q_np[-1]: .8f} | target {qT}")
    print(f"i(T) = {i_np[-1]: .8f} | target {iT}")
    print(f"E(u)                  = {L2_sq:.8f}")
    print(f"L2 control            = {L2:.8f}")
    print(f"Integral |u'|^2       = {smooth:.8f}")
    print(f"Spectral centroid     = {centroid:.8f}")

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
    }


# ============================================================
# TRAIN THE FOUR ARCHITECTURAL COMBINATIONS
# ============================================================

EPOCHS = 15000

results = []

results.append(
    train_combo(
        MLPState(hidden=64),
        MLPControl(hidden=64),
        name="State MLP + Control MLP",
        epochs=EPOCHS
    )
)

results.append(
    train_combo(
        MLPState(hidden=64),
        FourierKANControl(modes=16, hidden=64),
        name="State MLP + Control FourierKAN",
        epochs=EPOCHS
    )
)

results.append(
    train_combo(
        FourierKANState(modes=16, hidden=64),
        MLPControl(hidden=64),
        name="State FourierKAN + Control MLP",
        epochs=EPOCHS
    )
)

results.append(
    train_combo(
        FourierKANState(modes=16, hidden=64),
        FourierKANControl(modes=16, hidden=64),
        name="State FourierKAN + Control FourierKAN",
        epochs=EPOCHS
    )
)

# ============================================================
# SHORT LABELS FOR FIGURES
# ============================================================

label_map = {
    "State MLP + Control MLP": "MLP / MLP",
    "State MLP + Control FourierKAN": "MLP / FourierKAN",
    "State FourierKAN + Control MLP": "FourierKAN / MLP",
    "State FourierKAN + Control FourierKAN": "FourierKAN / FourierKAN",
}

short_names = [label_map[r["name"]] for r in results]

# Classical-control Fourier metrics, computed with the same convention
# used for the neural controls.
classic_L2_sq, classic_L2_metric, classic_smooth, classic_freq, classic_amp, classic_centroid = compute_metrics(
    t_classic,
    u_classic
)

# ============================================================
# FIGURE 1: CONTROL PROFILES
# ============================================================

plt.figure(figsize=(11, 4.8))

plt.plot(
    t_classic,
    u_classic,
    label=fr"Minimum-energy control, $E(u)={u_classic_L2_sq:.3f}$",
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

plt.savefig("rlc_control_profiles.pdf", bbox_inches="tight")
plt.savefig("rlc_control_profiles.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# FIGURE 2: PHASE PLANE
# ============================================================

plt.figure(figsize=(6.4, 6.4))

plt.plot(
    q_classic,
    i_classic,
    label="Minimum-energy control",
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

plt.savefig("rlc_phase_plane.pdf", bbox_inches="tight")
plt.savefig("rlc_phase_plane.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# FIGURE 3: FOURIER SPECTRA OF THE CONTROLS
# ============================================================

plt.figure(figsize=(11, 4.8))

plt.semilogy(
    classic_freq,
    classic_amp + 1e-12,
    linestyle="--",
    linewidth=2.6,
    label="Minimum-energy control"
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
plt.legend(fontsize=8, frameon=True)
plt.grid(True, which="both", alpha=0.35)
plt.tight_layout()

plt.savefig("rlc_fourier_spectra.pdf", bbox_inches="tight")
plt.savefig("rlc_fourier_spectra.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# FIGURE 4: CONTROL ENERGY
# ============================================================

energies = [r["L2_sq"] for r in results]
smooths = [r["smooth"] for r in results]
centroids = [r["centroid"] for r in results]

plt.figure(figsize=(9.5, 4.3))

plt.bar(range(len(short_names)), energies)
plt.axhline(
    u_classic_L2_sq,
    linestyle="--",
    linewidth=1.8,
    label="Minimum-energy control"
)

plt.xticks(range(len(short_names)), short_names, rotation=20, ha="right")
plt.ylabel(r"$E(u)=\int_0^T u(t)^2\,dt$")
plt.legend(fontsize=8, frameon=True)
plt.grid(True, axis="y", alpha=0.35)
plt.tight_layout()

plt.savefig("rlc_control_energy.pdf", bbox_inches="tight")
plt.savefig("rlc_control_energy.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# FIGURE 5: CONTROL SMOOTHNESS
# ============================================================

plt.figure(figsize=(9.5, 4.3))

plt.bar(range(len(short_names)), smooths)

plt.xticks(range(len(short_names)), short_names, rotation=20, ha="right")
plt.ylabel(r"$\int_0^T |u'(t)|^2\,dt$")
plt.grid(True, axis="y", alpha=0.35)
plt.tight_layout()

plt.savefig("rlc_control_smoothness.pdf", bbox_inches="tight")
plt.savefig("rlc_control_smoothness.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# FIGURE 6: SPECTRAL CENTROID
# ============================================================

plt.figure(figsize=(9.5, 4.3))

plt.bar(range(len(short_names)), centroids)

plt.xticks(range(len(short_names)), short_names, rotation=20, ha="right")
plt.ylabel("Spectral centroid")
plt.grid(True, axis="y", alpha=0.35)
plt.tight_layout()

plt.savefig("rlc_spectral_centroid.pdf", bbox_inches="tight")
plt.savefig("rlc_spectral_centroid.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# OPTIONAL: TRAINING LOSS
# ============================================================

plt.figure(figsize=(10, 4.5))

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

plt.savefig("rlc_training_loss.pdf", bbox_inches="tight")
plt.savefig("rlc_training_loss.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n================================================")
print("SUMMARY TABLE")
print("================================================")

print(
    f"{'Model':32s} | {'E(u)':>12s} | {'L2':>10s} | "
    f"{'Smooth':>12s} | {'Centroid':>10s} | {'q(T)':>10s} | {'i(T)':>10s}"
)
print("-" * 112)

print(
    f"{'Minimum-energy control':32s} | "
    f"{u_classic_L2_sq:12.6f} | "
    f"{u_classic_L2:10.6f} | "
    f"{classic_smooth:12.6f} | "
    f"{classic_centroid:10.6f} | "
    f"{q_classic[-1]:10.6f} | "
    f"{i_classic[-1]:10.6f}"
)

for r in results:
    print(
        f"{label_map[r['name']]:32s} | "
        f"{r['L2_sq']:12.6f} | "
        f"{r['L2']:10.6f} | "
        f"{r['smooth']:12.6f} | "
        f"{r['centroid']:10.6f} | "
        f"{r['q'][-1]:10.6f} | "
        f"{r['i'][-1]:10.6f}"
    )
