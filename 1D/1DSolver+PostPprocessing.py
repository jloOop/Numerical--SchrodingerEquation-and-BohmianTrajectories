#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU-friendly simulation and post-processing module for Bohmian trajectories
based on the time-dependent 1D Schrödinger equation with Robin boundary
conditions (absorbing BC): ∂z ψ = i*k*ψ in a box z ∈ [0, L].
We use the Crank–Nicolson finite-difference method.
"""

import time
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
import os
from numba import njit
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib

# -------------------------------------------------------------------------
# Global Matplotlib style (for nicer, more modern plots)
# -------------------------------------------------------------------------
plt.style.use("seaborn-v0_8")  # or "ggplot", "classic", ...

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 14,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "#f7f7f7",
})

start_time = time.time()  # Timing start

# -------------------------------------------------------------------------
# Output directory
# -------------------------------------------------------------------------
output_directory = "/Users/alirezajozani/Desktop/Rodi_T/GitHub/1D"
os.makedirs(output_directory, exist_ok=True)

# -------------------------------------------------------------------------
# 1) Parameters
# -------------------------------------------------------------------------
N = 2000
L = 400.0
z_full = np.linspace(0, L, N + 1)
z = z_full[:N]  # grid for psi (N points)
dz = L / N
dt = 0.001
T = 600.0
num_steps = int(T / dt)
k = 1.0
p_z = 1.0

# Gaussian initial parameters (used if Gaussian packet is activated)
z0 = L / 2.0  # center
sigma = 5.0   # width

# Monitoring positions (e.g., at 0, L/6, 2L/6, ..., L)
z_positions = [0, L/6, 2*L/6, 3*L/6, 4*L/6, 5*L/6, L]
z_indices = [int(round(z_pos / dz)) for z_pos in z_positions]
z_indices[-1] = N - 1  # ensure the last index doesn’t exceed N-1

time_between_snapshots = 0.2
snapshot_interval = max(1, int(time_between_snapshots / dt))
animation_fps = 20

# -------------------------------------------------------------------------
# 2) Initial Conditions: Plane wave OR Gaussian packet
# -------------------------------------------------------------------------

# --- Plane wave (currently used) ---
psi = np.exp(1j * p_z * z)
M = 10
Q = np.linspace(0, L, M + 2)[1:-1]  # uniform initial particle positions
active = np.ones(M, dtype=bool)
Q_history = [Q.copy()]

# --- Gaussian wave packet (alternative) ---
# psi = ((1 / (2 * np.pi * sigma**2))**0.25
#        * np.exp(-(z - z0)**2 / (4 * sigma**2))
#        * np.exp(1j * p_z * z))
#
# M = 10
# Q = np.random.normal(loc=z0, scale=sigma, size=M)  # initial sample from Gaussian
# Q = np.clip(Q, 0, L)  # ensure particles stay within domain
# active = np.ones(M, dtype=bool)
# Q_history = [Q.copy()]

# -------------------------------------------------------------------------
# 3) Crank–Nicolson Discretization
# -------------------------------------------------------------------------
r = dt / (4.0 * dz**2)

subdiag = -1j * r * np.ones(N - 1, dtype=complex)
diag = (1.0 + 2.0j * r) * np.ones(N, dtype=complex)
supdiag = -1j * r * np.ones(N - 1, dtype=complex)

A = diags([subdiag, diag, supdiag], [-1, 0, 1],
          shape=(N, N), dtype=complex).tocsc()

boundary_term = 1.0 + 1.0j * r + r * k * dz
off_diag = -1j * r

# Robin boundary conditions
A[0, 0] = boundary_term
A[0, 1] = off_diag
A[N - 1, N - 2] = off_diag
A[N - 1, N - 1] = boundary_term

lu = splu(A)

@njit
def compute_rhs(psi, r, k, dz, N):
    RHS = np.zeros(N, dtype=np.complex128)
    RHS[1:N-1] = (
        1j * r * psi[0:N-2]
        + (1.0 - 2.0j * r) * psi[1:N-1]
        + 1j * r * psi[2:N]
    )
    rhs_boundary = 1.0 - 1j * r - r * k * dz
    rhs_off = 1j * r
    RHS[0] = rhs_boundary * psi[0] + rhs_off * psi[1]
    RHS[N-1] = rhs_boundary * psi[N-1] + rhs_off * psi[N-2]
    return RHS

@njit
def compute_dpsi_dz(psi, dz, N):
    dpsi_dz = np.zeros(N, dtype=np.complex128)
    # central differences in the interior
    dpsi_dz[1:N-1] = (psi[2:N] - psi[0:N-2]) / (2 * dz)
    # one-sided at boundaries
    dpsi_dz[0] = (psi[1] - psi[0]) / dz
    dpsi_dz[N-1] = (psi[N-1] - psi[N-2]) / dz
    return dpsi_dz

def interpolate_psi_and_dpsi(Q_val, z, psi, dpsi_dz):
    """
    Linear interpolation of psi and dpsi_dz at position Q_val.
    """
    j = np.searchsorted(z, Q_val) - 1
    if j < 0:
        j = 0
    elif j >= len(z) - 1:
        j = len(z) - 2
    s = (Q_val - z[j]) / (z[j+1] - z[j])
    psi_Q = (1 - s) * psi[j] + s * psi[j+1]
    dpsi_Q = (1 - s) * dpsi_dz[j] + s * dpsi_dz[j+1]
    return psi_Q, dpsi_Q

# -------------------------------------------------------------------------
# 4) Time Evolution
# -------------------------------------------------------------------------
time_subset = np.linspace(0, T, 1000)
step_subset = [int(t / dt) for t in time_subset]

psi_at_z = np.zeros((len(z_indices), len(time_subset)), dtype=complex)
psi_at_z[:, 0] = psi[z_indices]

psi_history = [psi.copy()]
time_history = [0.0]

for step in range(num_steps):
    # Crank–Nicolson step
    RHS = compute_rhs(psi, r, k, dz, N)
    psi = lu.solve(RHS)
    dpsi_dz = compute_dpsi_dz(psi, dz, N)

    # Update Bohmian trajectories
    for m in range(M):
        if active[m]:
            psi_Q, dpsi_Q = interpolate_psi_and_dpsi(Q[m], z, psi, dpsi_dz)
            if np.abs(psi_Q) > 1e-10:
                v = np.imag(dpsi_Q / psi_Q)
            else:
                v = 0.0
            Q[m] += v * dt
            if Q[m] < 0 or Q[m] > L:
                active[m] = False

    # Store psi at selected positions
    if step in step_subset:
        idx = step_subset.index(step)
        psi_at_z[:, idx] = psi[z_indices]

    # Store snapshots for plotting / animation
    if (step + 1) % snapshot_interval == 0 or (step + 1) == num_steps:
        psi_history.append(psi.copy())
        time_history.append((step + 1) * dt)
        Q_history.append(Q.copy())

# -------------------------------------------------------------------------
# 5) Plotting and Saving
# -------------------------------------------------------------------------
save_dir = output_directory
os.makedirs(save_dir, exist_ok=True)

Q_history_array = np.array(Q_history)   # Shape: (num_snapshots, M)
num_snapshots = Q_history_array.shape[0]
M = Q_history_array.shape[1]
time_history = np.array(time_history)

# -------------------------------------------------------------------------
# 5.1 Trajectory PNG (nicer styling)
# -------------------------------------------------------------------------
fig_traj, ax_traj = plt.subplots()

# Use a colormap for trajectories
cmap = plt.cm.cividis  # choices: viridis, plasma, inferno, cividis, magma etc.

for m in range(M):
    color = cmap(m / max(M - 1, 1))
    ax_traj.plot(time_history,
                 Q_history_array[:, m],
                 color=color,
                 linewidth=2.0,
                 alpha=0.8)

# Domain boundaries
ax_traj.axhline(y=0, color="black", linestyle="--", alpha=0.4)
ax_traj.axhline(y=L, color="black", linestyle="--", alpha=0.4)

ax_traj.set_xlabel("Time")
ax_traj.set_ylabel(r"Position $Q(t)$")
ax_traj.set_title("Bohmian Particle Trajectories")

ax_traj.set_xlim(time_history[0], time_history[-1])
ax_traj.set_ylim(-0.05 * L, 1.05 * L)

fig_traj.tight_layout()

png_path = os.path.join(save_dir,
                        f"bohmian_trajectories__kappa={k:.0f}.png")
print("Saving PNG to:", png_path)
fig_traj.savefig(png_path, dpi=300)
plt.show()

# -------------------------------------------------------------------------
# 5.2 Animation (MP4 + GIF) with shaded |psi|^2
# -------------------------------------------------------------------------
fig, ax = plt.subplots()
psi0_abs2 = np.abs(psi_history[0])**2

# Line for |psi|^2
psi_line, = ax.plot(z, psi0_abs2,
                    linewidth=1.5,
                    label=r"$|	heta|^2$")

# Filled region under |psi|^2
psi_fill = ax.fill_between(z, psi0_abs2, 0, alpha=0.0)

# Bohmian particles
particle_scatter = ax.scatter(Q_history[0],
                              np.zeros_like(Q_history[0]),
                              s=40,
                              color="red",
                              edgecolors="black",
                              linewidths=0.5,
                              label="Bohmian particles")

ax.set_xlim(0, L)
max_psi2 = max([np.max(np.abs(psi)**2) for psi in psi_history])
ax.set_ylim(-0.1 * max_psi2, 1.1 * max_psi2)

ax.set_xlabel("z")
ax.set_ylabel(r"$|	heta|^2$")
ax.set_title(f"Bohmian Trajectories at t = {time_history[0]:.2f}")
ax.legend()

def update(frame):
    global psi_fill

    psi_abs2 = np.abs(psi_history[frame])**2

    # Update line
    psi_line.set_ydata(psi_abs2)

    # Update filled region: remove old and redraw
    psi_fill.remove()
    psi_fill = ax.fill_between(z, psi_abs2, 0, alpha=0.0)

    # Update particles
    particle_scatter.set_offsets(
        np.c_[Q_history[frame], np.zeros_like(Q_history[frame])]
    )

    ax.set_title(
    rf"Time evolution of $ho = |	heta|^2$ and Bohmian particles at t = {time_history[frame]:.2f}"
)
    return psi_line, particle_scatter, psi_fill

ani = animation.FuncAnimation(
    fig,
    update,
    frames=range(0, len(time_history), 10),
    interval=50,
    blit=False
)

#mp4_path = os.path.join(save_dir, f"bohmian_animation_kappa={k:.0f}.mp4")
gif_path = os.path.join(save_dir, f"bohmian_animation_kappa={k:.0f}.gif")

# --- Save MP4 ---
# print("Trying to save MP4 to:", mp4_path)
# try:
#     ani.save(mp4_path, writer='ffmpeg', fps=animation_fps)
#     print("MP4 animation saved successfully:", mp4_path)
# except Exception as e:
#     print("Could not save MP4 with ffmpeg:", e)

# --- Save GIF ---
print("Trying to save GIF to:", gif_path)
try:
    writer = PillowWriter(fps=animation_fps)
    ani.save(gif_path, writer=writer)
    print("GIF animation saved successfully:", gif_path)
except Exception as e2:
    print("Could not save GIF:", e2)

plt.show()
plt.close()

# -------------------------------------------------------------------------
# Timing end
# -------------------------------------------------------------------------
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")
