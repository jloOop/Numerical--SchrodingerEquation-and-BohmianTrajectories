#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3-D Crank-Nicolson Schrödinger Solver (GPU-Optimized with CuPy)
Features:
  - Sparse matrix approach with spinor Robin BC at top z, Neumann at bottom z, and Dirichlet in x/y
  - Rashba SOC in bulk (x-y plane)
  - Jacobi-preconditioned GMRES
  - Asynchronous plotting to minimize GPU blocking
  - Full diagnostic outputs (GIF, contours, scatter, isosurface)
  - Bohmian trajectories for spin-dependent case
"""
import sys, time, os, glob, logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import imageio.v2 as iio
# Backend selection - Force CuPy (GPU)
import cupy as cp
from cupyx.scipy.sparse import diags, eye, kron, coo_matrix, csr_matrix
from cupyx.scipy.sparse.linalg import gmres, LinearOperator
backend = "CuPy"
print(f"[info] backend: {backend}")
# Directories
out_dir = Path(os.getenv("OUTDIR", ".")) / "250.Spinor_Siddhant_DirichlerABC_theta=0+C_Omega=50_L=10" 
frame_dir = out_dir / "frames"           #
for d in (out_dir, frame_dir):
    d.mkdir(parents=True, exist_ok=True)
   
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(out_dir / "simulation_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)
# ── tee stdout → log file, like the cylindrical code ───────────────
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data); f.flush()
    def flush(self):
        for f in self.files: f.flush()
sys.stdout = Tee(sys.stdout, open(out_dir / "stdout.txt", "w"))
# Utility
def to_cpu(arr):
    """Return a NumPy array irrespective of backend."""
    return cp.asnumpy(arr)
# Parameters
Nx = 200
Ny = 200
Nz = 1000
Lx = 10.0
Ly = 10.0
Lz = 10.0
k_bc = cp.pi
dt = 5e-4
T_final = 20.0

#p0 = cp.array([0., 0., 1.])
omega = 50.0 # Assuming a value for omega; adjust as needed
Vxy0 = omega**2
V0z = 0.0
alpha=0.0

theta = 0.0 # Polar angle (θ=0 pure up, θ=π/2 equatorial, θ=π pure down)
phi = 0.0 # Relative phase for down (controls S_x/S_y)
DTYPE_R, DTYPE_C = cp.float32, cp.complex64
Mpart = 50000
# Mesh and initial condition
x = cp.linspace(0.0, Lx, Nx, endpoint=False, dtype=DTYPE_R)
y = cp.linspace(0.0, Ly, Ny, endpoint=False, dtype=DTYPE_R)
z = cp.linspace(0.0, Lz, Nz, endpoint=False, dtype=DTYPE_R)
hx = float(x[1] - x[0])
hy = float(y[1] - y[0])
hz = float(z[1] - z[0])

X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
mid_x = Nx // 2
mid_y = Ny // 2
mid_z = Nz // 2

prefactor = cp.sqrt(2 * omega / cp.pi)
Rho = cp.sqrt((X - Lx/2)**2 + (Y - Ly/2)**2)
exp_part = cp.exp(-omega / 2 * Rho**2) # 3D
mask = (Z > 0) & (Z < 1) # Width 1, centered at ~14.5
sin_part = cp.sin(cp.pi * Z) * mask.astype(cp.float32) # Shifted sinusoid to start at 0 within the mask
# Define scalar part (shared for up/down)
psi_scalar = prefactor * sin_part * exp_part # 3D everywhere; outside slab it’s 0

# Define initial spin state (general: both up and down)
c_up = cp.cos(theta / 2)
c_down = cp.sin(theta / 2) * cp.exp(1j * phi)

# Apply to scalar part (no phase term as requested)
psi_up = c_up * psi_scalar
psi_down = c_down * psi_scalar

# Enforce initial Dirichlet BCs
psi_up[[0, -1], :, :] = 0
psi_up[:, [0, -1], :] = 0
psi_up[:, :, 0] = 0
psi_down[[0, -1], :, :] = 0
psi_down[:, [0, -1], :] = 0
psi_down[:, :, 0] = 0

# Normalize the wavefunction (now for full spinor)
def normalize_wavefunction(psi_up, psi_down, hx, hy, hz):
    """Normalize the full spinor wavefunction to ensure total probability = 1."""
    total_prob = cp.sum(cp.abs(psi_up)**2 + cp.abs(psi_down)**2) * (hx * hy * hz)
    norm = cp.sqrt(total_prob)
    psi_up /= norm
    psi_down /= norm
    return psi_up, psi_down

psi_up, psi_down = normalize_wavefunction(psi_up, psi_down, hx, hy, hz) # Normalize total
psi_flat = cp.concatenate((psi_up.ravel(), psi_down.ravel())).astype(DTYPE_C)


# Bohmian particles (cleaned; keeps your clipping approach)
cp.random.seed(0)
Q = cp.empty((Mpart, 3), dtype=DTYPE_R)

# |ψ|^2 ∝ exp(-ω ρ^2)  ⇒  σ = 1/√(2ω)
sigma = 1.0 / cp.sqrt(2.0 * omega)
mean = Lx / 2 #since Lx=Ly

# x, y ~ Normal(mean, σ), then clip to the box
Q[:, 0] = cp.clip(cp.random.normal(mean, sigma, Mpart), 0, Lx)
Q[:, 1] = cp.clip(cp.random.normal(mean, sigma, Mpart), 0, Ly)

# z ~ 2 sin^2(π z) on [0,1] via rejection sampling
n_gen = 10 * Mpart
while True:
    u = cp.random.uniform(0, 1, n_gen)
    r = cp.random.uniform(0, 2, n_gen)
    keep = r < 2 * cp.sin(cp.pi * u)**2
    if int(keep.sum()) >= Mpart:
        Q[:, 2] = u[keep][:Mpart]   # stays in [0,1]
        break
    n_gen *= 2

eps = min(hx, hy, hz) * 0.5 #To avoid singularities and blowing up
Q[:,0] = cp.clip(Q[:,0], eps, Lx-eps)
Q[:,1] = cp.clip(Q[:,1], eps, Ly-eps)
Q[:,2] = cp.clip(Q[:,2], eps, 1.0-eps)


traj = []
#vel_list = []
#grad_up_list = []
#grad_down_list = []

# Harmonic oscillator potential in x and y directions
V_real = 0.5 * Vxy0 * ((X - Lx/2)**2 + (Y - Ly/2)**2)

# Absorbing potential 

z_step = 0.80 * Lz     
mask   = (Z >= z_step)

V = V_real.astype(DTYPE_C) + (-1j * V0z) * mask.astype(DTYPE_C)

V_diag = V.ravel().astype(DTYPE_C)
V_full = cp.concatenate((V_diag, V_diag))



# ── save coordinate grids & constants once ─────────────────────────
# ── save coordinate grids & constants once ─────────────────────────
X_cpu, Y_cpu, Z_cpu = map(to_cpu,(X,Y,Z))
np.save(out_dir/"X_cpu.npy", X_cpu)
np.save(out_dir/"Y_cpu.npy", Y_cpu)
np.save(out_dir/"Z_cpu.npy", Z_cpu)
constants = {
    "hx": hx, "hy": hy, "hz": hz, "Lx": Lx, "Ly": Ly, "Lz": Lz, "dt": dt,
    "Nx": Nx, "Ny": Ny, "Nz": Nz, "mid_x": mid_x, "mid_y": mid_y, "mid_z": mid_z,
    "Mpart": Mpart
}
np.savez(out_dir/"constants.npz", **constants)
logging.info("[info] Saved coordinate grids and constants")
# <<< END MOVED BLOCK

# frees hundreds of MB, no effect on solver or outputs
del X, Y, Z, Rho, exp_part, psi_scalar, mask, sin_part
cp.get_default_memory_pool().free_all_blocks()



#Print all parameters to stdout (which is teed to stdout.txt)
print("Simulation Parameters:")
print(f"Nx: {Nx}")
print(f"Ny: {Ny}")
print(f"Nz: {Nz}")
print(f"theta: {theta}")
print(f"Lx: {Lx}")
print(f"Ly: {Ly}")
print(f"Lz: {Lz}")
print(f"k_bc: {k_bc}")
print(f"dt: {dt}")
print(f"T_final: {T_final}")
print(f"V0z: {V0z}")
print(f"alpha: {alpha}")
print(f"DTYPE_R: {DTYPE_R}")
print(f"DTYPE_C: {DTYPE_C}")
print(f"omega: {omega}")
print(f"hx: {hx}")
print(f"hy: {hy}")
print(f"hz: {hz}")
print(f"mid_x: {mid_x}")
print(f"mid_y: {mid_y}")
print(f"mid_z: {mid_z}")
print(f"out_dir: {out_dir}")
print(f"mask: (Z > 0) & (Z < 1) # Width 1, centered at ~14.5")
print("-------------------------")


def L_z(N_z, dz, k_bc):
    """
    Discretizes H = -(1/2) d^2/dz^2 with:
      z=0 : Dirichlet (modified stencil at i=1, empty row at i=0)
      z=L : Robin ABC (∂z ψ + i k_bc ψ = 0)
    Returns CSR matrix.
    """
    inv = 1.0 / (dz * dz)
    data, indices, indptr = [], [], [0]
    for i in range(N_z):
        if i == 0:
            indptr.append(len(indices))
        elif i == N_z - 1:
            indices += [i - 1, i]
            data += [-0.5 * inv, 0.5 * inv - 1j * k_bc / (2.0 * dz)]
            indptr.append(len(indices))
        else:
            left = i - 1
            right = i + 1
            if left == 0:
                # Modified for Dirichlet: omit left, adjust diag
                indices += [i, right]
                data += [inv, -0.5 * inv]
                indptr.append(len(indices))
            else:
                indices += [left, i, right]
                data += [-0.5 * inv, inv, -0.5 * inv]
                indptr.append(len(indices))
    data = cp.array(data, dtype=DTYPE_C)
    indices = cp.array(indices, dtype=cp.int32)
    indptr = cp.array(indptr, dtype=cp.int32)
    L_z_raw = csr_matrix((data, indices, indptr), shape=(N_z, N_z))
    return L_z_raw



def L_dirichlet(N, d):
    """
    Constructs a second-derivative operator in x/y with Dirichlet boundary
    conditions, using CuPy's CSR matrix format.
    """
    inv_d2 = 1 / d**2
    half_inv_d2 = 0.5 * inv_d2
    data = []
    indices = []
    indptr = [0]
    for i in range(N):
        if i == 0 or i == N - 1:
            indptr.append(indptr[-1]) # Empty row for Dirichlet
        else:
            left = i-1
            right = i+1
            if left == 0:
                indices.extend([i, right])
                data.append(inv_d2)
                data.append(-half_inv_d2)
                indptr.append(indptr[-1] + 2)
            elif right == N-1:
                indices.extend([left, i])
                data.append(-half_inv_d2)
                data.append(inv_d2)
                indptr.append(indptr[-1] + 2)
            else:
                indices.extend([left, i, right])
                data.append(-half_inv_d2)
                data.append(inv_d2)
                data.append(-half_inv_d2)
                indptr.append(indptr[-1] + 3)
    data = cp.array(data, dtype=DTYPE_C)
    indices = cp.array(indices, dtype=cp.int32)
    indptr = cp.array(indptr, dtype=cp.int32)
    L_raw = csr_matrix((data, indices, indptr), shape=(N, N))
    return L_raw


# Laplacian with BCs
L1 = L_z(Nz, hz, k_bc)
L2 = L_dirichlet(Nx, hx)
L3 = L_dirichlet(Ny, hy)
Ix = eye(Nx, dtype=DTYPE_C, format='csr')
Iy = eye(Ny, dtype=DTYPE_C, format='csr')
Iz = eye(Nz, dtype=DTYPE_C, format='csr')

Lap_scalar = (kron(L2, kron(Iy, Iz)) + kron(Ix, kron(L3, Iz)) + kron(Ix, kron(Iy, L1))).tocsr()
Lap_diag = kron(eye(2, dtype=DTYPE_C, format='csr'), Lap_scalar)

# Build coupling matrix C for spinor ABC at top boundary (Hermitian when k_bc=0)
rows_list, cols_list, data_list = [], [], []
coef = 0.5 / hz
d_coef_x = 1.0 / (2.0 * hx)
d_coef_y = 1.0 / (2.0 * hy)
Ngrid = Nx * Ny * Nz

for ix in range(Nx):
    for iy in range(Ny):
        z_top = Nz - 1
        base = ix * Ny * Nz + iy * Nz + z_top

        # ---------- UP row: +(∂x - i∂y)/hz acting on ψ_down ----------
        row_up = base
        # x: left -, right +
        if ix > 0:
            col = Ngrid + (ix - 1) * Ny * Nz + iy * Nz + z_top
            rows_list.append(row_up); cols_list.append(col); data_list.append(coef * (-d_coef_x))
        if ix < Nx - 1:
            col = Ngrid + (ix + 1) * Ny * Nz + iy * Nz + z_top
            rows_list.append(row_up); cols_list.append(col); data_list.append(coef * (+d_coef_x))
        # y: (iy-1): +i, (iy+1): -i
        if iy > 0:
            col = Ngrid + ix * Ny * Nz + (iy - 1) * Nz + z_top
            rows_list.append(row_up); cols_list.append(col); data_list.append(coef * (+1j) * d_coef_y)
        if iy < Ny - 1:
            col = Ngrid + ix * Ny * Nz + (iy + 1) * Nz + z_top
            rows_list.append(row_up); cols_list.append(col); data_list.append(coef * (-1j) * d_coef_y)

        # ---------- DOWN row:  - (∂x + i∂y)/hz acting on ψ_up  <-- NOTE THE LEADING MINUS
        row_down = Ngrid + base
        # x: left +, right -  (because of the overall leading minus)
        if ix > 0:
            col = (ix - 1) * Ny * Nz + iy * Nz + z_top
            rows_list.append(row_down); cols_list.append(col); data_list.append(coef * (+d_coef_x))
        if ix < Nx - 1:
            col = (ix + 1) * Ny * Nz + iy * Nz + z_top
            rows_list.append(row_down); cols_list.append(col); data_list.append(coef * (-d_coef_x))
        # y: (iy-1): +i, (iy+1): -i  (because −( +i∂y ) = −i∂y ⇒ coefficients flip sign vs up-block conj transpose)
        if iy > 0:
            col = ix * Ny * Nz + (iy - 1) * Nz + z_top
            rows_list.append(row_down); cols_list.append(col); data_list.append(coef * (+1j) * d_coef_y)
        if iy < Ny - 1:
            col = ix * Ny * Nz + (iy + 1) * Nz + z_top
            rows_list.append(row_down); cols_list.append(col); data_list.append(coef * (-1j) * d_coef_y)

C = coo_matrix(
    (cp.array(data_list, dtype=DTYPE_C), (cp.array(rows_list), cp.array(cols_list))),
    shape=(2 * Ngrid, 2 * Ngrid), dtype=DTYPE_C
).tocsr()

Lap_spinor = (Lap_diag + C).tocsr()

# Define component matrices for debug
#Lx_spin = kron(eye(2, dtype=DTYPE_C, format='csr'), kron(L2, kron(Iy, Iz)))
#Ly_spin = kron(eye(2, dtype=DTYPE_C, format='csr'), kron(Ix, kron(L3, Iz)))
#Lz_spin = kron(eye(2, dtype=DTYPE_C, format='csr'), kron(Ix, kron(Iy, L1))) + C
# ---------- DEBUG: which part is non-Hermitian ---------------------
# def _frob(A): # Frobenius norm for sparse matrix
#     return np.linalg.norm(to_cpu(A.data))
# # Convert to SciPy sparse coo
# Lx_cpu       = Lx_spin.tocoo()
# Ly_cpu       = Ly_spin.tocoo()
# Lz_cpu       = Lz_spin.tocoo()
# C_cpu        = C.tocoo()
# H_noC_cpu    = Lap_diag.tocoo()
# H_withC_cpu  = (Lap_diag + C).tocoo()
# Lz_plusC_cpu = (Lz_spin + C).tocoo()

# for lbl, A in [
#     ('C (spinor ABC)', C_cpu),
#     ('bulk x',         Lx_cpu),
#     ('bulk y',         Ly_cpu),
#     ('z-block',        Lz_cpu),
#     ('z-block + C',    Lz_plusC_cpu),
#     ('TOTAL',          H_noC_cpu),
#     ('TOTAL + C',      H_withC_cpu),
# ]:
#     anti = (A - A.conjugate().transpose()).tocoo()
#     print(f'DEBUG {lbl:<12s}: ‖A − A†‖_F = {_frob(anti):.3e}')
# print('--------------------------------------------------------\n')
# # -----------------
# Crank-Nicolson matrices
Ntot = 2 * Nx * Ny * Nz
Id = eye(Ntot, dtype=DTYPE_C, format='csr')

Potential = diags(V_full, 0, format='csr')
A = Id + 1j*dt/2 * (Lap_spinor + Potential).tocsr()
B = Id - 1j*dt/2 * (Lap_spinor + Potential).tocsr()
inv_diag = 1.0 / A.diagonal()
M = LinearOperator(shape=A.shape, matvec=lambda x: inv_diag * x)
# ── fused 4-th order gradient (NumPy/CuPy) ───────────────────────────
def grad4(u, h, ax):
    r = cp.roll
    g = (-r(u, -2, ax) + 8 * r(u, -1, ax) - 8 * r(u, 1, ax) + r(u, 2, ax)) / (12 * h)
    if ax == 0:
        g[1, :, :] = (-3 * u[1, :, :] + 4 * u[2, :, :] - u[3, :, :]) / (2 * h)
        g[-2, :, :] = (u[-4, :, :] - 4 * u[-3, :, :] + 3 * u[-2, :, :]) / (2 * h)
        g[0, :, :] = (u[1, :, :] - u[0, :, :]) / h
        g[-1, :, :] = (u[-1, :, :] - u[-2, :, :]) / h
    elif ax == 1:
        g[:, 1, :] = (-3 * u[:, 1, :] + 4 * u[:, 2, :] - u[:, 3, :]) / (2 * h)
        g[:, -2, :] = (u[:, -4, :] - 4 * u[:, -3, :] + 3 * u[:, -2, :]) / (2 * h)
        g[:, 0, :] = (u[:, 1, :] - u[:, 0, :]) / h
        g[:, -1, :] = (u[:, -1, :] - u[:, -2, :]) / h
    else:
        g[:, :, 1] = (-3 * u[:, :, 1] + 4 * u[:, :, 2] - u[:, :, 3]) / (2 * h)
        g[:, :, -2] = (u[:, :, -4] - 4 * u[:, :, -3] + 3 * u[:, :, -2]) / (2 * h)
        g[:, :, 0] = (u[:, :, 1] - u[:, :, 0]) / h
        g[:, :, -1] = (u[:, :, -1] - u[:, :, -2]) / h
    return g
# ── trilinear interpolation (vectorised) ─────────────────────────────

def interp3(arr, Q, x, y, z):
    # arr shape: (Nx, Ny, Nz); Q shape: (M, 3)
    hx = float(x[1] - x[0]); Nx = len(x)
    hy = float(y[1] - y[0]); Ny = len(y)
    hz = float(z[1] - z[0]); Nz = len(z)

    # cell indices
    i = cp.clip(((Q[:,0] - x[0]) / hx).astype(cp.int32), 0, Nx-2)
    j = cp.clip(((Q[:,1] - y[0]) / hy).astype(cp.int32), 0, Ny-2)
    k = cp.clip(((Q[:,2] - z[0]) / hz).astype(cp.int32), 0, Nz-2)

    # local coords in cell
    sx = (Q[:,0] - x[i]) / hx
    sy = (Q[:,1] - y[j]) / hy
    sz = (Q[:,2] - z[k]) / hz

    def at(di, dj, dk): return arr[i+di, j+dj, k+dk]

    return ((1-sx)*(1-sy)*(1-sz)*at(0,0,0) +
             sx   *(1-sy)*(1-sz)*at(1,0,0) +
            (1-sx)* sy   *(1-sz)*at(0,1,0) +
             sx   * sy   *(1-sz)*at(1,1,0) +
            (1-sx)*(1-sy)* sz   *at(0,0,1) +
             sx   *(1-sy)* sz   *at(1,0,1) +
            (1-sx)* sy   * sz   *at(0,1,1) +
             sx   * sy   * sz   *at(1,1,1))


def velocity(Q, psi_up, psi_down, d_up, d_down, x, y, z, hx, hy, hz):             #Bohm Velociy dQ/dt=v(Q,t)= J(Q,t) / ρ(Q,t) + 1/2 [∇×S](Q,t) / ρ(Q,t),
    rho = cp.abs(psi_up)**2 + cp.abs(psi_down)**2 #ρ(r,t)=ψ†ψ=∣ψ↑∣2+∣ψ↓∣2
    J   = [cp.imag(psi_up.conj()*d_up[i] + psi_down.conj()*d_down[i]) for i in range(3)] #J(r,t)=ℑ(ψ†∇ψ)
    # up_conj_down = psi_up.conj() * psi_down ##S(r,t)=ψ†σψ is the (unnormalized) spin density vector
    # Sx = 2*cp.real(up_conj_down); Sy = 2*cp.imag(up_conj_down) #Sx=2ℜ(ψ↑∗ψ↓)
    # Sz = cp.abs(psi_up)**2 - cp.abs(psi_down)**2 #Sy=2ℑ(ψ↑∗ψ↓) , #Sz=∣ψ↑∣2−∣ψ↓∣2.
    

    # curl_x =  grad4(Sz, hy, 1) - grad4(Sy, hz, 2)
    # curl_y =  grad4(Sx, hz, 2) - grad4(Sz, hx, 0)
    # curl_z =  grad4(Sy, hx, 0) - grad4(Sx, hy, 1)
    # curl   = [curl_x, curl_y, curl_z]
    
    v = cp.zeros_like(Q)
    rho_Q  = interp3(rho, Q, x, y, z)
    m = rho_Q > 1e-10
    if not cp.any(m):
        return v
    J_Q    = cp.stack([interp3(J[i],    Q, x, y, z) for i in range(3)], axis=1)
    #curl_Q = cp.stack([interp3(curl[i], Q, x, y, z) for i in range(3)], axis=1)
    
    
    # v[m] = (J_Q[m] + 0.5*curl_Q[m]) / rho_Q[m,None]
    v[m] = (J_Q[m]) / rho_Q[m,None]
    return v


# ── simulation bookkeeping ─────────────────────────────────────────
num_steps = int(round(T_final/dt))
prob_steps_cpu = set(to_cpu(cp.linspace(1,num_steps,1000,dtype=int)).tolist())
view_steps_cpu = set(to_cpu(cp.linspace(1,num_steps,10, dtype=int)).tolist())
total_probs = []
prob_times = []
# ── main time loop ─────────────────────────────────────────────────
print(f"[info] Starting time loop – {num_steps} steps")
start_time = time.time()
plot_pool = ProcessPoolExecutor(max_workers=2)
tol, maxiter, restart = 5e-6, 1000, 30
for n in range(1, num_steps+1):
    # progress every 100 steps
    if n % 100 == 0:
        elapsed = time.time()-start_time
        eta = (num_steps-n)*(elapsed/n)/60
        print(f"[progress] Step {n}/{num_steps} (t={n*dt:.2f}) | ETA: {eta:.2f} min", flush=True)
    # Crank–Nicolson advance
    rhs = B @ psi_flat
    psi_flat,_ = gmres(A, rhs, x0=psi_flat, tol=tol, maxiter=maxiter, M=M, restart=restart)
    psi_3d = psi_flat.reshape(2, Nx, Ny, Nz)
    psi_up = psi_3d[0]
    psi_down = psi_3d[1]
    
    # Enforce Dirichlet BCs after solve
    psi_up[[0, -1], :, :] = 0
    psi_up[:, [0, -1], :] = 0
    psi_up[:, :, 0] = 0
    psi_down[[0, -1], :, :] = 0
    psi_down[:, [0, -1], :] = 0
    psi_down[:, :, 0] = 0
        
    # Bohmian update
    d_up = [grad4(psi_up, hx, 0), grad4(psi_up, hy, 1), grad4(psi_up, hz, 2)]
    d_down = [grad4(psi_down, hx, 0), grad4(psi_down, hy, 1), grad4(psi_down, hz, 2)]
    v = velocity(Q, psi_up, psi_down, d_up, d_down, x, y, z, hx, hy, hz)
   # vel_list.append(to_cpu(v.copy()))
    
    # Interpolate gradients at current Q (before update)
    # Bohm velocity uses interp3 and per-axis spacings
    # Sample gradients at particle positions with interp3
    Q += v * dt
    d_up_Q = cp.stack([interp3(d_up[i], Q, x, y, z) for i in range(3)], axis=1)
    d_down_Q = cp.stack([interp3(d_down[i], Q, x, y, z) for i in range(3)], axis=1)
   # grad_up_list.append(to_cpu(d_up_Q))
   # grad_down_list.append(to_cpu(d_down_Q))
    # Save Bohmian data
    Q[:,0] = cp.clip(Q[:,0], 0, Lx - 1e-7)
    Q[:,1] = cp.clip(Q[:,1], 0, Ly - 1e-7)
    Q[:,2] = cp.clip(Q[:,2], 0, Lz - 1e-7)
    
    # Save Bohmian trajectories data
    
    traj.append(to_cpu(Q.copy()))


    # ── diagnostics / I-O cadence ────────────────────────────────
    if (n % 100 == 0) or (n in prob_steps_cpu):
        total_prob = float(to_cpu(cp.sum(cp.abs(psi_3d[0])**2 + cp.abs(psi_3d[1])**2)*(hx*hy*hz)))
        total_probs.append(total_prob)
        prob_times.append(n*dt)
        logging.info(f"[Probability] t={n*dt:.2f}, ∫|ψ|² dV = {total_prob:.6f}")
        # save to disk (overwrite each time)
        #np.save(out_dir/"prob_times.npy", np.array(prob_times))
        #np.save(out_dir/"total_probs.npy", np.array(total_probs))
        #rho_prob = to_cpu(cp.abs(psi_3d[0])**2 + cp.abs(psi_3d[1])**2)
        t = n*dt
        #np.save(out_dir/f"rho_prob_t{t:.1f}.npy", rho_prob)
       
        # New: Save separate densities for up and down
        # rho_up = to_cpu(cp.abs(psi_3d[0])**2)
        # rho_down = to_cpu(cp.abs(psi_3d[1])**2)
        # np.save(out_dir/f"rho_up_t{t:.1f}.npy", rho_up)
        # np.save(out_dir/f"rho_down_t{t:.1f}.npy", rho_down)
       
    # optional live-view at sparse intervals (unchanged numerics)
    if n in view_steps_cpu:
        total_prob = float(to_cpu(cp.sum(cp.abs(psi_3d[0])**2 + cp.abs(psi_3d[1])**2)*(hx*hy*hz)))
        print(f"[Probability] t={n*dt:.2f}, Total Probability = {total_prob:.6f}", flush=True)
        # lightweight 2-D slices – plotting kept identical to original draft
        # rho_up = to_cpu(cp.abs(psi_3d[0])**2)
        # rho_down = to_cpu(cp.abs(psi_3d[1])**2)
        # rho = rho_up + rho_down
        # rho_xy, rho_xz, rho_yz = rho[:,:,mid_z], rho[:,mid_y,:], rho[mid_x,:,:]
        # plotting skipped here – unchanged from user version
# ── final flush ────────────────────────────────────────────────────
np.save(out_dir/"prob_times.npy", np.array(prob_times))
np.save(out_dir/"total_probs.npy", np.array(total_probs))
np.save(out_dir/"bohmian_traj.npy", np.array(traj))
#np.save(out_dir/"bohmian_vel.npy", np.array(vel_list))
#np.save(out_dir/"grad_up_at_Q.npy", np.array(grad_up_list))
#np.save(out_dir/"grad_down_at_Q.npy", np.array(grad_down_list))
bohm_times = np.arange(1, num_steps + 1) * dt
np.save(out_dir/"bohm_times.npy", bohm_times)
elapsed = time.time()-start_time
print(f"Total execution time: {elapsed:.2f} s", flush=True)
print(f"[info] All output files are in: {out_dir.resolve()}")
print(f"[info] Log saved to {out_dir/'simulation_log.txt'}")