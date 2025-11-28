# Python-Scripts

## Solvers

**1. 3D Spinor Schrödinger Equation**

The script `solver_3D_spinor.py` implements the full 3D time evolution of a two-component (spinor) wave function using a Crank–Nicolson scheme on a Cartesian grid. It uses Dirichlet boundary conditions in the transverse directions, a spinor Robin absorbing boundary condition at the top, and a GPU-accelerated sparse linear solve (CuPy + GMRES with a simple diagonal preconditioner). The code constructs the harmonic-oscillator–like trapping potential in the transverse plane, prepares a localized spinor wave packet, advances it in time, and writes all relevant output to disk (coordinate grids, constants, `rho_prob_t*.npy` snapshots at fixed physical times, `prob_times.npy`, `total_probs.npy`, logs, etc.) in a format that is directly compatible with the post-processing scripts.

**2. Spin-dependent Bohmian Dynamics and Arrival time**

Within the same script, the numerically computed spinor solution is also used to set up and propagate Bohmian particle trajectories. Initial particle positions are sampled from the underlying probability density, and the dynamics are governed by the spin-dependent Bohmian velocity field associated with the evolving spinor wave function. The code is designed to track a large ensemble of particles, monitor first-arrival at the top absorbing plane, and (optionally) store a selected subset of trajectories and arrival times for detailed analysis and visualization.


### Bohmian Trajectory Selection & Visualization

The script `plot_selected_trajs.py` is a flexible tool to **select**, **analyze**, and **visualize** Bohmian trajectories produced by the 3D solver. It automatically adapts to three possible data formats:

1. **New “selected” format**  
   - `bohmian_traj_selected.npy` (Nsteps × K_sel × 3)  
   - `bohm_times.npy` (Nsteps)  
   - `traj_indices.npy` (K_sel) – mapping to global particle indices  
   - optionally: `bohm_arrived_mask_selected.npy`, `bohm_t_hit_selected.npy`  
   If selected-arrival data are missing but full arrival information exists, it reconstructs them via `traj_indices.npy`.

2. **Old “subset” format**  
   - `traj_subset.npy` (Nsnap × K₀ × 3)  
   - `traj_subset_times.npy` (Nsnap)  
   - `traj_subset_idx.npy` (K₀)  
   - optionally: `bohm_arrived_mask.npy`, `bohm_t_hit.npy`  
   Arrival data are mapped from the full population to the subset using `traj_subset_idx.npy`.

3. **Full data format**  
   - `bohmian_traj.npy` (Nsteps × M_part × 3)  
   - `bohm_times.npy` (Nsteps)  
   - optionally: `bohm_arrived_mask.npy`, `bohm_t_hit.npy`  

Using `constants.npz`, the script reconstructs the simulation domain and the **detector plane** at \(z_{\mathrm{det}} = L_z - h_z\), so plots are always aligned with the actual absorber used in the solver.

You control which trajectories are shown via command-line options:
- `--K` : number of particles to plot/analyze  
- `--mode` : selection strategy  
  - `random` – random sample from the visible pool  
  - `arrived` / `unarrived` – only detected / only undetected particles  
  - `earliest` / `latest` – those with smallest / largest first-hit times  
  - `spread` – particles whose hit times are roughly evenly spread across the arrival-time distribution  
- `--seed` : RNG seed for reproducible random selections  

For the chosen set of particles, the script:

- Plots **3D trajectories** in the full box, with the detector plane drawn at the top and **color-coding arrived vs. non-arrived** particles (`traj_sel_3D.png`).  
- Produces **2D projections** (XY, XZ, YZ) with the detector plane indicated in the vertical projections (`traj_sel_projections.png`).  
- Plots **\(z(t)\)** for each selected trajectory, marking the first-hit time on each curve when available and showing the detector height as a horizontal line (`traj_sel_z_vs_t.png`).  
- If first-hit times are present for the selected set, it also plots a **histogram of \(t_{\text{hit}}\)** (`traj_sel_t_hit_hist.png`), giving a quick visual summary of arrival-time statistics for exactly the trajectories you inspected.

In short, `plot_selected_trajs.py` is your “microscope” for Bohmian dynamics: it lets you pick meaningful subsets of particles (earliest, latest, spread-out in time, etc.) and then inspect their paths and arrival times in a geometrically faithful way.
