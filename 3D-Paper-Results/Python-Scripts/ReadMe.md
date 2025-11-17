## 3D Postprocessing: Probability, Loss Rate, Plots & GIFs

This script is a **post-processing pipeline** for my 3D spinor Schrödinger simulations.  
It reads the simulation output (saved as NumPy arrays), computes global quantities, and generates 2D/3D visualizations and GIFs.

### What the script does

Once you run it inside (or under) a simulation folder, it:

1. **Automatically finds the data directory**

   - Walks upwards from the script location until it finds a `constants.npz`.
   - Uses that directory as `data_dir`.
   - Creates a subfolder:
     ```text
     plots_general2/
     ```
     where all PNGs and GIFs are written.

2. **Loads simulation metadata and grids**

   Expects the following files in `data_dir`:

   - `constants.npz` with keys:
     - `Lx, Ly, Lz` – box lengths  
     - `hx, hy, hz` – grid spacings  
     - `Nx, Ny, Nz` – grid sizes  
     - `mid_x, mid_y, mid_z` – middle indices
   - `X_cpu.npy`, `Y_cpu.npy`, `Z_cpu.npy` – 3D coordinate grids

3. **Total probability vs time**

   Uses:
   - `prob_times.npy`
   - `total_probs.npy`

   It:

   - Trims the arrays if their lengths don’t match.
   - Plots and saves:
     ```text
     plots_general2/total_probability_vs_time.png
     ```

4. **Probability loss rate**

   From `total_probs(t)` it computes the discrete derivative
   \(-\frac{d}{dt}\|\psi_t\|^2\) using forward/central/backward differences
   and saves:

   ```text
   plots_general2/probability_loss_rate_vs_time.png
