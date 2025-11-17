

# Numerical PDE solvers to study fundamental questions in quantum mechanics

This repository contains Python-based numerical solvers for the linear time-dependent Schrödinger equation in 3D–3D (space) and 1D (time). The goal is to explore fundamental questions in quantum mechanics, such as the evolution of the wave function of a single particle in a finite box (laboratory-style setup), its trajectories under the Bohmian equation of motion, arrival times of a particle at a detector, and spin–orbit effects under various potentials and boundary conditions, with a focus on absorbing boundary conditions.

      
## Key Features
1. **Solvers and Simulations**:  
   We are discreatizing! Implementations of finite difference methods (e.g., Crank-Nicolson, forward/backward/central differences) to solve the time-dependent Schrödinger equation. Supports 1D and 3D grids with different potential such as harmonic potentials, absorbing boundaries, and spinor extensions for spin-1/2 particles.
To see the details of the solver look at "!!!"

3. **Data Analysis and Visualization**:  
   Post-processing scripts analyze simulation data on CPU (using NumPy, SciPy, and Matplotlib). Outputs include probability distributions, Bohmian trajectories, and visualizations like contour plots, GIFs, and histograms of arrival times.

4. **Technology Stack**:  
   - Primary focus on GPU acceleration using CUDA via CuPy for high-performance simulations.  
   - Tested on NVIDIA A40, A100 and H200 GPUs on the Helix cluster, but includes CPU fallbacks.  
   - Core libraries: CuPy, NumPy, SciPy, Matplotlib. No external dependencies beyond Python 3.8+ and CUDA 11+ for GPU mode.

5. **Mathematical Approach**:  
   The solvers use mainly finite difference discretization for the PDE, with Crank-Nicolson for time-stepping to ensure stability and correct implemenatation of (non)unitarity of the big matrices with suitable boundary conditions include Dirichlet, Neumann, and Robin (absorbing) to for instance answer some question regarding Bohmian mechanics and Bohmian trajectory ariival time of a particle in finite boxes.

   6. **acknowledgment**  I was introduced to this project with Prof. Roderich Tumulka and funded my project with Dean. It is is my priviledge to have an opportunity to work him and other here in Tübingen.

## Getting Started

.....
