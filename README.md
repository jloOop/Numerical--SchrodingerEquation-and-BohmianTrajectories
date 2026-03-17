# Numerical methods for quantum particle arrival-time problems

This repository contains Python-based numerical solvers and post-processing tools for the linear time-dependent Schrödinger equation in finite-box and waveguide geometries. Its main focus is the long-standing problem of quantum arrival-time distributions, with absorbing boundary conditions and complex absorbing potentials serving as detector models. The code further supports Bohmian trajectory analysis, probability-current diagnostics, and studies of how arrival-time statistics depend on confinement, spin structure, and boundary conditions. It includes simulations for both spin-0 and spin-1/2 particles.

## Key Features

1. **Solvers and simulations**  
   Python implementations of finite-difference solvers for the time-dependent Schrödinger equation, with emphasis on Crank–Nicolson time stepping and standard forward, backward, and central difference discretizations. The framework supports both 1D and 3D grids, including harmonic potentials, complex absorbing potentials, absorbing boundary conditions, and spinor extensions for spin-1/2 particles.

   For the mathematical formulation of the problems and details of the numerical methods, see `!!!`.

2. **Data analysis and visualization**  
   Post-processing scripts analyze simulation output on the CPU using NumPy, SciPy, and Matplotlib. They produce probability-density plots, Bohmian trajectory visualizations, contour plots, arrival-time histograms, and animated GIFs for interpreting the dynamics and detector statistics.

3. **Technology stack**  
   - Primary focus on GPU acceleration using CUDA through CuPy for high-performance simulations.  
   - Tested on NVIDIA A40, A100, and H200 GPUs on the Helix cluster, with CPU fallbacks also available.  
   - Core libraries include CuPy, NumPy, SciPy, and Matplotlib, with no external dependencies beyond Python 3.8+ and CUDA 11+ for GPU mode.

4. **Mathematical approach**  
   The numerical schemes are designed to treat both unitary and non-unitary evolution in a consistent finite-difference framework, with particular attention to absorbing boundary conditions and detector models relevant to arrival-time questions.

5. **Acknowledgment**  
   I was introduced to this project by Prof. Roderich Tumulka, and it is a privilege to work under his supervision in Tübingen, Germany. I also gratefully acknowledge the computational resources provided by the Helix cluster, NHR@Paderborn (PC²), and NHR@ZIB.

## Getting Started

...
