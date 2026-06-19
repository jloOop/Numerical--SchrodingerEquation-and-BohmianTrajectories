# Selected 3D Visualizations and Single-Particle Evolution

This repository presents a selected set of numerical results related to the paper “[…]”.  
It is organized into two main folders:

---

## 1. `Python-Scripts/`

This folder contains the **solver scripts** and **loader/post-processing scripts**:

- **Solver scripts**: 3D time-dependent Schrödinger solvers, including spinor dynamics, Spinor absorbing boundary conditions, and Bohmian trajectories.  
- **Loader / post-processing scripts**: tools to read the saved data (e.g. `rho_prob_t*.npy`) and generate visualizations such as contour plots, isosurfaces, 3D scatter plots, and animations (GIFs).

The idea is to keep **physics / numerics** (solvers) and **visualization** (loaders) clearly separated.

---

## 2. `3D-results/`

This folder collects representative results for the setups studied in the paper.  
It includes, for various parameter choices:

- **3D time evolution (GIFs)** of a spinor wave packet in a finite box,  
- **Propagation towards and interaction with an absorbing boundary condition** at the top (with and without spinor structure at the top boundary),  
- **Bohmian particle trajectories**, guided by the probability density and the spin-dependent Bohmian equation of motion.

Together, these outputs provide a visual companion to the theoretical and numerical analysis presented in the paper.
