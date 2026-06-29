# Project scope

## What this repository supports

This repository supports evidence for:

- Python scientific-computing workflows;
- finite-difference Schrödinger/Pauli detector-model simulations;
- absorbing boundary conditions and complex absorbing potentials;
- detector-present detection-time distributions;
- probability-current, surface-flux, and norm-loss diagnostics;
- Bohmian trajectory sampling as a diagnostic / Monte Carlo comparison tool;
- selected 1D and 3D simulation outputs, figures, and GIFs;
- publication-related reproducibility and post-processing material;
- GPU/HPC workflows through CUDA via CuPy where supported by the relevant scripts and runs.

## What this repository does not support

Do not use this repository to claim:

- production software engineering;
- mature open-source package maintenance;
- commercial simulation-platform development;
- CI/CD, cloud deployment, backend/frontend engineering, or MLOps;
- CUDA C/C++ expertise;
- an experimental detector implementation;
- a universal arrival-time law;
- complete reproducibility of all raw HPC outputs inside the Git repository.


## Technical distinctions to preserve

- **Detection time**: detector-present click-time distribution from a specified nonunitary detector model.
- **Arrival time**: broader conceptual problem; detector-free proposals should not be conflated with detector-present ABC/CAP simulations.
- **Bohmian trajectories**: used here as trajectory sampling / diagnostic comparison in the detector-model context, not as an unsupported experimental measurement claim.
- **ABC and CAP**: idealized detector models; neither implies a completed physical detector implementation.
- **CUDA via CuPy**: acceptable where supported; do not claim CUDA C/C++.

## Manual-check items before public claims

- Confirm the exact final publication metadata and DOI.
- Confirm whether the PDF in `Paper/` is the final accepted version or an older submission PDF.
- Confirm which 3D solver and loader scripts are present in the repository versus release archives.
- Confirm whether `requirements.txt` and `environment.yml` work on a clean machine.
- Confirm which local examples can run without full HPC data.
- Confirm license/reuse status.
