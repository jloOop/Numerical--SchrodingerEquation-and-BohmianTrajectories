
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

## Safe CV wording

```text
Maintained a public research-code repository supporting numerical quantum-dynamics simulations, detector-time statistics, probability-current diagnostics, Bohmian trajectory sampling, plotting, and representative post-processing material.
```

```text
Used Python/CuPy finite-difference workflows for non-Hermitian Schrödinger/Pauli detector models involving absorbing boundary conditions and complex absorbing potentials.
```

## Safe recruiter wording

```text
This GitHub repository is a research-code and reproducibility portfolio for numerical quantum detection-time simulations. It shows Python/CuPy scientific-computing workflows, absorbing-boundary and complex-absorbing-potential detector models, diagnostic post-processing, selected figures, and representative animations.
```

## Safe technical-hiring-manager wording

```text
The repository documents finite-difference Schrödinger/Pauli detector-model workflows, non-Hermitian ABC/CAP evolution, probability-current and norm-loss diagnostics, and trajectory sampling/post-processing used for publication-related numerical analysis. It should be evaluated as research code, not as a production solver library.
```

## Publication / reproducibility scope

The repository is associated with:

> A. Jozani and R. Tumulka, “Detection Time Distribution Predicted Using Absorbing Boundary Conditions and Imaginary Potentials.”

The repository can document selected scripts, parameters, reduced outputs, figures, and media used for publication-related analysis. It should not be expected to contain every raw HPC output file. Large raw outputs should be kept external or attached through releases with clear provenance.

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
