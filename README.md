# Numerical Schrödinger Equation and Bohmian Trajectories

Focused research-code and reproducibility repository for numerical quantum detection-time simulations using absorbing boundary conditions and complex absorbing potentials. The repository documents Python/CuPy finite-difference workflows, probability-current and norm-loss diagnostics, and Bohmian-trajectory sampling used to support publication-related figures and numerical analysis.

This is a scientific-computing and numerical-physics portfolio repository. It is not a production software package.

## Related paper

Associated paper:

> A. Jozani and R. Tumulka, “Detection Time Distribution Predicted Using Absorbing Boundary Conditions and Imaginary Potentials.”

Repository status note: this repository contains publication-related research code, selected data, figures, and post-processing material. Add the final DOI/journal metadata here only after checking the accepted/published bibliographic record.

## Scientific problem

The project studies how to compute the distribution of the time at which a quantum detector registers a particle. The simulations compare various detector-present models based on:

- **absorbing boundary conditions** (ABC), where probability is removed through a detecting boundary;
- **complex absorbing potentials** (CAP), where probability is removed in an absorbing detector region;
- **diagnostic trajectory sampling**, where Bohmian trajectories are used as a numerical comparison or sampling tool inside the detector-model context.

The key point is that **detector-present detection-time distributions** are not the same object as a detector-free arrival-time proposal. Bohmian trajectories in this repository should be read as trajectory sampling and diagnostic comparison, not as an experimental measurement claim and not as a universal arrival-time law.

## Start here

| Goal | Open first | Notes |
|---|---|---|
| Understand the project | `PROJECT_SCOPE.md` | Scope, supported claims, and claim boundaries. |
| See repository organization | `REORG_PLAN.md` | Current structure and safe future cleanup plan. |
| Inspect 1D examples | `1D/README.md` | CPU-friendly 1D workflow and figures/GIFs. |
| Inspect selected 3D outputs | `Data/README.md` | Selected data/GIF/release links, not full raw HPC output. |
| Inspect paper context | `Paper/README.md` | Paper PDF and citation-status notes. |
| Inspect scripts | `Python_Scripts/README.md` and `1D/PythonScripts/README.md` | Current script/documentation entry points; some 3D script paths require manual check. |
| Check dependencies | `requirements.txt`, `environment.yml` | CPU-safe defaults plus CuPy/CUDA note. |
| Cite or reuse | `CITATION.cff`, `LICENSE_NOTICE.md` | Citation metadata and reuse-status warning. |

## Repository map

```text
Numerical--SchrodingerEquation-and-BohmianTrajectories/
├── README.md
├── PROJECT_SCOPE.md
├── REORG_PLAN.md
├── CITATION.cff
├── LICENSE_NOTICE.md
├── requirements.txt
├── environment.yml
├── 1D/
│   ├── README.md
│   ├── Outputs/
│   └── PythonScripts/
├── Data/
│   └── 3D-results/
│       ├── Bohmian-Trajectories/
│       └── TimeEvolution-WaveFunction-Gifs/
├── Paper/
├── Python_Scripts/
├── docs/
├── examples/
├── figures/
└── legacy/
```

The current repository uses historical folder names such as `1D/`, `Data/`, `Paper/`, and `Python_Scripts/`. They are preserved in this pass to avoid breaking paths or publication/reproducibility links. See `REORG_PLAN.md` for a safe manual cleanup plan.

## Methods summary

| Component | Description | Claim boundary |
|---|---|---|
| TDSE / Pauli evolution | Finite-difference workflows for Schrödinger/Pauli detector models. | Research code, not a general quantum-computing toolkit. |
| Time stepping | Crank–Nicolson style implicit finite-difference evolution where implemented. | Confirm exact script-level implementation before citing a specific file. |
| Linear algebra | Sparse/non-Hermitian systems and GMRES/Krylov workflow are supported by the project evidence; current script paths should be checked. | Do not present as a polished solver library. |
| ABC | Absorbing boundary removes probability through a detector surface. | Detector model, not a physical detector implementation. |
| CAP | Complex absorbing potential removes probability in a detector region. | Detector model; absorber parameters affect reflection. |
| Detection statistics | Norm loss, probability current, surface flux, and trajectory histograms/post-processing. | Detector-present statistics, not a universal arrival-time law. |
| GPU workflow | CUDA via CuPy where supported by scripts/HPC runs. | Do not claim CUDA C/C++ expertise. |

## Representative outputs

Representative media currently include:

- 1D wave-packet / plane-wave evolution GIFs in `1D/Outputs/`;
- 1D trajectory plots in `1D/Outputs/`;
- selected 3D trajectory-release links in `Data/3D-results/Bohmian-Trajectories/`;
- selected 3D wave-function GIF release links in `Data/3D-results/TimeEvolution-WaveFunction-Gifs/`.

Use `FIGURE_GALLERY_TEMPLATE.md` to build a clean gallery of selected PNG/GIF outputs with one-line descriptions.

## Minimal reproducibility path

A full reproduction of every 3D run may require large outputs and GPU/HPC resources. A safe minimal path is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python "1D/PythonScripts/1DSolver+PostPprocessing.py"
```

Before running, inspect the script for local output paths and adjust them to a relative directory such as `outputs/1d/`. The current 1D script may contain historical local paths and should be checked before use.

For GPU/CuPy runs, install a CuPy wheel matching your CUDA version separately, for example `cupy-cuda12x` or `cupy-cuda11x`. Do not install a CUDA-specific CuPy package without checking the local CUDA runtime.

## Data policy

This repository should contain selected, reduced, and presentation-ready data only. Full raw HPC output can be large and should be kept outside the main Git history or attached through GitHub Releases when appropriate. Whenever a figure/GIF depends on external data, document the data source, parameters, and expected output path.

## Claim boundaries

Safe summary:

> Public research-code and reproducibility repository for Python/CuPy finite-difference workflows supporting numerical quantum detection-time simulations, ABC/CAP detector models, probability-current and norm-loss diagnostics, Bohmian-trajectory sampling, and publication-related post-processing.

Do not describe this repository as:

- production software;
- a commercial simulation platform;
- a complete open-source library;
- CI/CD-backed software;
- cloud/backend/frontend/MLOps software;
- CUDA C/C++ evidence;
- an experimental detector implementation;
- a universal arrival-time law.

## Citation and reuse

See `CITATION.cff` for citation metadata and `LICENSE_NOTICE.md` for reuse status. If no license has been added, no general reuse license is granted by default.






