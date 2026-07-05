# g2-sextic-potential

This repository collects reference implementations and verification code for studies of $G_2$-invariant sextic potentials, orbit-space geometry, and related exceptional Lie-algebra structures. The work is organized into two complementary codebases:

- The Python implementation suite in [python/README.md](python/README.md) provides self-contained scripts and property-based tests for the main formulas in the companion papers.
- The SageMath implementation suite in [sagemath/README.md](sagemath/README.md) provides symbolic and numerical Sage workflows for the same mathematical objects.

## Repository structure

- [python/](python/) — Python reference code and tests
  - [python/paper1_g2_information_geometry.py](python/paper1_g2_information_geometry.py) — information geometry and thermodynamic-length calculations
  - [python/paper2_g2_critical_orbits.py](python/paper2_g2_critical_orbits.py) — critical orbit classification and stabilized potential analysis
  - [python/paper3_g2_su3_restriction.py](python/paper3_g2_su3_restriction.py) — restriction of $G_2$ invariants to $SU(3)$ and orbit-space maps
  - [python/test_paper1_invariants.py](python/test_paper1_invariants.py), [python/test_paper2_invariants.py](python/test_paper2_invariants.py), [python/test_paper3_invariants.py](python/test_paper3_invariants.py) — verification suites
- [sagemath/](sagemath/) — SageMath implementations of the same mathematical results
  - [sagemath/g2_truncated_sextic_potential.py](sagemath/g2_truncated_sextic_potential.py)
  - [sagemath/g2_stabilized_information_geometry.py](sagemath/g2_stabilized_information_geometry.py)

## What this project contains

The code in this repository is designed to make the closed-form algebraic results from the corresponding papers executable and testable:

- exact critical radii and Hessian spectra
- energy-crossing conditions for critical orbits
- stabilized potential formulas
- partition-function and Fisher-metric asymptotics
- restriction identities between $G_2$ and $SU(3)$ invariants
- numerical checks and hypothesis-driven test coverage

## Quick start

### Python

Install the required dependencies, then run the example scripts or tests:

```bash
pip install numpy scipy hypothesis pytest
python python/paper1_g2_information_geometry.py
python python/paper2_g2_critical_orbits.py
python python/paper3_g2_su3_restriction.py
pytest python/test_paper1_invariants.py python/test_paper2_invariants.py python/test_paper3_invariants.py -v
```

### SageMath

Run the SageMath scripts directly with a Sage installation:

```bash
sage -python sagemath/g2_truncated_sextic_potential.py
sage -python sagemath/g2_stabilized_information_geometry.py
```

## Notes

- The folder-specific READMEs in [python/README.md](python/README.md) and [sagemath/README.md](sagemath/README.md) contain the detailed implementation notes, requirements, and usage guidance.
- The Python code is intended to be run with standard CPython plus the listed scientific packages, while the SageMath code targets a full SageMath environment.
