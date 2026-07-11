# G2 Invariant Sextic Potential — SageMath Implementations

This folder contains fully working SageMath implementations of the two companion papers by Muhammad Uzair Javed on the exceptional Lie algebra $\mathrm{G}_2$ invariant theory.

| File | Paper | Scope |
|---|---|---|
| `g2_truncated_sextic_potential.py` | *Exact Critical Orbit Classification of a Truncated G2-Invariant Sextic Potential* | Exact algebraic classification of critical orbits, Hessian spectrum, crossing condition, and stabilized cubic solve |
| `g2_stabilized_information_geometry.py` | *Information Geometry of a Coercively Regularized G2-Invariant Sextic Potential* | Exact phase-transition layer, partition-function and Fisher-metric numerics, and thermodynamic-length verification |

Both files target pure Python 3 syntax over the SageMath library, using `import sage.all as sage` rather than `from sage.all import *`, and avoid interactive preparser macros.

---

## Requirements

- SageMath 10.3 or newer
- No optional Sage packages required
- No SciPy/NumPy dependency; the quadrature routines are implemented in-file

## Verified usage

`g2_truncated_sextic_potential.py` was verified in this environment with the Sage CLI:

```bash
sage -q g2_truncated_sextic_potential.py
```

and completed successfully, exiting with status `0`.

`g2_stabilized_information_geometry.py` was revised (2026-07) to fix its
thermodynamic-length check — the previous `thermodynamic_length_integral`
never took a `beta` argument and only re-verified a closed-form calculus
identity, not the paper's actual finite-beta asymptotic claim (Theorem
7.7). It has **not yet been re-run against a real Sage installation**
after this fix; run it yourself and confirm before relying on it:

```bash
sage -q g2_stabilized_information_geometry.py
```

You can also run them directly inside a Sage session:

```sage
load("g2_truncated_sextic_potential.py")
load("g2_stabilized_information_geometry.py")
```

---

## What is implemented

### `g2_truncated_sextic_potential.py`

This script implements the closed-form results from Paper 1 for

$$V = \mu^2 I_2 + \kappa I_2^2 + \lambda I_6$$

and the stabilized extension

$$V_{\mathrm{stab}} = V + \eta I_2^4.$$

It includes:

- `short_root_radii` and `long_root_radius` for the nonzero critical radii
- `short_root_hessian` and `long_root_hessian` for the normal-slice Hessian spectrum
- `crossing_kappa_star` for the exact energy-crossing condition
- `stabilized_critical_radius` for the exact cubic solve in the stabilized model
- `eta_crit` and `global_vacuum_inequality` for the stabilization criteria

The verification suite checks stationarity, Hessian consistency, the crossing identity, and the stabilized cubic root.

### `g2_stabilized_information_geometry.py`

This script implements the regularized Gibbs model from Paper 2:

$$V_{\mathrm{reg}} = \mu^2 I_2 + \kappa I_2^2 + \lambda I_6 + \nu I_2^3,$$

with the coercivity condition $\nu > |\lambda|$.

It includes:

- `kappa_c`, `kappa_coalescence`, `u_plus`, and `ground_state_energy`
- `A_curvature` for the broken-phase Fisher-metric coefficient
- `constant_C0` and `constant_C1` for the closed-form asymptotic coefficients
- `partition_function_reduced` and `fisher_metric_numeric` for direct finite-$\beta$ numerics
- `logistic_profile_length_identity` — closed-form calculus check that the idealized crossover profile integrates to $\pi$ (has no `beta` argument; verifies the algebra of the limiting profile only, not the paper's asymptotic claim)
- `kappa_crossover_center` and `finite_beta_thermodynamic_length` — the genuine, $\beta$-dependent thermodynamic length (Theorem 7.7), built from `partition_function_reduced`/`fisher_metric_numeric` at an actual finite $\beta$, expected to converge to $\pi$ only slowly as $\beta\to\infty$

The verification suite checks ground-state continuity, stationarity, curvature consistency, the closed-form $\pi$ identity, the genuine finite-$\beta$ trend toward $\pi$, and the Fisher-metric scaling behavior.

---

## Design conventions

- No `assert` statements for validation; invalid inputs raise explicit exceptions
- Coercive scalar validation is used rather than strict parent equality checks
- Immutable dataclasses hold the structural parameters
- Logging is used instead of ad-hoc printing
- Randomized tests are seeded deterministically
- Heavy computations are guarded by Sage timeouts where appropriate

## Notes

- The normalization constant `c0` in the asymptotic coefficients of Paper 2 is left as a free positive constant in the implementation, matching the paper’s own discussion.
- The numerical quadrature routines are intentionally simple and robust for the smooth, exponentially peaked integrands in these models.
- The implementations were also checked against the paper PDFs for the key formulas and numerical statements.
