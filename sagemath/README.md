# G2 Invariant Sextic Potential — SageMath Implementations

Pure-Python/SageMath implementations of two companion papers by Muhammad Uzair Javed on the
exceptional Lie algebra `g2` invariant theory:

| File | Paper | Scope |
|---|---|---|
| `g2_truncated_sextic_potential.py` | *Exact Critical Orbit Classification of a Truncated G2-Invariant Sextic Potential* | Exact algebra: critical orbits, Hessian spectrum, energy crossing, stabilized cubic |
| `g2_stabilized_information_geometry.py` | *Information Geometry of a Coercively Regularized G2-Invariant Sextic Potential* | Exact phase transition + rigorous/numeric partition-function and Fisher-metric asymptotics |

Both target **pure Python 3 syntax over the SageMath library** (no interactive `R.<x> = ...`
preparser macros), using `import sage.all as sage` rather than `from sage.all import *`.

---

## Requirements

- **SageMath >= 10.3** (tested against no earlier version; relies on `RealField`, `SR`,
  `gen_legendre_P`, `alarm`/`AlarmInterrupt`, `set_random_seed`).
- No optional Sage packages, no SciPy/NumPy — quadrature is implemented natively in-file.
- Run with the Sage Python interpreter, e.g.:
  ```bash
  sage -python g2_truncated_sextic_potential.py
  sage -python g2_stabilized_information_geometry.py
  ```
  or from inside a Sage session: `sage: load("g2_truncated_sextic_potential.py")`.

> **Not executed in this environment.** These files were authored and reviewed without access
> to a live Sage runtime (only plain CPython 3 was available), so only `py_compile` syntax
> checks were run. Please execute the verification suites (`run_all_tests()` in each file,
> also triggered by `__main__`) in a real SageMath install before trusting numeric output —
> in particular the Gauss–Legendre node/weight construction (`gen_legendre_P` + `.roots()`)
> and the POSIX-signal-based `alarm`/`cancel_alarm` timeout guards.

---

## `g2_truncated_sextic_potential.py`

Implements Paper 1's closed-form results for
`V = mu^2*I2 + kappa*I2^2 + lambda*I6` (and its degree-8 stabilized extension
`V_stab = V + eta*I2^4`):

- `short_root_radii`, `long_root_radius` — nonzero critical radii (Theorem 3).
- `short_root_hessian`, `long_root_hessian` — normal-slice Hessian eigenvalues
  (Propositions 6.1/6.2), each with an explicit singlet/triplet multiplicity split.
- `crossing_kappa_star` — the exact energy-crossing condition
  `kappa* = -12^(1/4) * sqrt(lambda * mu^2)` (Theorem 5).
- `stabilized_critical_radius` — exact cubic solve (Cardano radical form or trigonometric
  form, chosen by discriminant sign) for the stabilized potential (Eq. 8.1–8.3),
  alarm-guarded against runaway symbolic evaluation.
- `eta_crit` — exact critical stabilizing coupling (Eq. 8.8) at which the long-root branch
  becomes a genuine local minimum.
- `global_vacuum_inequality` — the exact inequality (Eq. 8.9) for global-vacuum status.

**`TestAlgebraicInvariants`** (50 random samples per test, bounded parameter ranges):
stationarity of all critical radii, Hessian eigenvalues cross-checked against finite-difference
second derivatives, the crossing condition verified by direct energy comparison, and the
stabilized cubic root verified against Eq. 8.1.

Running the file's `__main__` block also reproduces the paper's worked numerical example
`(mu^2, kappa, lambda) = (1, -2, 1)`.

---

## `g2_stabilized_information_geometry.py`

Implements Paper 2's Gibbs-measure model
`V_reg = mu^2*I2 + kappa*I2^2 + lambda*I6 + nu*I2^3` (`nu > |lambda|`):

- `kappa_c`, `kappa_coalescence`, `u_plus`, `ground_state_energy` — the fully rigorous
  exact phase-transition layer (Sections 3–4).
- `A_curvature` — the broken-phase Fisher-metric coefficient `A(kappa)` (Theorem 6.4).
- `constant_C0`, `constant_C1` — the closed-form Laplace-expansion coefficients
  (Eq. C0expl/C1expl), reported up to the paper's own unspecified normalization constant `c0`.
- `partition_function_reduced`, `fisher_metric_numeric` — a **from-scratch numerical**
  evaluation of the finite-`beta` partition function and Fisher metric (custom adaptive
  Gauss–Legendre quadrature, no external numeric libraries), used to independently
  cross-check the paper's asymptotic claims rather than merely restate them.
- `thermodynamic_length_integral` — numerically verifies the universal crossover length
  constant `pi` (Theorem thm:length).

**`TestAlgebraicInvariants`** (20 random samples per test — capped lower than Paper 1's suite
because each test here involves numerical integration, not just algebra): ground-state
continuity at `kappa_c`, stationarity of `u_+`, `A(kappa)` against a finite-difference envelope
derivative, the `pi` thermodynamic-length identity, and a directional (not asymptotically exact)
check that the numeric Fisher metric approaches `beta * A(kappa)` as `beta` grows.

---

## Design notes / conventions followed

- **No `assert`** for validation — all preconditions raise explicit `ValueError` /
  `NonPhysicalBranchError` / `NonPhysicalRegimeError`.
- **Coercive validation** (`expected_ring.has_coerce_map_from(...)` / explicit cast) instead of
  strict `parent() ==` checks.
- **Frozen dataclasses** (`G2PotentialConfig`, `RegularizedG2Config`) hold all structural
  parameters immutably.
- **`logging`** (module-level logger) instead of `print` for diagnostics/warnings.
- **Deterministic seeding** (`sage.set_random_seed`) in every randomized test.
- **Heavy-computation timeout guards** (`sage.alarm` / `AlarmInterrupt`) around the stabilized
  cubic solve and the adaptive quadrature.
- Each public function's docstring states an explicit `Complexity:` line.

## Known limitations

- `constant_C0` / `constant_C1` in Paper 2 depend on an unspecified positive normalization
  constant `c0` (fixed by the Killing-form measure normalization, which the paper does not pin
  down numerically); it defaults to `1` here. Only ratios such as `C1(kappa)/C0` are
  normalization-independent, as the paper itself notes.
- The numeric partition-function integrator in Paper 2 is a plain nested Gauss–Legendre /
  bisection scheme, adequate for the exponentially peaked but smooth integrands here — it is
  not a general-purpose oscillatory-integral solver.
- Symbolic (`sage.SR`) branches of a few functions return the general closed form without
  re-deriving the existence/positivity conditions that the numeric (`RealField`) branches check
  explicitly; those conditions should be verified separately if you substitute symbolic results
  into a downstream numeric pipeline.
