# G2-Invariant Sextic Potential — Reference Implementations

Three independent, self-contained Python packages translating the closed-form
results of three companion papers on the exceptional Lie algebra `g2` (and
its relation to `su(3)`) into tested, production-style code.

| File | Paper | Contents |
|---|---|---|
| `paper1_g2_information_geometry.py` | *Information Geometry of a Coercively Regularized G2-Invariant Sextic Potential* | Reduced cubic potential, ground-state energy & phase transition, rigorous partition-function asymptotics, Fisher metric (broken/unbroken/crossover), KL divergence, thermodynamic length |
| `test_paper1_invariants.py` | — | Hypothesis property-based test suite for the above |
| `paper2_g2_critical_orbits.py` | *Exact Critical Orbit Classification of a Truncated G2-Invariant Sextic Potential* | Exact critical radii, normal-slice Hessian spectrum, exact energy-crossing condition, stabilized (η) potential with a Cardano cubic solver |
| `test_paper2_invariants.py` | — | Hypothesis property-based test suite for the above |
| `paper3_g2_su3_restriction.py` | *A Note on the Restriction of G2 Invariants to SU(3)* | The A2/G2 primitive invariants, the restriction identity I6 = 2·I3² − I2³, the induced orbit-space bijection ι(a,c) = (a, 2c²−a³) between the SU(3) half-chamber and the full G2 cone, and the pullback of the ambient Hessian tensor under ι |
| `test_paper3_invariants.py` | — | Hypothesis property-based test suite for the above |

---

## Quick start

```bash
pip install -r requirements.txt   # see bottom of each .py file for the exact list

# Run each module's built-in self-check (prints a small worked example, e.g.
# reproduces Paper 2's numerical example: u_s,+=1, u_s,-=1/3, u_l≈0.215, ...)
python paper1_g2_information_geometry.py
python paper2_g2_critical_orbits.py
python paper3_g2_su3_restriction.py

# Run the full verification suites
pytest test_paper1_invariants.py -v
pytest test_paper2_invariants.py -v
pytest test_paper3_invariants.py -v
```

## Requirements

```
numpy>=1.26.0
scipy>=1.11.0        # paper1 only (thermodynamic-length quadrature)
hypothesis>=6.100.0
pytest>=8.0.0
```

## What's implemented vs. what's a documented approximation

Every closed-form formula in each paper (critical radii, Hessian
eigenvalues, energy-crossing conditions, the C0/C1 asymptotic
coefficients, the logistic crossover profile, the restriction identity,
the orbit-space bijection, the pullback tensor, etc.) is implemented as
an exact algebraic function — no fitting, no placeholders. A few points
are called out explicitly in the code/docstrings because the papers
themselves only prove bounds (not closed leading terms) or explicitly
flag a subtlety there:

- **Paper 1, unbroken-phase Fisher metric / KL divergence:** the paper
  proves `O(beta^-2)` / `O(beta^-2)` bounds (Theorems 6.2, 7.2) but does
  not give an explicit leading coefficient, so the code returns `0.0` as
  the leading-order point estimate and logs why.
- **Paper 1, Jacobian constant `c0`:** the Weyl-integration normalization
  constant is, per the paper, irrelevant to every reported observable
  (it cancels in all `C1/C0` ratios). It is fixed to `1.0` and exposed as
  a config field for transparency.
- **Paper 3, pullback tensor vs. composed-function Hessian:** the paper
  explicitly warns that the pullback `g^Φ = JᵀgᶠJ` of the *ambient tensor*
  is **not** the Hessian of the composed scalar `Φ(a,c)=F(a,2c²−a³)`.
  The code implements both objects separately
  (`pullback_hessian_closed_form` vs. `composed_function_hessian`) and a
  dedicated test (`test_T7`) confirms they generically differ whenever
  `F_b ≠ 0`, and coincide exactly when `F_b = 0`.

## Verified numerical checks

- Paper 2's worked example `(mu2, kappa, lambda_) = (1, -2, 1)` reproduces
  the paper's stated values exactly: `u_s,+ = 1`, `u_s,- = 1/3`,
  `u_l ≈ 0.215`, `V_s,+ = 0`, `V_l ≈ 0.113`, and the radius ratio
  `u_s,+/u_l → 2+√3` exactly at the crossing point `kappa*`.
- Paper 1's local thermodynamic length numerically converges toward `π`
  as `beta → ∞` (e.g. `3.06 → 3.09 → 3.11 → 3.11` across increasing
  `beta`), matching the paper's Theorem 7.4.
- Paper 3's restriction identity `I6 = 2·I3² − I2³` holds to floating-point
  exactness (residual `0.000e+00`) over a full grid of `(r, theta)`, and
  `ι` composed with its inverse recovers the original `(a,c)` and `(a,b)`
  pairs to `~1e-9` precision.

## Known subtleties fixed during development (kept here for transparency)

- The finite-`beta` crossover peak in Paper 1 is *not* centered exactly
  at `kappa_c`; it is displaced by an `O(log(beta)/beta)` shift coming
  from the `5*log(beta) - log(C1/C0)` term in the paper's own crossover
  variable `u`. `kappa_crossover_center()` computes this shift exactly
  rather than approximating the window center by `kappa_c`.
- Paper 2's stabilized-potential cubic solver special-cases the
  discriminant-≈0 boundary (which is exactly where `eta = eta_crit`
  lives): the generic one-real-root Cardano sum only returns the simple
  root there and silently drops the physically relevant double root.
- Paper 3's `iota_inverse` clips tiny negative round-off in
  `b + a^3` before taking the square root, so boundary points
  `b = -a^3` don't spuriously raise a domain error.

