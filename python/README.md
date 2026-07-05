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

## Verification status

Every formula in every module was cross-checked line-by-line against the
corresponding paper (not just run — actually diffed against the paper's
stated equations) and against a from-scratch execution of both the
self-checks and the full Hypothesis suites in a clean environment:

| Suite | Result |
|---|---|
| `test_paper1_invariants.py` | **9/9 passed** (stable across repeated unseeded runs) |
| `test_paper2_invariants.py` | **8/8 passed** (stable across repeated unseeded runs) |
| `test_paper3_invariants.py` | **8/8 passed** (stable across 8 independent unseeded runs, after fixing two tolerance bugs — see "Known subtleties" below) |

Formulas confirmed to match their paper exactly, including (non-exhaustive):
`kappa_c`, `kappa_coal`, `u_plus`, `C0`, `C1(kappa)` (Eq. C0expl/C1expl),
the uniform partition-function expansion (Thm 5.2), `A(kappa)` (Thm 6.3),
the Fisher metric in all three regimes (Thms 6.3–6.5), the KL divergence
in all three regimes (Thms 7.1–7.3/7.5), the short/long-root radii and
Hessian eigenvalues (Thm 1, Props 2–3), the exact crossing point
`kappa* = -12^(1/4)*sqrt(lambda*mu2)` (Thm "Exact crossing condition"),
the stabilized-cubic `p, q` coefficients, `eta_crit` (Eq. 8.8), the
global-vacuum inequality (Eq. 8.9), and the restriction identity / ι
bijection / pullback tensor (Thm 3.1, §4) in Paper 3.

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
- **Test suite fix (`test_paper2_invariants.py`, T1/T2):** the original
  code used `pytest.skip(...)` inside a Hypothesis `@given`-decorated
  test body to guard against parameter draws outside the short-root
  existence domain (`kappa**2 <= 3*lambda_*mu2`). Because Hypothesis
  tries small/boundary-value examples first, the very first draw
  frequently landed in the excluded region, and a `Skipped` exception
  raised mid-exploration aborts the *entire* test rather than just that
  one example — so T1 and T2 were being reported as passing/skipped
  while never actually exercising a single valid case. Fixed by
  replacing `pytest.skip(...)` with Hypothesis's `assume(...)`, which
  discards the invalid draw and continues searching. Confirmed with a
  standalone check that all 50 generated examples per test now execute
  the assertion body (0 did before the fix).
- **Test suite fix (`test_paper3_invariants.py`, T3):** the injectivity
  check compared `iota(a,c1)` and `iota(a,c2)` against a fixed absolute
  separation threshold, but `db/dc = 4c → 0` as `c → 0`, so two distinct
  `c` values near the origin can legitimately map to `b` values closer
  than any fixed threshold even though `iota` remains injective there.
  Fixed by only asserting separation once `(c1+c2)` is bounded away from
  that degenerate point (`> 1e-3`), which is where the map is genuinely
  bi-Lipschitz; verified stable across 8 independent unseeded Hypothesis
  runs after the fix (it failed intermittently before).
- **Test suite fix (`test_paper3_invariants.py`, T7):** the `F_b` cutoff
  used to pick between the "should coincide" and "should differ"
  branches (`1e-7`) and the tolerance used inside the "coincide" branch
  (also `1e-7`) were set independently, but the correction term the test
  is checking scales like `~6*a*F_b` (up to `18*F_b` for `a<=3`) — so an
  `F_b` just under the cutoff could produce a difference just over the
  tolerance. Fixed by widening the "coincide" tolerance to `1e-4`
  (comfortably above the worst-case `~1.8e-5` correction at the `1e-6`
  cutoff) while keeping the "differs" branch's `1e-8` tolerance, which
  the correction term clears easily whenever `F_b` exceeds the cutoff.
