"""
test_paper2_invariants.py

Property-based verification suite (Hypothesis-style) for
paper2_g2_critical_orbits.py. Isolated from the core execution logic
per the "Test Isolation" rule.

Each test targets a specific theorem/invariant asserted in the paper:

  T1  Short/long-root radii satisfy their exact defining quadratics
      (Theorem 3.1)
  T2  Hessian signatures are exactly (1,3) / (0,4) / (3,1) for
      u_s,+ / u_s,- / u_l respectively (Theorem "Hessian signatures")
  T3  The radius ratio u_s,+/u_l equals exactly 2+sqrt(3) at kappa*
      (Theorem "Exact crossing condition", Step 4)
  T4  The two branches' on-shell energies are equal at kappa*
      (Theorem "Exact crossing condition")
  T5  Stabilized cubic roots satisfy the exact quartic-stationarity
      equation (Eq. 8.1) to numerical precision
  T6  eta_crit makes the radial eigenvalue vanish exactly (Eq. 8.6/8.8)
  T7  Triplet eigenvalue is independent of eta (Eq. 8.5)
  T8  Robustness to pathological / out-of-domain inputs
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, strategies as st

from paper2_g2_critical_orbits import (
    RootBranch,
    TruncatedG2Config,
    crossing_kappa_star,
    eta_crit,
    hessian_spectrum,
    long_root_radius,
    on_shell_energy,
    radial_eigenvalue_stabilized,
    short_root_radii,
    stabilized_critical_radii,
    triplet_eigenvalue_stabilized,
    verify_radius_ratio_at_crossing,
)


mu2_strategy = st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
lambda_strategy = st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
kappa_neg_strategy = st.floats(min_value=-10.0, max_value=-2.5, allow_nan=False, allow_infinity=False)


class TestMathematicalInvariants:
    """Isolated Hypothesis test suite validating Paper 2's theorems numerically."""

    @settings(max_examples=50, deadline=1000)
    @given(mu2=mu2_strategy, lambda_=lambda_strategy, kappa=kappa_neg_strategy)
    def test_T1_radii_satisfy_defining_quadratics(self, mu2, lambda_, kappa):
        cfg = TruncatedG2Config(mu2=mu2, lambda_=lambda_)
        if kappa**2 <= 3.0 * lambda_ * mu2:
            pytest.skip("Outside short-root existence domain for this sample.")
        u_s_plus, u_s_minus = short_root_radii(cfg, kappa)
        u_l = long_root_radius(cfg, kappa)

        for u in (u_s_plus, u_s_minus):
            residual = 3.0 * lambda_ * u**2 + 2.0 * kappa * u + mu2
            assert abs(residual) < 1e-8, f"Short-root residual too large: {residual}"

        residual_l = 3.0 * lambda_ * u_l**2 - 2.0 * kappa * u_l - mu2
        assert abs(residual_l) < 1e-8, f"Long-root residual too large: {residual_l}"

    @settings(max_examples=50, deadline=1000)
    @given(mu2=mu2_strategy, lambda_=lambda_strategy, kappa=kappa_neg_strategy)
    def test_T2_hessian_signatures_exact(self, mu2, lambda_, kappa):
        cfg = TruncatedG2Config(mu2=mu2, lambda_=lambda_)
        if kappa**2 <= 3.0 * lambda_ * mu2:
            pytest.skip("Outside short-root existence domain for this sample.")

        spec_splus = hessian_spectrum(cfg, kappa, RootBranch.SHORT_PLUS)
        spec_sminus = hessian_spectrum(cfg, kappa, RootBranch.SHORT_MINUS)
        spec_long = hessian_spectrum(cfg, kappa, RootBranch.LONG)

        assert spec_splus.signature == (1, 3), f"u_s,+ signature: {spec_splus.signature}"
        assert spec_sminus.signature == (0, 4), f"u_s,- signature: {spec_sminus.signature}"
        assert spec_long.signature == (3, 1), f"u_l signature: {spec_long.signature}"

    @settings(max_examples=50, deadline=1000)
    @given(mu2=mu2_strategy, lambda_=lambda_strategy)
    def test_T3_radius_ratio_equals_two_plus_sqrt3_at_crossing(self, mu2, lambda_):
        cfg = TruncatedG2Config(mu2=mu2, lambda_=lambda_)
        kstar = crossing_kappa_star(cfg)
        ratio = verify_radius_ratio_at_crossing(cfg, kstar)
        expected = 2.0 + math.sqrt(3.0)
        assert math.isclose(ratio, expected, rel_tol=1e-6), (
            f"u_s+/u_l at kappa* should be exactly 2+sqrt(3); got {ratio}, expected {expected}"
        )

    @settings(max_examples=50, deadline=1000)
    @given(mu2=mu2_strategy, lambda_=lambda_strategy)
    def test_T4_energies_equal_at_crossing(self, mu2, lambda_):
        cfg = TruncatedG2Config(mu2=mu2, lambda_=lambda_)
        kstar = crossing_kappa_star(cfg)
        u_s_plus, _ = short_root_radii(cfg, kstar)
        u_l = long_root_radius(cfg, kstar)
        V_s_plus = on_shell_energy(cfg, kstar, u_s_plus, epsilon=1)
        V_l = on_shell_energy(cfg, kstar, u_l, epsilon=-1)
        assert math.isclose(V_s_plus, V_l, rel_tol=1e-6, abs_tol=1e-8), (
            f"Energies should cross exactly at kappa*; got V_s,+={V_s_plus}, V_l={V_l}"
        )

    @settings(max_examples=40, deadline=1000)
    @given(
        mu2=mu2_strategy,
        lambda_=lambda_strategy,
        kappa=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        eta=st.floats(min_value=0.05, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    def test_T5_stabilized_roots_satisfy_quartic_stationarity(self, mu2, lambda_, kappa, eta):
        cfg = TruncatedG2Config(mu2=mu2, lambda_=lambda_)
        for epsilon in (1, -1):
            roots = stabilized_critical_radii(cfg, kappa, eta, epsilon)
            for u in roots:
                residual = 4.0 * eta * u**3 + 3.0 * lambda_ * epsilon * u**2 + 2.0 * kappa * u + mu2
                assert abs(residual) < 1e-6, (
                    f"Stabilized stationarity residual too large: {residual} "
                    f"(u={u}, eta={eta}, kappa={kappa}, epsilon={epsilon})"
                )

    @settings(max_examples=50, deadline=1000)
    @given(mu2=mu2_strategy, lambda_=lambda_strategy, kappa=kappa_neg_strategy)
    def test_T6_eta_crit_zeroes_radial_eigenvalue(self, mu2, lambda_, kappa):
        cfg = TruncatedG2Config(mu2=mu2, lambda_=lambda_)
        eta_c = eta_crit(cfg, kappa)
        assert eta_c > 0.0, f"eta_crit should be positive in this regime; got {eta_c}"

        u_star = (2.0 * kappa + math.sqrt(4.0 * kappa**2 + 9.0 * lambda_ * mu2)) / (3.0 * lambda_)
        omega1 = radial_eigenvalue_stabilized(cfg, kappa, eta_c, u_star, epsilon=-1)
        assert abs(omega1) < 1e-6, f"Radial eigenvalue should vanish at eta_crit; got {omega1}"

    @settings(max_examples=50, deadline=1000)
    @given(
        mu2=mu2_strategy,
        lambda_=lambda_strategy,
        u=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    def test_T7_triplet_eigenvalue_independent_of_eta(self, mu2, lambda_, u):
        cfg = TruncatedG2Config(mu2=mu2, lambda_=lambda_)
        # Triplet eigenvalue formula takes no eta argument at all by construction;
        # confirm its sign flips correctly between short (+1) and long (-1) branches.
        omega3_short = triplet_eigenvalue_stabilized(cfg, u, epsilon=1)
        omega3_long = triplet_eigenvalue_stabilized(cfg, u, epsilon=-1)
        assert omega3_short < 0.0, "Short-root triplet eigenvalue must be negative (Eq. 8.5)."
        assert omega3_long > 0.0, "Long-root triplet eigenvalue must be positive (Eq. 8.5)."
        assert math.isclose(omega3_short, -omega3_long, rel_tol=1e-9)

    def test_T8_robustness_to_pathological_inputs(self):
        cfg = TruncatedG2Config(mu2=1.0, lambda_=1.0)

        with pytest.raises(ValueError):
            TruncatedG2Config(mu2=-1.0, lambda_=1.0)  # mu2 must be > 0

        with pytest.raises(ValueError):
            TruncatedG2Config(mu2=1.0, lambda_=-1.0)  # lambda_ must be > 0

        with pytest.raises(ValueError):
            short_root_radii(cfg, kappa=0.0)  # kappa must be < 0 and satisfy discriminant

        with pytest.raises(ValueError):
            short_root_radii(cfg, kappa=-1.0)  # kappa^2=1 < 3*lambda*mu2=3: no real branch

        with pytest.raises(ValueError):
            stabilized_critical_radii(cfg, kappa=-2.0, eta=-1.0, epsilon=-1)  # eta must be > 0

        with pytest.raises(ValueError):
            stabilized_critical_radii(cfg, kappa=-2.0, eta=1.0, epsilon=0)  # invalid epsilon

        with pytest.raises(ValueError):
            on_shell_energy(cfg, kappa=-2.0, u=1.0, epsilon=2)  # invalid epsilon


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
