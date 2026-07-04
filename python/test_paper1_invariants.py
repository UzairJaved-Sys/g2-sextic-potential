"""
test_paper1_invariants.py

Property-based verification suite (Hypothesis-style) for
paper1_g2_information_geometry.py. Isolated from the core execution
logic per the "Test Isolation" rule.

Each test targets a specific theorem/invariant asserted in the paper:

  T1  Coercivity / finiteness  -> Theorem 4.1 (existence of Z)
  T2  Continuity of E0 at kappa_c, and E0 <= 0 always -> Theorem 3.2
  T3  u_+(kappa_c) = sqrt(mu2/Delta) exactly            -> Theorem 3.2
  T4  E0(kappa_c) = 0 exactly                            -> Theorem 3.2
  T5  A(kappa) > 0 on the broken phase                   -> Theorem 6.1
  T6  Fisher metric is non-negative wherever defined     -> general IG fact
  T7  KL divergence is non-negative (to leading order,
      broken-phase quadratic form)                       -> Thm 7.1 + IG
  T8  Local thermodynamic length -> pi as beta -> infinity -> Theorem 7.4
  T9  Robustness to pathological / extreme inputs (NaN, inf, huge beta)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from paper1_g2_information_geometry import (
    G2PotentialConfig,
    Phase,
    curvature_A,
    fisher_metric_kappakappa,
    ground_state_energy,
    local_thermodynamic_length,
    partition_function_asymptotic,
    phase_of,
    u_plus,
)


# Reasonable, physically valid parameter strategies (avoid degenerate near-zero cases
# that would trigger legitimate ValueErrors rather than genuine bugs).
mu2_strategy = st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
lambda_strategy = st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False)
extra_delta_strategy = st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)


def _make_cfg(mu2: float, lambda_: float, extra_delta: float) -> G2PotentialConfig:
    """Build a valid config with nu = lambda_ + extra_delta so Delta = extra_delta > 0."""
    return G2PotentialConfig(mu2=mu2, lambda_=lambda_, nu=lambda_ + extra_delta)


class TestMathematicalInvariants:
    """Isolated Hypothesis test suite validating Paper 1's theorems numerically."""

    @settings(max_examples=50, deadline=1000)
    @given(mu2=mu2_strategy, lambda_=lambda_strategy, extra_delta=extra_delta_strategy)
    def test_T1_partition_function_finite_and_positive(self, mu2, lambda_, extra_delta):
        cfg = _make_cfg(mu2, lambda_, extra_delta)
        beta = 100.0
        for kappa in (cfg.kappa_c - 2.0, cfg.kappa_c, cfg.kappa_c + 2.0):
            Z = partition_function_asymptotic(cfg, beta, kappa)
            assert math.isfinite(Z), f"Z not finite at kappa={kappa}"
            assert Z > 0.0, f"Z must be strictly positive (Theorem 4.1); got {Z}"

    @settings(max_examples=50, deadline=1000)
    @given(mu2=mu2_strategy, lambda_=lambda_strategy, extra_delta=extra_delta_strategy)
    def test_T2_ground_state_energy_nonpositive_and_continuous(self, mu2, lambda_, extra_delta):
        cfg = _make_cfg(mu2, lambda_, extra_delta)
        kc = cfg.kappa_c
        for kappa in np.linspace(kc - 3.0, kc + 3.0, 13):
            E0 = ground_state_energy(cfg, float(kappa))
            assert E0 <= 1e-9, f"E0 must be <=0 everywhere; got {E0} at kappa={kappa}"

        # continuity: E0 just left of kappa_c should approach 0
        eps = 1e-6
        E0_left = ground_state_energy(cfg, kc - eps)
        E0_right = ground_state_energy(cfg, kc + eps)
        assert E0_right == 0.0
        assert abs(E0_left - 0.0) < 1e-3, "E0 should be continuous (-> 0) approaching kappa_c"

    @settings(max_examples=50, deadline=1000)
    @given(mu2=mu2_strategy, lambda_=lambda_strategy, extra_delta=extra_delta_strategy)
    def test_T3_u_plus_at_kappa_c_matches_closed_form(self, mu2, lambda_, extra_delta):
        cfg = _make_cfg(mu2, lambda_, extra_delta)
        up_at_kc = u_plus(cfg, cfg.kappa_c)
        expected = math.sqrt(cfg.mu2 / cfg.Delta)
        assert math.isclose(up_at_kc, expected, rel_tol=1e-9), (
            f"u_+(kappa_c) should equal sqrt(mu2/Delta) exactly; "
            f"got {up_at_kc}, expected {expected}"
        )

    @settings(max_examples=50, deadline=1000)
    @given(mu2=mu2_strategy, lambda_=lambda_strategy, extra_delta=extra_delta_strategy)
    def test_T4_ground_state_energy_vanishes_at_kappa_c(self, mu2, lambda_, extra_delta):
        cfg = _make_cfg(mu2, lambda_, extra_delta)
        # ground_state_energy returns exactly 0 at kappa_c by the >= branch;
        # additionally verify f(u_+(kappa_c)) itself -> 0 (matching Theorem 3.2).
        up = u_plus(cfg, cfg.kappa_c)
        f_val = cfg.mu2 * up + cfg.kappa_c * up**2 + cfg.Delta * up**3
        assert abs(f_val) < 1e-8, f"f(u_+(kappa_c)) should vanish; got {f_val}"

    @settings(max_examples=50, deadline=1000)
    @given(mu2=mu2_strategy, lambda_=lambda_strategy, extra_delta=extra_delta_strategy)
    def test_T5_curvature_A_positive_in_broken_phase(self, mu2, lambda_, extra_delta):
        cfg = _make_cfg(mu2, lambda_, extra_delta)
        for offset in (0.5, 1.5, 3.0):
            kappa = cfg.kappa_c - offset
            A = curvature_A(cfg, kappa)
            assert A > 0.0, f"A(kappa) must be > 0 in the broken phase; got {A} at kappa={kappa}"

    @settings(max_examples=50, deadline=1000)
    @given(mu2=mu2_strategy, lambda_=lambda_strategy, extra_delta=extra_delta_strategy)
    def test_T6_fisher_metric_nonnegative(self, mu2, lambda_, extra_delta):
        cfg = _make_cfg(mu2, lambda_, extra_delta)
        beta = 200.0
        for kappa in (cfg.kappa_c - 2.0, cfg.kappa_c, cfg.kappa_c + 2.0):
            g = fisher_metric_kappakappa(cfg, beta, kappa)
            assert g >= 0.0, f"Fisher metric must be non-negative; got {g} at kappa={kappa}"

    @settings(max_examples=30, deadline=2000)
    @given(mu2=mu2_strategy, lambda_=lambda_strategy, extra_delta=extra_delta_strategy)
    def test_T7_kl_divergence_nonnegative_broken_phase(self, mu2, lambda_, extra_delta):
        cfg = _make_cfg(mu2, lambda_, extra_delta)
        beta = 200.0
        kappa = cfg.kappa_c - 2.0
        kappa_prime = kappa + 0.01  # small perturbation, both still broken
        from paper1_g2_information_geometry import kl_divergence

        kl = kl_divergence(cfg, beta, kappa, kappa_prime)
        assert kl >= -1e-9, f"KL divergence must be >= 0; got {kl}"

    def test_T8_thermodynamic_length_converges_to_pi(self):
        cfg = G2PotentialConfig(mu2=1.0, lambda_=1.0, nu=2.5)
        betas = [1.0e4, 1.0e6, 1.0e8]
        lengths = [local_thermodynamic_length(cfg, beta=b) for b in betas]
        for length in lengths:
            assert math.isfinite(length)
        # Should be converging toward pi; check the largest-beta case is closest.
        errors = [abs(length - math.pi) for length in lengths]
        assert errors[-1] <= errors[0] + 1e-2, (
            f"Thermodynamic length should approach pi as beta grows; errors={errors}"
        )
        assert errors[-1] < 0.5, f"Length at largest beta too far from pi: {lengths[-1]}"

    def test_T9_robustness_to_pathological_inputs(self):
        cfg = G2PotentialConfig(mu2=1.0, lambda_=1.0, nu=2.0)

        with pytest.raises(ValueError):
            G2PotentialConfig(mu2=-1.0, lambda_=1.0, nu=2.0)  # mu2 must be > 0

        with pytest.raises(ValueError):
            G2PotentialConfig(mu2=1.0, lambda_=2.0, nu=1.0)  # nu must exceed |lambda_|

        with pytest.raises(ValueError):
            partition_function_asymptotic(cfg, beta=-1.0, kappa=0.0)  # beta must be > 0

        with pytest.raises(ValueError):
            u_plus(cfg, kappa=0.0)  # 0^2 < 3*Delta*mu2 for this cfg -> no real branch

        # NaN / inf kappa should raise cleanly via downstream math domain errors,
        # not silently propagate NaN through the pipeline.
        with pytest.raises((ValueError, OverflowError)):
            _ = ground_state_energy(cfg, kappa=float("inf"))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
