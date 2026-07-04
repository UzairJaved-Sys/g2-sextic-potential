"""
test_paper3_invariants.py

Property-based verification suite (Hypothesis-style) for
paper3_g2_su3_restriction.py. Isolated from the core execution logic
per the "Test Isolation" rule.

Each test targets a specific claim in the paper:

  T1  Restriction identity I6^(G2) = 2*(I3^(A2))^2 - (I2^(A2))^3 holds
      exactly for all (r, theta)                          (Eq. 1)
  T2  The image of iota on the half-chamber always satisfies
      -a^3 <= b <= a^3, with endpoints attained at c=0, c=a^{3/2}
                                                            (Section 3)
  T3  iota restricted to the half-chamber is injective     (Theorem 3.1)
  T4  iota restricted to the half-chamber is surjective onto D_{G2},
      i.e. iota_inverse is a genuine two-sided inverse      (Theorem 3.1)
  T5  iota is even in c (so it is exactly 2-to-1 off the boundary on
      the full cone, motivating the half-chamber restriction) (Sec. 3)
  T6  The closed-form pullback tensor matches the generic J^T g J
      tensor-transformation law                             (Section 4)
  T7  The pullback tensor is *not*, in general, the same as the full
      Hessian of the composed function (the paper's explicit caveat)
  T8  Robustness to pathological / out-of-domain inputs
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from paper3_g2_su3_restriction import (
    composed_function_hessian,
    in_A2_half_chamber,
    in_G2_cone,
    iota,
    iota_inverse,
    jacobian_iota,
    pullback_hessian_closed_form,
    pullback_hessian_via_jacobian,
    restriction_identity_residual,
    verify_bijection_bounds,
)


r_strategy = st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False)
theta_strategy = st.floats(min_value=0.0, max_value=2.0 * math.pi, allow_nan=False, allow_infinity=False)
a_strategy = st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False)
hessian_component_strategy = st.floats(
    min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
)


class TestMathematicalInvariants:
    """Isolated Hypothesis test suite validating Paper 3's claims numerically."""

    @settings(max_examples=100, deadline=1000)
    @given(r=r_strategy, theta=theta_strategy)
    def test_T1_restriction_identity_holds_exactly(self, r, theta):
        residual = restriction_identity_residual(r, theta)
        assert abs(residual) < 1e-9, f"Restriction identity violated: residual={residual}"

    @settings(max_examples=100, deadline=1000)
    @given(a=a_strategy, frac=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    def test_T2_image_bounds_and_endpoints(self, a, frac):
        c = frac * a**1.5  # sweeps c in [0, a^{3/2}]
        lower, b, upper = verify_bijection_bounds(a, c)
        assert lower - 1e-9 <= b <= upper + 1e-9, (
            f"Image b={b} outside bounds [{lower},{upper}] for a={a}, c={c}"
        )
        # endpoint checks
        _, b_at_zero, _ = verify_bijection_bounds(a, 0.0)
        assert math.isclose(b_at_zero, -(a**3), abs_tol=1e-9), (
            f"c=0 should map to b=-a^3; got {b_at_zero}, expected {-(a**3)}"
        )
        _, b_at_max, _ = verify_bijection_bounds(a, a**1.5)
        assert math.isclose(b_at_max, a**3, abs_tol=1e-9), (
            f"c=a^(3/2) should map to b=a^3; got {b_at_max}, expected {a**3}"
        )

    @settings(max_examples=100, deadline=1000)
    @given(
        a=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        c1_frac=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        c2_frac=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def test_T3_iota_injective_on_half_chamber(self, a, c1_frac, c2_frac):
        c1, c2 = c1_frac * a**1.5, c2_frac * a**1.5
        _, b1 = iota(a, c1)
        _, b2 = iota(a, c2)
        if abs(c1 - c2) > 1e-6:
            assert abs(b1 - b2) > 1e-9, (
                f"Distinct c1={c1}, c2={c2} on the half-chamber mapped to the same b={b1}; "
                "iota should be injective there."
            )
        else:
            assert math.isclose(b1, b2, abs_tol=1e-6)

    @settings(max_examples=100, deadline=1000)
    @given(
        a=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        b_frac=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    )
    def test_T4_iota_surjective_with_genuine_inverse(self, a, b_frac):
        b = b_frac * a**3  # sweeps b in [-a^3, a^3], i.e. all of D_{G2} at this a
        assert in_G2_cone(a, b)
        a_rec, c_rec = iota_inverse(a, b)
        assert c_rec >= -1e-9, f"Recovered c={c_rec} should be non-negative (half-chamber)."
        assert in_A2_half_chamber(a_rec, c_rec), "Recovered (a,c) must lie in the half-chamber."
        a_re, b_re = iota(a_rec, c_rec)
        assert math.isclose(a_re, a, abs_tol=1e-9)
        assert math.isclose(b_re, b, abs_tol=1e-6), (
            f"iota(iota_inverse(a,b)) should recover b; got {b_re}, expected {b}"
        )

    @settings(max_examples=50, deadline=1000)
    @given(a=a_strategy, c=st.floats(min_value=0.0, max_value=10.0, allow_nan=False))
    def test_T5_iota_even_in_c(self, a, c):
        _, b_pos = iota(a, c)
        _, b_neg = iota(a, -c)
        assert math.isclose(b_pos, b_neg, abs_tol=1e-9), (
            "iota must be even in c: iota(a,c) and iota(a,-c) should share the same image."
        )

    @settings(max_examples=50, deadline=1000)
    @given(
        a=a_strategy,
        c=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        F_aa=hessian_component_strategy,
        F_ab=hessian_component_strategy,
        F_bb=hessian_component_strategy,
    )
    def test_T6_closed_form_matches_jacobian_transform(self, a, c, F_aa, F_ab, F_bb):
        closed = pullback_hessian_closed_form(a, c, F_aa, F_ab, F_bb)
        via_jac = pullback_hessian_via_jacobian(a, c, F_aa, F_ab, F_bb)
        for name, x, y in zip(("g_aa", "g_ac", "g_cc"), closed, via_jac):
            assert math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9), (
                f"{name} mismatch: closed-form={x}, via-Jacobian={y}"
            )

    @settings(max_examples=30, deadline=1000)
    @given(
        a=st.floats(min_value=0.1, max_value=3.0, allow_nan=False),
        c=st.floats(min_value=0.1, max_value=5.0, allow_nan=False),
        F_a=hessian_component_strategy,
        F_b=hessian_component_strategy,
        F_aa=hessian_component_strategy,
        F_ab=hessian_component_strategy,
        F_bb=hessian_component_strategy,
    )
    def test_T7_pullback_tensor_differs_from_composed_hessian_in_general(
        self, a, c, F_a, F_b, F_aa, F_ab, F_bb
    ):
        pullback = pullback_hessian_closed_form(a, c, F_aa, F_ab, F_bb)
        composed = composed_function_hessian(a, c, F_a, F_b, F_aa, F_ab, F_bb)
        # They agree iff the first-derivative correction terms vanish, which
        # happens only when F_b == 0 (since b_aa=-6a!=0, b_cc=4!=0 generically).
        # For generic nonzero F_b, the two objects must differ in at least
        # one component -- this is exactly the paper's explicit caveat.
        # Note: the correction terms scale as F_b * O(a), so numerical precision
        # requires a slightly relaxed tolerance when |F_b| is very small.
        if abs(F_b) > 1e-7:
            assert not all(
                math.isclose(p, q, rel_tol=1e-8, abs_tol=1e-8)
                for p, q in zip(pullback, composed)
            ), (
                "Pullback tensor and composed-function Hessian coincided even though "
                f"F_b={F_b} != 0; the paper's caveat that these differ should hold generically."
            )
        else:
            # When F_b ≈ 0, the b_aa/b_cc correction terms are negligible and the two
            # objects should coincide to numerical precision. Use relaxed tolerance
            # to account for floating-point error propagation.
            for p, q in zip(pullback, composed):
                assert math.isclose(p, q, rel_tol=1e-7, abs_tol=1e-7)

    def test_T8_robustness_to_pathological_inputs(self):
        with pytest.raises(ValueError):
            iota(a=-1.0, c=0.0)  # a must be >= 0

        with pytest.raises(ValueError):
            iota_inverse(a=1.0, b=2.0)  # b=2 > a^3=1: outside D_G2

        with pytest.raises(ValueError):
            iota_inverse(a=-1.0, b=0.0)  # a must be >= 0

        with pytest.raises(ValueError):
            verify_bijection_bounds(a=1.0, c=-0.5)  # c must be >= 0 in the half-chamber

        # a=0, c=0 is a legitimate boundary point (the origin); should not raise.
        a0, b0 = iota(0.0, 0.0)
        assert a0 == 0.0 and b0 == 0.0

        # Jacobian at the origin should still be well-defined and finite.
        J0 = jacobian_iota(0.0, 0.0)
        assert np.all(np.isfinite(J0))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
