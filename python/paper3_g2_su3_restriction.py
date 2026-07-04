"""
paper3_g2_su3_restriction.py

Production implementation of:
"A Note on the Restriction of G2 Invariants to SU(3)" (M. U. Javed, 2026).

Implements, with full type hints and structured logging:
  * The primitive Weyl invariants of A2=su(3) and G2 on their respective
    rank-2 Cartan subalgebras: I2^(A2), I3^(A2), I2^(G2), I6^(G2).
  * The algebraic restriction identity (Eq. 1 / Appendix A):
        I6^(G2) = 2*(I3^(A2))^2 - (I2^(A2))^3.
  * The induced orbit-space map iota(a, c) = (a, 2c^2 - a^3) and its
    inverse on the half-chamber, together with the domain predicates for
    the two orbit-space cones D_{A2}, D_{A2}^+, D_{G2} (Theorem 3.1).
  * The Jacobian of iota and the pullback of an ambient Hessian tensor
    g^F on D_{G2} to a tensor g^Phi on D_{A2}^+ (Section 4), explicitly
    distinguished from the (different) Hessian of the composed scalar
    function Phi(a,c) = F(a, 2c^2-a^3).

Backend selection
-----------------
Every object here is an elementary closed-form polynomial map between
R^2 domains together with one small (2x2) linear-algebra pullback. No
automatic differentiation, optimization, or large-matrix machinery is
required by the theory; `numpy`/`math` at float64 precision is the
correct and sufficient backend. Finite-difference Hessians are used
*only* inside the test suite, to independently cross-check the paper's
closed-form pullback formulas against the generic tensor transformation
law J^T g J -- this is a verification device, not part of the theory.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Primitive Weyl invariants on the two Cartan subalgebras
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class A2Invariants:
    """Primitive SU(3)=A2 Weyl invariants (a, c) = (I2, I3) at a Cartan point.

    Attributes:
        a: I2^(A2) = r^2 >= 0.
        c: I3^(A2) = r^3 * cos(3*theta).
    """

    a: float
    c: float

    def __post_init__(self) -> None:
        if self.a < 0.0:
            raise ValueError(f"a = I2^(A2) must be >= 0; got {self.a}")


@dataclass(frozen=True)
class G2Invariants:
    """Primitive G2 Weyl invariants (a, b) = (I2, I6) at a Cartan point.

    Attributes:
        a: I2^(G2) = r^2 >= 0.
        b: I6^(G2) = r^6 * cos(6*theta).
    """

    a: float
    b: float

    def __post_init__(self) -> None:
        if self.a < 0.0:
            raise ValueError(f"a = I2^(G2) must be >= 0; got {self.a}")


def a2_invariants_from_polar(r: float, theta: float) -> A2Invariants:
    """Compute (I2^(A2), I3^(A2)) = (r^2, r^3*cos(3*theta)) from polar coordinates.

    Complexity:
        Time: O(1). Space: O(1).
    """
    if r < 0.0:
        raise ValueError(f"r must be >= 0; got {r}")
    return A2Invariants(a=r**2, c=r**3 * math.cos(3.0 * theta))


def g2_invariants_from_polar(r: float, theta: float) -> G2Invariants:
    """Compute (I2^(G2), I6^(G2)) = (r^2, r^6*cos(6*theta)) from polar coordinates.

    Note: this uses the *same* polar angle theta as the A2 restriction,
    matching the standard embedding SU(3) subset G2 used in the paper
    (the A2 maximal torus coincides with the G2 Cartan subalgebra under
    that embedding).

    Complexity:
        Time: O(1). Space: O(1).
    """
    if r < 0.0:
        raise ValueError(f"r must be >= 0; got {r}")
    return G2Invariants(a=r**2, b=r**6 * math.cos(6.0 * theta))


def restriction_identity_residual(r: float, theta: float) -> float:
    """Residual of Eq. (1): I6^(G2) - [2*(I3^(A2))^2 - (I2^(A2))^3].

    Should vanish identically (this is an exact trigonometric identity,
    cos(6*theta) = 2*cos^2(3*theta) - 1, not an approximation).

    Complexity:
        Time: O(1). Space: O(1).
    """
    a2 = a2_invariants_from_polar(r, theta)
    g2 = g2_invariants_from_polar(r, theta)
    rhs = 2.0 * a2.c**2 - a2.a**3
    return g2.b - rhs


# ---------------------------------------------------------------------------
# 2. Orbit-space cones and domain predicates
# ---------------------------------------------------------------------------
def in_A2_cone(a: float, c: float, tol: float = 1e-9) -> bool:
    """True iff (a,c) lies in the full SU(3) orbit-space cone D_{A2} = {a>=0, |c|<=a^{3/2}}."""
    if a < -tol:
        return False
    a_clipped = max(a, 0.0)
    return abs(c) <= a_clipped ** 1.5 + tol


def in_A2_half_chamber(a: float, c: float, tol: float = 1e-9) -> bool:
    """True iff (a,c) lies in the half-chamber D_{A2}^+ = {a>=0, 0<=c<=a^{3/2}}."""
    return c >= -tol and in_A2_cone(a, c, tol=tol)


def in_G2_cone(a: float, b: float, tol: float = 1e-9) -> bool:
    """True iff (a,b) lies in the G2 orbit-space cone D_{G2} = {a>=0, |b|<=a^3}."""
    if a < -tol:
        return False
    a_clipped = max(a, 0.0)
    return abs(b) <= a_clipped**3 + tol


# ---------------------------------------------------------------------------
# 3. The induced map iota and its inverse on the half-chamber (Theorem 3.1)
# ---------------------------------------------------------------------------
def iota(a: float, c: float) -> Tuple[float, float]:
    """The orbit-space map iota(a,c) = (a, 2*c^2 - a^3).

    Well-defined on all of D_{A2} (it is even in c), but only injective
    on the half-chamber D_{A2}^+ (Theorem 3.1).

    Complexity:
        Time: O(1). Space: O(1).
    """
    if a < 0.0:
        raise ValueError(f"a must be >= 0; got {a}")
    return a, 2.0 * c**2 - a**3


def iota_inverse(a: float, b: float) -> Tuple[float, float]:
    """Inverse of iota restricted to the half-chamber D_{A2}^+ -> D_{G2}.

    Given (a,b) in D_{G2}, returns the unique (a,c) in D_{A2}^+ (c >= 0)
    with iota(a,c) = (a,b), namely c = sqrt((b + a^3) / 2).

    Raises:
        ValueError: if (a,b) is not in D_{G2} (up to tolerance), i.e. if
            a < 0 or |b| > a^3, in which case no real non-negative c exists.

    Complexity:
        Time: O(1). Space: O(1).
    """
    if not in_G2_cone(a, b):
        raise ValueError(
            f"(a,b)=({a},{b}) is not in the G2 cone D_{{G2}} = {{a>=0, |b|<=a^3}}; "
            "no preimage under iota|_{D_A2^+} exists."
        )
    radicand = b + a**3
    # Guard against tiny negative values from floating-point roundoff at the boundary.
    radicand = max(radicand, 0.0)
    c = math.sqrt(radicand / 2.0)
    return a, c


def verify_bijection_bounds(a: float, c: float) -> Tuple[float, float, float]:
    """Return (lower_bound, image_b, upper_bound) demonstrating -a^3 <= iota(a,c).b <= a^3
    for c in [0, a^{3/2}] (the inequality chain proved in Section 3).

    Complexity:
        Time: O(1). Space: O(1).
    """
    if not in_A2_half_chamber(a, c):
        raise ValueError(f"(a,c)=({a},{c}) is not in the half-chamber D_A2^+.")
    _, b = iota(a, c)
    return -(a**3), b, a**3


# ---------------------------------------------------------------------------
# 4. Jacobian of iota and pullback of the ambient Hessian tensor (Section 4)
# ---------------------------------------------------------------------------
def jacobian_iota(a: float, c: float) -> np.ndarray:
    """Jacobian matrix J_{(a,c)} = d(iota)/d(a,c) of iota(a,c) = (a, 2c^2 - a^3).

        J = [[     1,   0],
             [ -3a^2,  4c]]

    Complexity:
        Time: O(1). Space: O(1) (returns a 2x2 array).
    """
    return np.array([[1.0, 0.0], [-3.0 * a**2, 4.0 * c]], dtype=np.float64)


def pullback_hessian_closed_form(
    a: float, c: float, F_aa: float, F_ab: float, F_bb: float
) -> Tuple[float, float, float]:
    """Closed-form pullback g^Phi = J^T g^F J of the ambient Hessian tensor g^F
    (with components F_aa, F_ab, F_bb on D_{G2}) to D_{A2}^+ (Section 4, Eqs. 2-4).

    IMPORTANT (per the paper's own caveat): this is the pullback of the
    ambient *tensor* g^F, not the Hessian of the composed scalar function
    Phi(a,c) = F(a, 2c^2-a^3) -- the latter would include extra terms
    from first derivatives of F contracted against second derivatives of
    iota (see `composed_function_hessian` below for that different object).

    Returns:
        (g_aa, g_ac, g_cc).

    Complexity:
        Time: O(1). Space: O(1).
    """
    g_aa = F_aa - 6.0 * a**2 * F_ab + 9.0 * a**4 * F_bb
    g_ac = 4.0 * c * (F_ab - 3.0 * a**2 * F_bb)
    g_cc = 16.0 * c**2 * F_bb
    return g_aa, g_ac, g_cc


def pullback_hessian_via_jacobian(
    a: float, c: float, F_aa: float, F_ab: float, F_bb: float
) -> Tuple[float, float, float]:
    """Generic tensor-transformation computation g^Phi = J^T g^F J via explicit
    2x2 matrix multiplication, used to cross-check `pullback_hessian_closed_form`.

    Complexity:
        Time: O(1) (constant-size 2x2 matrix ops). Space: O(1).
    """
    J = jacobian_iota(a, c)
    gF = np.array([[F_aa, F_ab], [F_ab, F_bb]], dtype=np.float64)
    g_phi = J.T @ gF @ J
    return float(g_phi[0, 0]), float(g_phi[0, 1]), float(g_phi[1, 1])


def composed_function_hessian(
    a: float,
    c: float,
    F_a: float,
    F_b: float,
    F_aa: float,
    F_ab: float,
    F_bb: float,
) -> Tuple[float, float, float]:
    """Full Hessian of the composed scalar function Phi(a,c) = F(a, 2c^2 - a^3),
    INCLUDING the first-derivative correction terms that the pullback tensor
    g^Phi in `pullback_hessian_closed_form` deliberately omits.

    By the chain rule for second derivatives of a composition with
    b(a,c) = 2c^2 - a^3 (b_a=-3a^2, b_c=4c, b_aa=-6a, b_cc=4, b_ac=0):

        Phi_aa = F_aa + 2*F_ab*b_a + F_bb*b_a^2 + F_b*b_aa
        Phi_ac = F_ab*b_c + F_bb*b_a*b_c + F_b*b_ac  (b_ac=0)
        Phi_cc = F_bb*b_c^2 + F_b*b_cc

    Complexity:
        Time: O(1). Space: O(1).
    """
    b_a, b_c = -3.0 * a**2, 4.0 * c
    b_aa, b_ac, b_cc = -6.0 * a, 0.0, 4.0

    Phi_aa = F_aa + 2.0 * F_ab * b_a + F_bb * b_a**2 + F_b * b_aa
    Phi_ac = F_ab * b_c + F_bb * b_a * b_c + F_b * b_ac
    Phi_cc = F_bb * b_c**2 + F_b * b_cc
    return Phi_aa, Phi_ac, Phi_cc


# ---------------------------------------------------------------------------
# 5. Module self-check entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # 1. Verify the restriction identity over a grid of (r, theta).
    max_residual = 0.0
    for r in np.linspace(0.0, 3.0, 7):
        for theta in np.linspace(0.0, 2.0 * math.pi, 13):
            residual = restriction_identity_residual(float(r), float(theta))
            max_residual = max(max_residual, abs(residual))
    logger.info("Max |I6 - (2*I3^2 - I2^3)| over test grid: %.3e (should be ~0)", max_residual)

    # 2. Demonstrate the bijection on a sample point.
    a, c = 2.0, 1.5
    lower, b, upper = verify_bijection_bounds(a, c)
    logger.info("iota(a=%.2f, c=%.2f) = (%.2f, %.4f); bound [-a^3, a^3] = [%.2f, %.2f]",
                a, c, a, b, lower, upper)
    a_back, c_back = iota_inverse(a, b)
    logger.info("iota_inverse(a=%.2f, b=%.4f) = (%.2f, %.4f) (should recover c=%.2f)",
                a, b, a_back, c_back, c)

    # 3. Cross-check the closed-form pullback metric against J^T g J.
    F_aa, F_ab, F_bb = 1.3, -0.7, 2.1
    closed = pullback_hessian_closed_form(a, c, F_aa, F_ab, F_bb)
    via_jac = pullback_hessian_via_jacobian(a, c, F_aa, F_ab, F_bb)
    logger.info("Pullback (closed form) = %s", closed)
    logger.info("Pullback (via Jacobian) = %s (should match)", via_jac)


# requirements.txt
# numpy>=1.26.0
# hypothesis>=6.100.0
# pytest>=8.0.0
