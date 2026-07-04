"""
paper2_g2_critical_orbits.py

Production implementation of:
"Exact Critical Orbit Classification of a Truncated G2-Invariant Sextic
Potential" (M. U. Javed, 2026).

Implements, with full type hints and structured logging:
  * The reduced potential V_red(r, theta) = mu2*r^2 + kappa*r^4 + lambda*r^6*cos(6 theta)
    and the exact closed-form nonzero critical radii u_{s,+}, u_{s,-}, u_l
    (Theorem 3.1).
  * The exact normal-slice Hessian eigenvalue spectrum at each critical
    orbit, including multiplicities (Propositions 4-5, Theorem "Hessian
    signatures").
  * The exact analytic energy-crossing condition
        kappa* = -12^{1/4} * sqrt(lambda*mu2)
    between the larger short-root branch and the long-root branch
    (Theorem "Exact crossing condition"), derived via the golden-ratio-like
    radius relation u_{s,+}/u_l = 2 + sqrt(3).
  * The stabilized potential V_stab = mu2*I2 + kappa*I2^2 + lambda*I6 + eta*I2^4
    (Section 8): its exact quartic-root critical radii via a real-Cardano
    depressed-cubic solver, the exact critical stabilizing coupling
    eta_crit (Eq. 8.8), and the exact global-vacuum inequality (Eq. 8.9).

Backend selection
-----------------
Every quantity in this paper is a closed-form algebraic expression (root
formulas, Hessian eigenvalues, a single real cubic solved via Cardano's
trigonometric formula). There are no gradients, no iterative optimization,
and no large linear-algebra systems, so `numpy`/`math` at float64 is the
correct and sufficient backend; no JAX/Torch/GPU acceleration is
warranted by the theory.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Immutable physical configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TruncatedG2Config:
    """Immutable parameters of the truncated G2-invariant sextic potential.

    The paper's definiteness convention (Sec. 2.2) requires lambda_ > 0
    and mu2 > 0 throughout; this is enforced at construction.

    Attributes:
        mu2: mass-squared parameter mu^2 > 0.
        lambda_: sextic anisotropy coupling lambda > 0 (paper uses lambda;
            escaped with trailing underscore, Python keyword collision guard).

    Complexity:
        Time: O(1) to construct/validate. Space: O(1).
    """

    mu2: float
    lambda_: float

    def __post_init__(self) -> None:
        if self.mu2 <= 0.0:
            raise ValueError(f"mu2 must be > 0; got {self.mu2}")
        if self.lambda_ <= 0.0:
            raise ValueError(
                f"lambda_ must be > 0 per the paper's definiteness convention; got {self.lambda_}"
            )


class RootBranch(str, Enum):
    """Which of the three nonzero critical branches is being referenced."""

    SHORT_PLUS = "short_plus"   # u_{s,+}: larger short-root radius (epsilon=+1)
    SHORT_MINUS = "short_minus"  # u_{s,-}: smaller short-root radius (epsilon=+1)
    LONG = "long"               # u_l: long-root radius (epsilon=-1)


# ---------------------------------------------------------------------------
# 2. Exact nonzero critical radii  (Theorem 3.1)
# ---------------------------------------------------------------------------
def short_root_radii(cfg: TruncatedG2Config, kappa: float) -> Tuple[float, float]:
    """Exact roots u_{s,+}, u_{s,-} of 3*lambda*u^2 + 2*kappa*u + mu2 = 0.

    Existence requires kappa < 0 and kappa^2 > 3*lambda*mu2 (Theorem 3.1.1).

    Returns:
        (u_s_plus, u_s_minus), both guaranteed real and positive.

    Complexity:
        Time: O(1). Space: O(1).
    """
    disc = kappa**2 - 3.0 * cfg.lambda_ * cfg.mu2
    if kappa >= 0.0 or disc <= 0.0:
        raise ValueError(
            "Short-root branches require kappa < 0 and kappa^2 > 3*lambda_*mu2; "
            f"got kappa={kappa}, kappa^2={kappa**2:.6g}, 3*lambda_*mu2={3.0*cfg.lambda_*cfg.mu2:.6g}."
        )
    sqrt_disc = math.sqrt(disc)
    u_s_plus = (-kappa + sqrt_disc) / (3.0 * cfg.lambda_)
    u_s_minus = (-kappa - sqrt_disc) / (3.0 * cfg.lambda_)
    if u_s_plus <= 0.0 or u_s_minus <= 0.0:
        raise ValueError("Computed short-root radii are not both positive; check kappa domain.")
    return u_s_plus, u_s_minus


def long_root_radius(cfg: TruncatedG2Config, kappa: float) -> float:
    """Exact unique positive root u_l of 3*lambda*u^2 - 2*kappa*u - mu2 = 0.

    This root exists for all real kappa (discriminant kappa^2+3*lambda*mu2 > 0 always).

    Complexity:
        Time: O(1). Space: O(1).
    """
    disc = kappa**2 + 3.0 * cfg.lambda_ * cfg.mu2
    u_l = (kappa + math.sqrt(disc)) / (3.0 * cfg.lambda_)
    if u_l <= 0.0:
        raise ValueError(f"Long-root radius must be positive; got {u_l}.")
    return u_l


# ---------------------------------------------------------------------------
# 3. Exact normal-slice Hessian spectrum  (Propositions 4-5, Lemma "stabdecomp")
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HessianSpectrum:
    """Normal-slice Hessian eigenvalues at a nonzero critical orbit.

    Attributes:
        omega_singlet: radial-mode eigenvalue (multiplicity 1).
        omega_triplet: SU(2)-triplet eigenvalue (multiplicity 3, degenerate by Schur's lemma).
        orbit_dim: dimension of the G2 orbit (always 10 for these critical points,
            Lemma "stabdecomp").
    """

    omega_singlet: float
    omega_triplet: float
    orbit_dim: int = 10

    @property
    def signature(self) -> Tuple[int, int]:
        """(#positive, #negative) eigenvalues among {omega_singlet, 3x omega_triplet}."""
        vals = [self.omega_singlet] + [self.omega_triplet] * 3
        pos = sum(1 for v in vals if v > 0.0)
        neg = sum(1 for v in vals if v < 0.0)
        return pos, neg


def hessian_spectrum(cfg: TruncatedG2Config, kappa: float, branch: RootBranch) -> HessianSpectrum:
    """Exact normal-slice Hessian eigenvalues at the given critical branch.

    Short-root (epsilon=+1):  omega_1 = 8*u*(kappa + 3*lambda*u), omega_3 = -36*lambda*u^2.
    Long-root  (epsilon=-1):  omega_1' = -8*u*sqrt(kappa^2+3*lambda*mu2), omega_3' = 36*lambda*u^2.

    Complexity:
        Time: O(1). Space: O(1).
    """
    if branch is RootBranch.SHORT_PLUS:
        u, _ = short_root_radii(cfg, kappa)
        omega1 = 8.0 * u * (kappa + 3.0 * cfg.lambda_ * u)
        omega3 = -36.0 * cfg.lambda_ * u**2
    elif branch is RootBranch.SHORT_MINUS:
        _, u = short_root_radii(cfg, kappa)
        omega1 = 8.0 * u * (kappa + 3.0 * cfg.lambda_ * u)
        omega3 = -36.0 * cfg.lambda_ * u**2
    elif branch is RootBranch.LONG:
        u = long_root_radius(cfg, kappa)
        omega1 = -8.0 * u * math.sqrt(kappa**2 + 3.0 * cfg.lambda_ * cfg.mu2)
        omega3 = 36.0 * cfg.lambda_ * u**2
    else:  # pragma: no cover - exhaustive Enum
        raise ValueError(f"Unknown branch {branch}")
    return HessianSpectrum(omega_singlet=omega1, omega_triplet=omega3)


# ---------------------------------------------------------------------------
# 4. On-shell energies and the exact crossing condition  (Theorem "Exact crossing")
# ---------------------------------------------------------------------------
def on_shell_energy(cfg: TruncatedG2Config, kappa: float, u: float, epsilon: int) -> float:
    """On-shell reduced energy at a critical point: V|_crit = -kappa*u^2 - 2*lambda*epsilon*u^3.

    Complexity:
        Time: O(1). Space: O(1).
    """
    if epsilon not in (1, -1):
        raise ValueError(f"epsilon must be +1 or -1; got {epsilon}")
    return -kappa * u**2 - 2.0 * cfg.lambda_ * epsilon * u**3


def crossing_kappa_star(cfg: TruncatedG2Config) -> float:
    """Exact analytic crossing point kappa* = -12^(1/4) * sqrt(lambda*mu2) (Theorem "Exact crossing").

    At this kappa, the on-shell energy of the larger short-root branch
    u_{s,+} equals that of the long-root branch u_l.

    Complexity:
        Time: O(1). Space: O(1).
    """
    return -(12.0 ** 0.25) * math.sqrt(cfg.lambda_ * cfg.mu2)


def verify_radius_ratio_at_crossing(cfg: TruncatedG2Config, kappa_star: float) -> float:
    """Return u_{s,+}/u_l evaluated at kappa_star; theory predicts exactly 2+sqrt(3)."""
    u_s_plus, _ = short_root_radii(cfg, kappa_star)
    u_l = long_root_radius(cfg, kappa_star)
    return u_s_plus / u_l


# ---------------------------------------------------------------------------
# 5. Stabilized potential with positive eta*I2^4 term  (Section 8)
# ---------------------------------------------------------------------------
def _solve_depressed_cubic_real_roots(p: float, q: float) -> List[float]:
    """Solve z^3 + p*z + q = 0 for all real roots via Cardano/trigonometric formula.

    Handles both the one-real-root regime (discriminant >= 0, Cardano's
    formula) and the three-real-roots regime (discriminant < 0, the
    numerically stable trigonometric form), matching Eqs. (8.2)-(8.3).

    Complexity:
        Time: O(1). Space: O(1) (returns a list of at most 3 floats).
    """
    discriminant = (q / 2.0) ** 2 + (p / 3.0) ** 3

    # Boundary case D ~ 0: the cubic has a double root r and a simple root
    # s = -2r, with r = -3q/(2p) (derived by matching coefficients of
    # (z-r)^2(z-s)). The generic one-real-root Cardano sum below returns
    # only s in this regime, silently dropping the double root r -- which,
    # for this model, is frequently the physically relevant critical
    # radius (e.g. exactly at eta = eta_crit). We special-case it explicitly.
    disc_scale = max(abs(p) ** 3, q**2, 1.0)
    if abs(discriminant) < 1e-12 * disc_scale:
        if abs(p) < 1e-14:
            return [0.0]
        r = -3.0 * q / (2.0 * p)
        s = -2.0 * r
        return sorted({r, s})

    if discriminant >= 0.0:
        sqrt_disc = math.sqrt(discriminant)
        # Real cube roots (Python's ** on negative floats with fractional
        # exponent returns complex; use math.copysign + abs()**(1/3) instead).
        def real_cbrt(x: float) -> float:
            return math.copysign(abs(x) ** (1.0 / 3.0), x)

        term1 = real_cbrt(-q / 2.0 + sqrt_disc)
        term2 = real_cbrt(-q / 2.0 - sqrt_disc)
        return [term1 + term2]

    # Three real roots: trigonometric form (Eq. 8.3).
    if p >= 0.0:
        raise ValueError("Trigonometric branch requires p < 0 when discriminant < 0.")
    r = math.sqrt(-(p / 3.0) ** 3)
    cos_arg = float(np.clip(-q / (2.0 * r), -1.0, 1.0))
    phi = math.acos(cos_arg)
    roots = [
        2.0 * math.sqrt(-p / 3.0) * math.cos((phi + 2.0 * math.pi * k) / 3.0)
        for k in range(3)
    ]
    return roots


def stabilized_critical_radii(
    cfg: TruncatedG2Config, kappa: float, eta: float, epsilon: int
) -> List[float]:
    """Exact positive real roots u>0 of 4*eta*u^3 + 3*lambda*epsilon*u^2 + 2*kappa*u + mu2 = 0.

    Depresses the cubic via u = z - b/(3a) with a=4*eta, b=3*lambda*epsilon,
    c=2*kappa, d=mu2 (Eq. 8.1), solves for z via Cardano/trigonometric
    formula, then filters to strictly positive real roots of u.

    Complexity:
        Time: O(1). Space: O(1).
    """
    if eta <= 0.0:
        raise ValueError(f"eta must be > 0 for the stabilized potential; got {eta}")
    if epsilon not in (1, -1):
        raise ValueError(f"epsilon must be +1 or -1; got {epsilon}")

    lambda_ = cfg.lambda_
    mu2 = cfg.mu2

    # Depression shift u = z - b/(3a) with a=4*eta, b=3*lambda*epsilon:
    # b/(3a) = 3*lambda*epsilon / (12*eta) = lambda*epsilon / (4*eta).
    shift = (lambda_ * epsilon) / (4.0 * eta)

    # Paper's Eq. (8.1)-derivation closed forms for p, q (using epsilon^2=1,
    # epsilon^3=epsilon), given directly in terms of eta, lambda, kappa, mu2 --
    # NOT to be re-derived through a generic a,b,c,d substitution, since the
    # paper's p,q formulas already ARE the fully-substituted result.
    p = (8.0 * eta * kappa - 3.0 * lambda_**2) / (16.0 * eta**2)
    q = (lambda_**3 * epsilon - 4.0 * eta * lambda_ * kappa * epsilon + 8.0 * eta**2 * mu2) / (
        32.0 * eta**3
    )

    z_roots = _solve_depressed_cubic_real_roots(p, q)
    u_roots = [z - shift for z in z_roots]
    positive_roots = sorted(u for u in u_roots if u > 1e-12)
    if not positive_roots:
        logger.warning(
            "No positive critical radius found for eta=%.6g, kappa=%.6g, epsilon=%d",
            eta, kappa, epsilon,
        )
    return positive_roots


def radial_eigenvalue_stabilized(
    cfg: TruncatedG2Config, kappa: float, eta: float, u: float, epsilon: int
) -> float:
    """Exact radial (singlet) Hessian eigenvalue of the stabilized potential (Eq. 8.4).

    omega_1 = 8*u*(kappa + 3*lambda*epsilon*u + 6*eta*u^2).

    Complexity:
        Time: O(1). Space: O(1).
    """
    return 8.0 * u * (kappa + 3.0 * cfg.lambda_ * epsilon * u + 6.0 * eta * u**2)


def triplet_eigenvalue_stabilized(cfg: TruncatedG2Config, u: float, epsilon: int) -> float:
    """Exact triplet Hessian eigenvalue, independent of eta (Eq. 8.5): -36*lambda*epsilon*u^2."""
    return -36.0 * cfg.lambda_ * epsilon * u**2


def eta_crit(cfg: TruncatedG2Config, kappa: float) -> float:
    """Exact critical stabilizing coupling eta_crit at which the long-root
    radial eigenvalue passes through zero (Eq. 8.8).

    u* = (2*kappa + sqrt(4*kappa^2 + 9*lambda*mu2)) / (3*lambda)   (Eq. 8.7)
    eta_crit = (3*lambda*u* - kappa) / (6*u*^2)                    (Eq. 8.8)

    Complexity:
        Time: O(1). Space: O(1).
    """
    u_star = (2.0 * kappa + math.sqrt(4.0 * kappa**2 + 9.0 * cfg.lambda_ * cfg.mu2)) / (
        3.0 * cfg.lambda_
    )
    if u_star <= 0.0:
        raise ValueError(f"u* must be positive; got {u_star}")
    return (3.0 * cfg.lambda_ * u_star - kappa) / (6.0 * u_star**2)


def is_global_vacuum_long_root(cfg: TruncatedG2Config, kappa: float, eta: float, u: float) -> bool:
    """Exact global-vacuum inequality for the long-root branch (Eq. 8.9):
    the branch beats the origin iff 3*eta*u^2 - 2*lambda*u + kappa > 0.

    Complexity:
        Time: O(1). Space: O(1).
    """
    return (3.0 * eta * u**2 - 2.0 * cfg.lambda_ * u + kappa) > 0.0


# ---------------------------------------------------------------------------
# 6. Module self-check entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cfg = TruncatedG2Config(mu2=1.0, lambda_=1.0)
    kappa = -2.0

    u_s_plus, u_s_minus = short_root_radii(cfg, kappa)
    u_l = long_root_radius(cfg, kappa)
    logger.info("u_s,+ = %.6f, u_s,- = %.6f, u_l = %.6f", u_s_plus, u_s_minus, u_l)

    spec_splus = hessian_spectrum(cfg, kappa, RootBranch.SHORT_PLUS)
    spec_sminus = hessian_spectrum(cfg, kappa, RootBranch.SHORT_MINUS)
    spec_long = hessian_spectrum(cfg, kappa, RootBranch.LONG)
    logger.info("Signature u_s,+: %s (expect (1,3))", spec_splus.signature)
    logger.info("Signature u_s,-: %s (expect (0,4))", spec_sminus.signature)
    logger.info("Signature u_l:   %s (expect (3,1))", spec_long.signature)

    V_splus = on_shell_energy(cfg, kappa, u_s_plus, epsilon=1)
    V_l = on_shell_energy(cfg, kappa, u_l, epsilon=-1)
    logger.info("V_s,+ = %.6f, V_l = %.6f (paper's numerical example: 0, 0.113)", V_splus, V_l)

    kstar = crossing_kappa_star(cfg)
    ratio = verify_radius_ratio_at_crossing(cfg, kstar)
    logger.info("kappa* = %.6f, u_s+/u_l at kappa* = %.6f (theory: 2+sqrt(3)=%.6f)",
                kstar, ratio, 2.0 + math.sqrt(3.0))

    eta_c = eta_crit(cfg, kappa)
    logger.info("eta_crit(kappa=%.2f) = %.6f", kappa, eta_c)
    u_roots = stabilized_critical_radii(cfg, kappa, eta=eta_c, epsilon=-1)
    logger.info("Stabilized long-root critical radii at eta_crit: %s", u_roots)


# requirements.txt
# numpy>=1.26.0
# hypothesis>=6.100.0
# pytest>=8.0.0
