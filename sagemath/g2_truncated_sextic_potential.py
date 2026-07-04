"""
g2_truncated_sextic_potential.py

Exact critical-orbit classification of the truncated G2-invariant sextic
potential  V = mu^2 * I2 + kappa * I2^2 + lambda * I6  (Javed, Paper 1),
and its degree-8 stabilized extension V_stab = V + eta * I2^4.

SageMath version required: >= 10.3
Optional packages: none (pure sage.all + logging)

Pure Python 3 syntax targeting the SageMath library (no interactive
preparser macros). Import style follows the namespace-hygiene rule:
`import sage.all as sage` rather than `from sage.all import *`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import sage.all as sage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class AlgebraicComplexityError(Exception):
    """Raised when a heavy algebraic computation exceeds its timeout budget."""


class NonPhysicalBranchError(Exception):
    """Raised when a requested critical branch does not exist for the given
    parameters (e.g. discriminant negative, or root non-positive)."""


# ---------------------------------------------------------------------------
# Immutable configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class G2PotentialConfig:
    """Immutable parameter block for the truncated / stabilized G2 potential.

    Parents:
        mu2, kappa, lam, eta are elements of the Symbolic Ring (sage.SR) when
        used for exact symbolic derivation, or elements of
        sage.RealField(prec) when a concrete numeric instantiation is
        required. Both are accepted via coercive validation (see
        `_coerce_scalar`). The theory assumes lambda > 0, mu2 > 0 for the
        unstabilized model; eta > 0 is additionally required for the
        stabilized model of Section 8.

    Complexity: O(1) construction; all fields are scalars.
    """

    mu2: object
    kappa: object
    lam: object
    eta: Optional[object] = None
    prec: int = 200

    def __post_init__(self) -> None:
        if self.prec < 53:
            raise ValueError("Precision must be at least 53 bits.")


def _coerce_scalar(x, target_ring):
    """Coercive validation: prefer natural Sage coercion over strict
    parent equality, falling back to explicit casting.

    Complexity: O(1).
    """
    try:
        if hasattr(x, "parent") and target_ring.has_coerce_map_from(x.parent()):
            return target_ring(x)
        return target_ring(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot coerce {x!r} into {target_ring}.") from exc


def _safe_sqrt(value, ring):
    """Compute a square root within `ring`, raising a clear error rather
    than propagating a domain error, for non-negative real radicands.

    Complexity: O(1) (single root extraction).
    """
    if value < 0:
        raise NonPhysicalBranchError(
            f"Attempted sqrt of negative quantity {value}; branch does not exist."
        )
    return ring(value).sqrt()


# ---------------------------------------------------------------------------
# Section 3: nonzero critical orbits of the truncated potential
# ---------------------------------------------------------------------------

def short_root_radii(config: G2PotentialConfig, ring=None) -> Tuple[object, object]:
    """Return (u_{s,+}, u_{s,-}), the two short-root critical radii.

    Solves 3*lambda*u^2 + 2*kappa*u + mu^2 = 0 (Theorem 1, part 1).
    Existence requires kappa < 0 and kappa^2 > 3*lambda*mu2; this is checked
    explicitly rather than assumed.

    Complexity: O(1) ring operations (one quadratic solve).
    """
    ring = ring or sage.SR
    mu2 = _coerce_scalar(config.mu2, ring)
    kappa = _coerce_scalar(config.kappa, ring)
    lam = _coerce_scalar(config.lam, ring)

    disc = kappa ** 2 - 3 * lam * mu2
    if ring is sage.SR:
        u_plus = (-kappa + sage.sqrt(disc)) / (3 * lam)
        u_minus = (-kappa - sage.sqrt(disc)) / (3 * lam)
        return u_plus, u_minus

    if disc < 0 or kappa >= 0:
        raise NonPhysicalBranchError(
            "Short-root branch requires kappa < 0 and kappa^2 > 3*lambda*mu^2."
        )
    root = _safe_sqrt(disc, ring)
    u_plus = (-kappa + root) / (3 * lam)
    u_minus = (-kappa - root) / (3 * lam)
    return u_plus, u_minus


def long_root_radius(config: G2PotentialConfig, ring=None) -> object:
    """Return u_l, the unique positive long-root critical radius.

    Solves 3*lambda*u^2 - 2*kappa*u - mu^2 = 0 (Theorem 1, part 2).

    Complexity: O(1).
    """
    ring = ring or sage.SR
    mu2 = _coerce_scalar(config.mu2, ring)
    kappa = _coerce_scalar(config.kappa, ring)
    lam = _coerce_scalar(config.lam, ring)

    disc = kappa ** 2 + 3 * lam * mu2
    if ring is sage.SR:
        return (kappa + sage.sqrt(disc)) / (3 * lam)
    root = _safe_sqrt(disc, ring)
    return (kappa + root) / (3 * lam)


# ---------------------------------------------------------------------------
# Section 6: normal-slice Hessian spectrum (Propositions on short/long root)
# ---------------------------------------------------------------------------

def short_root_hessian(config: G2PotentialConfig, u_s, ring=None) -> Tuple[object, object]:
    """Return (omega_1, omega_3) for the short-root branch (Prop. 6.1).

    omega_1 = 8*u_s*(kappa + 3*lambda*u_s)   [multiplicity 1]
    omega_3 = -36*lambda*u_s^2               [multiplicity 3]

    Complexity: O(1).
    """
    ring = ring or sage.SR
    kappa = _coerce_scalar(config.kappa, ring)
    lam = _coerce_scalar(config.lam, ring)
    omega1 = 8 * u_s * (kappa + 3 * lam * u_s)
    omega3 = -36 * lam * u_s ** 2
    return omega1, omega3


def long_root_hessian(config: G2PotentialConfig, u_l, ring=None) -> Tuple[object, object]:
    """Return (omega_1', omega_3') for the long-root branch (Prop. 6.2).

    omega_1' = -8*u_l*sqrt(kappa^2 + 3*lambda*mu^2)   [multiplicity 1]
    omega_3' = 36*lambda*u_l^2                        [multiplicity 3]

    Complexity: O(1).
    """
    ring = ring or sage.SR
    mu2 = _coerce_scalar(config.mu2, ring)
    kappa = _coerce_scalar(config.kappa, ring)
    lam = _coerce_scalar(config.lam, ring)
    disc = kappa ** 2 + 3 * lam * mu2
    root = sage.sqrt(disc) if ring is sage.SR else _safe_sqrt(disc, ring)
    omega1p = -8 * u_l * root
    omega3p = 36 * lam * u_l ** 2
    return omega1p, omega3p


# ---------------------------------------------------------------------------
# Section 7: exact energy-crossing condition
# ---------------------------------------------------------------------------

def onshell_energy(kappa, lam, u, epsilon: int):
    """On-shell energy V_red|_crit = -kappa*u^2 - 2*lambda*epsilon*u^3
    (derivation in the proof of Theorem 5).

    epsilon must be +1 (short-root) or -1 (long-root); validated explicitly.

    Complexity: O(1).
    """
    if epsilon not in (1, -1):
        raise ValueError("epsilon must be +1 or -1 (cos 6theta at a critical ray).")
    return -kappa * u ** 2 - 2 * lam * epsilon * u ** 3


def crossing_kappa_star(config: G2PotentialConfig, ring=None) -> object:
    """Return kappa* = -12^(1/4) * sqrt(lambda * mu^2), the exact energy
    crossing condition of Theorem 5 between the larger short-root branch
    and the long-root branch.

    Complexity: O(1).
    """
    ring = ring or sage.SR
    mu2 = _coerce_scalar(config.mu2, ring)
    lam = _coerce_scalar(config.lam, ring)
    product = lam * mu2
    if ring is sage.SR:
        return -(sage.Integer(12) ** (sage.QQ(1) / 4)) * sage.sqrt(product)
    root = _safe_sqrt(product, ring)
    twelve_qtr = ring(12) ** (ring(1) / 4)
    return -twelve_qtr * root


# ---------------------------------------------------------------------------
# Section 8: stabilized potential, exact cubic solve via Cardano / trig form
# ---------------------------------------------------------------------------

def stabilized_critical_radius(
    config: G2PotentialConfig,
    epsilon: int,
    branch_k: int = 0,
    ring=None,
    timeout_seconds: int = 30,
):
    """Solve the exact depressed cubic 4*eta*u^3 + 3*lambda*eps*u^2
    + 2*kappa*u + mu^2 = 0  (Eq. 8.1) via the trigonometric Cardano form
    (Eq. 8.3) when the discriminant is negative (three real roots), or via
    the direct Cardano radical form (Eq. 8.2) otherwise.

    Parents: all scalars in `ring` (default sage.SR); branch_k in {0,1,2}
    selects which of the three real roots (Eq. 8.3) is returned.

    Complexity: O(1) ring operations, wrapped in an alarm-based timeout
    guard since Cardano's trigonometric form invokes transcendental
    functions (arccos, cos) that could stall under pathological symbolic
    input (e.g. unresolved assumptions).
    """
    if config.eta is None:
        raise ValueError("Stabilized model requires config.eta to be set (eta > 0).")
    if epsilon not in (1, -1):
        raise ValueError("epsilon must be +1 or -1.")
    if branch_k not in (0, 1, 2):
        raise ValueError("branch_k must be 0, 1, or 2.")

    ring = ring or sage.SR
    mu2 = _coerce_scalar(config.mu2, ring)
    kappa = _coerce_scalar(config.kappa, ring)
    lam = _coerce_scalar(config.lam, ring)
    eta = _coerce_scalar(config.eta, ring)

    if ring is not sage.SR and eta <= 0:
        raise NonPhysicalBranchError("Stabilizing coupling eta must be positive.")

    a = 4 * eta
    b = 3 * lam * epsilon
    c = 2 * kappa
    d = mu2

    p = (8 * eta * kappa - 3 * lam ** 2) / (16 * eta ** 2)
    q = (lam ** 3 * epsilon - 4 * eta * lam * kappa * epsilon + 8 * eta ** 2 * mu2) / (32 * eta ** 3)

    def _solve():
        Delta = (q / 2) ** 2 + (p / 3) ** 3
        shift = -(lam * epsilon) / (4 * eta)
        if ring is sage.SR:
            # Symbolic branch: return the Cardano radical form (Eq. 8.2)
            # directly; case distinction on Delta's sign is left to the
            # caller when substituting numeric values.
            term1 = (-q / 2 + sage.sqrt(Delta)) ** (sage.QQ(1) / 3)
            term2 = (-q / 2 - sage.sqrt(Delta)) ** (sage.QQ(1) / 3)
            return shift + term1 + term2

        if Delta >= 0:
            root_disc = _safe_sqrt(Delta, ring)
            term1 = _real_cbrt(-q / 2 + root_disc, ring)
            term2 = _real_cbrt(-q / 2 - root_disc, ring)
            return shift + term1 + term2

        # Delta < 0: trigonometric form, Eq. 8.3
        r_val = _safe_sqrt(-((p / 3) ** 3), ring)
        cos_arg = (-q / 2) / r_val
        cos_arg = max(min(cos_arg, ring(1)), ring(-1))  # guard against fp overshoot
        phi = ring(cos_arg).arccos()
        cbrt_r = _real_cbrt(r_val, ring)
        angle = (phi + 2 * sage.pi * branch_k) / 3
        return shift + 2 * cbrt_r * ring(angle).cos()

    if ring is sage.SR:
        return _solve()

    try:
        sage.alarm(timeout_seconds)
        result = _solve()
    except sage.AlarmInterrupt as exc:
        raise AlgebraicComplexityError(
            f"Stabilized cubic solve exceeded {timeout_seconds}s timeout."
        ) from exc
    finally:
        sage.cancel_alarm()
    return result


def _real_cbrt(value, ring):
    """Real cube root (handles negative radicands correctly, unlike a
    naive fractional power). Complexity: O(1)."""
    if value >= 0:
        return ring(value) ** (ring(1) / 3)
    return -((-ring(value)) ** (ring(1) / 3))


def eta_crit(config: G2PotentialConfig, ring=None) -> object:
    """Exact critical stabilizing coupling eta_crit (Eq. 8.8) at which the
    long-root branch flips from saddle to local minimum.

    u* is given by Eq. 8.7: u* = (2*kappa + sqrt(4*kappa^2 + 9*lambda*mu2)) / (3*lambda)
    eta_crit = (3*lambda*u* - kappa) / (6*u*^2)

    Complexity: O(1).
    """
    ring = ring or sage.SR
    mu2 = _coerce_scalar(config.mu2, ring)
    kappa = _coerce_scalar(config.kappa, ring)
    lam = _coerce_scalar(config.lam, ring)

    disc = 4 * kappa ** 2 + 9 * lam * mu2
    root = sage.sqrt(disc) if ring is sage.SR else _safe_sqrt(disc, ring)
    u_star = (2 * kappa + root) / (3 * lam)
    if ring is not sage.SR and u_star <= 0:
        raise NonPhysicalBranchError("u* must be positive for eta_crit to be physical.")
    return (3 * lam * u_star - kappa) / (6 * u_star ** 2)


def global_vacuum_inequality(eta, lam, kappa, u, ring=None) -> bool:
    """Evaluate the exact inequality (Eq. 8.9): 3*eta*u^2 - 2*lambda*u + kappa > 0
    determining whether the long-root branch is the global vacuum.

    Complexity: O(1).
    """
    ring = ring or sage.SR
    lhs = 3 * eta * u ** 2 - 2 * lam * u + kappa
    if ring is sage.SR:
        return bool(sage.SR(lhs) > 0)
    return lhs > 0


# ---------------------------------------------------------------------------
# Verification suite (property-based, bounded random generation)
# ---------------------------------------------------------------------------

class TestAlgebraicInvariants:
    """Structural invariant tests for the truncated / stabilized G2 potential.

    CI/CD-safe: each loop is capped at 50 random iterations, and random
    parameters are drawn with explicit bounds to avoid degree/coefficient
    explosion.
    """

    ITERATIONS = 50
    R200 = sage.RealField(200)

    def _random_positive(self, low=0.05, high=5.0):
        sage.set_random_seed(1234567)
        return self.R200(sage.RR.random_element(low, high).abs() + low)

    def test_short_root_stationarity(self) -> None:
        """Invariant: 3*lambda*u^2 + 2*kappa*u + mu^2 == 0 at u_{s,+/-}
        whenever the branch exists.

        Complexity of check: O(1) per sample.
        """
        sage.set_random_seed(42)
        successes = 0
        for _ in range(self.ITERATIONS):
            mu2 = self.R200(sage.RR.random_element(0.1, 3.0).abs())
            lam = self.R200(sage.RR.random_element(0.1, 3.0).abs())
            kappa_bound = (3 * lam * mu2).sqrt() + self.R200(0.5)
            kappa = -self.R200(sage.RR.random_element(0.0, 1.0)) * kappa_bound - self.R200(0.01)
            cfg = G2PotentialConfig(mu2=mu2, kappa=kappa, lam=lam)
            try:
                u_plus, u_minus = short_root_radii(cfg, ring=self.R200)
            except NonPhysicalBranchError:
                continue
            for u in (u_plus, u_minus):
                residual = 3 * lam * u ** 2 + 2 * kappa * u + mu2
                if abs(residual) > self.R200(1e-30):
                    raise ValueError(f"Short-root stationarity violated: residual={residual}")
            successes += 1
        logger.info("test_short_root_stationarity: %d/%d branches existed and verified",
                     successes, self.ITERATIONS)

    def test_long_root_stationarity(self) -> None:
        """Invariant: 3*lambda*u^2 - 2*kappa*u - mu^2 == 0 at u_l."""
        sage.set_random_seed(43)
        for _ in range(self.ITERATIONS):
            mu2 = self.R200(sage.RR.random_element(0.1, 3.0).abs())
            lam = self.R200(sage.RR.random_element(0.1, 3.0).abs())
            kappa = self.R200(sage.RR.random_element(-3.0, 3.0))
            cfg = G2PotentialConfig(mu2=mu2, kappa=kappa, lam=lam)
            u_l = long_root_radius(cfg, ring=self.R200)
            if u_l <= 0:
                raise ValueError("Long-root radius must be strictly positive.")
            residual = 3 * lam * u_l ** 2 - 2 * kappa * u_l - mu2
            if abs(residual) > self.R200(1e-28):
                raise ValueError(f"Long-root stationarity violated: residual={residual}")
        logger.info("test_long_root_stationarity: %d samples verified", self.ITERATIONS)

    def test_hessian_matches_numeric_second_derivative(self) -> None:
        """Cross-check the closed-form Hessian eigenvalues against a direct
        finite-difference second derivative of V_red along each axis."""
        sage.set_random_seed(44)
        h = self.R200(1e-6)
        for _ in range(self.ITERATIONS):
            mu2 = self.R200(sage.RR.random_element(0.1, 2.0).abs())
            lam = self.R200(sage.RR.random_element(0.1, 2.0).abs())
            kappa_bound = (3 * lam * mu2).sqrt() + self.R200(0.5)
            kappa = -self.R200(sage.RR.random_element(0.0, 1.0)) * kappa_bound - self.R200(0.01)
            cfg = G2PotentialConfig(mu2=mu2, kappa=kappa, lam=lam)
            try:
                u_s, _ = short_root_radii(cfg, ring=self.R200)
            except NonPhysicalBranchError:
                continue
            x0 = u_s.sqrt()

            def V_x(x):
                return mu2 * x ** 2 + kappa * x ** 4 + lam * x ** 6

            def V_y(y):
                # y-direction at fixed x0, epsilon=+1 slice: I6 ~ x^6 - 15x^4y^2 + ...
                return (mu2 * (x0 ** 2 + y ** 2) + kappa * (x0 ** 2 + y ** 2) ** 2
                        + lam * (x0 ** 6 - 15 * x0 ** 4 * y ** 2))

            vxx_numeric = (V_x(x0 + h) - 2 * V_x(x0) + V_x(x0 - h)) / h ** 2
            vyy_numeric = (V_y(h) - 2 * V_y(self.R200(0)) + V_y(-h)) / h ** 2

            omega1, omega3 = short_root_hessian(cfg, u_s, ring=self.R200)
            if abs(vxx_numeric - omega1) > self.R200(1e-3):
                raise ValueError(f"omega1 mismatch: closed-form={omega1}, numeric={vxx_numeric}")
            if abs(vyy_numeric - omega3) > self.R200(1e-3):
                raise ValueError(f"omega3 mismatch: closed-form={omega3}, numeric={vyy_numeric}")
        logger.info("test_hessian_matches_numeric_second_derivative: passed")

    def test_crossing_condition_numerically(self) -> None:
        """At kappa = kappa*, verify V_{s,+} == V_l to high precision, using
        randomly sampled (mu2, lambda)."""
        sage.set_random_seed(45)
        for _ in range(self.ITERATIONS):
            mu2 = self.R200(sage.RR.random_element(0.1, 2.0).abs())
            lam = self.R200(sage.RR.random_element(0.1, 2.0).abs())
            cfg_probe = G2PotentialConfig(mu2=mu2, kappa=self.R200(0), lam=lam)
            kappa_star = crossing_kappa_star(cfg_probe, ring=self.R200)
            cfg = G2PotentialConfig(mu2=mu2, kappa=kappa_star, lam=lam)
            try:
                u_s_plus, _ = short_root_radii(cfg, ring=self.R200)
                u_l = long_root_radius(cfg, ring=self.R200)
            except NonPhysicalBranchError:
                continue
            v_s = onshell_energy(kappa_star, lam, u_s_plus, epsilon=1)
            v_l = onshell_energy(kappa_star, lam, u_l, epsilon=-1)
            if abs(v_s - v_l) > self.R200(1e-20):
                raise ValueError(f"Crossing condition failed: V_s={v_s}, V_l={v_l}")
        logger.info("test_crossing_condition_numerically: passed")

    def test_stabilized_cubic_root_satisfies_eq81(self) -> None:
        """Verify the trigonometric Cardano solution actually satisfies the
        original depressed-cubic stationarity equation (Eq. 8.1)."""
        sage.set_random_seed(46)
        for _ in range(self.ITERATIONS):
            mu2 = self.R200(sage.RR.random_element(0.1, 2.0).abs())
            lam = self.R200(sage.RR.random_element(0.1, 2.0).abs())
            eta = self.R200(sage.RR.random_element(0.1, 2.0).abs()) + lam  # eta > lambda
            kappa = self.R200(sage.RR.random_element(-2.0, 2.0))
            cfg = G2PotentialConfig(mu2=mu2, kappa=kappa, lam=lam, eta=eta)
            epsilon = -1
            u = stabilized_critical_radius(cfg, epsilon=epsilon, branch_k=0, ring=self.R200)
            residual = 4 * eta * u ** 3 + 3 * lam * epsilon * u ** 2 + 2 * kappa * u + mu2
            if abs(residual) > self.R200(1e-15):
                raise ValueError(f"Stabilized cubic residual too large: {residual}")
        logger.info("test_stabilized_cubic_root_satisfies_eq81: passed")


def run_all_tests() -> None:
    """Run the full CI-safe verification suite. Complexity: O(1) modulo the
    fixed 50-iteration cap per test."""
    suite = TestAlgebraicInvariants()
    suite.test_short_root_stationarity()
    suite.test_long_root_stationarity()
    suite.test_hessian_matches_numeric_second_derivative()
    suite.test_crossing_condition_numerically()
    suite.test_stabilized_cubic_root_satisfies_eq81()
    logger.info("All G2 truncated-potential invariant tests passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Numerical example from Section 9: (mu2, kappa, lambda) = (1, -2, 1)
    R = sage.RealField(200)
    cfg_example = G2PotentialConfig(mu2=R(1), kappa=R(-2), lam=R(1))
    u_s_plus, u_s_minus = short_root_radii(cfg_example, ring=R)
    u_l = long_root_radius(cfg_example, ring=R)
    logger.info("u_s,+ = %s (expected 1)", u_s_plus)
    logger.info("u_s,- = %s (expected 1/3)", u_s_minus)
    logger.info("u_l   = %s (expected (-2+sqrt(7))/3 ~ 0.215)", u_l)

    omega1, omega3 = short_root_hessian(cfg_example, u_s_plus, ring=R)
    omega1p, omega3p = long_root_hessian(cfg_example, u_l, ring=R)
    logger.info("omega1=%s (expected 8), omega3=%s (expected -36)", omega1, omega3)
    logger.info("omega1'=%s (expected ~-4.56), omega3'=%s (expected ~1.67)", omega1p, omega3p)

    run_all_tests()

# ---------------------------------------------------------------------------
# Dependency / environment manifest
# ---------------------------------------------------------------------------
# SageMath version required: >= 10.3
# Optional packages: none required (pure sage.all core: SR, RR, RealField,
#   Integer, QQ, alarm/AlarmInterrupt, set_random_seed).
# External library bindings: none.
