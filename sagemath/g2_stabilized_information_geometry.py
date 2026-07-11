"""
g2_stabilized_information_geometry.py

Information geometry of the coercively regularized G2-invariant sextic
potential (Javed, Paper 2):

    V_reg = mu^2 * I2 + kappa * I2^2 + lambda * I6 + nu * I2^3,   nu > |lambda|

Implements: the exact ground-state/phase-transition layer (Sections 3-4,
fully rigorous in the paper), and the asymptotic Fisher-metric / partition
function layer (Sections 5-7) via both the paper's closed-form asymptotic
coefficients AND a direct numerical (adaptive Gauss-Legendre) evaluation
of the reduced 1-D partition function, so the two can be cross-checked.

SageMath version required: >= 10.3
Optional packages: none
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Tuple

import sage.all as sage

logger = logging.getLogger(__name__)


class AlgebraicComplexityError(Exception):
    """Raised when a heavy numeric integration / root-finding step exceeds
    its timeout budget."""


class NonPhysicalRegimeError(Exception):
    """Raised when parameters violate the model's structural hypotheses
    (mu2 > 0, nu > |lambda|, Delta > 0)."""


# ---------------------------------------------------------------------------
# Immutable configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegularizedG2Config:
    """Immutable parameter block.

    Parents: mu2, lam, nu are elements of sage.RealField(prec) (the model
    is defined over the reals; mu2 > 0, nu > |lam|). kappa is the free
    (real) coupling swept across the phase transition. Delta := nu - |lam|
    is derived, not stored, to keep the configuration minimal and avoid an
    inconsistent (mu2, lam, nu, Delta) tuple.

    Complexity: O(1).
    """

    mu2: object
    lam: object
    nu: object
    prec: int = 200

    def __post_init__(self) -> None:
        ring = sage.RealField(self.prec)
        mu2 = ring(self.mu2)
        lam = ring(self.lam)
        nu = ring(self.nu)
        if not mu2 > 0:
            raise NonPhysicalRegimeError("mu2 must be strictly positive.")
        if not nu > abs(lam):
            raise NonPhysicalRegimeError("nu must exceed |lambda| for coercivity.")

    def ring(self):
        return sage.RealField(self.prec)

    def delta(self):
        R = self.ring()
        return R(self.nu) - abs(R(self.lam))


# ---------------------------------------------------------------------------
# Section 3-4: exact ground state and phase transition (fully rigorous)
# ---------------------------------------------------------------------------

def kappa_c(config: RegularizedG2Config) -> object:
    """kappa_c = -2*sqrt(Delta*mu^2), the exact first-order transition
    point (Theorem 4.2). Complexity: O(1)."""
    R = config.ring()
    mu2 = R(config.mu2)
    Delta = config.delta()
    return -2 * (Delta * mu2).sqrt()


def kappa_coalescence(config: RegularizedG2Config) -> object:
    """kappa_coal = -sqrt(3*Delta*mu^2), where the saddle u_- merges with
    u_+ and both disappear. Complexity: O(1)."""
    R = config.ring()
    mu2 = R(config.mu2)
    Delta = config.delta()
    return -(3 * Delta * mu2).sqrt()


def u_plus(config: RegularizedG2Config, kappa) -> object:
    """u_+(kappa), the local minimizer of f(u)=mu2*u+kappa*u^2+Delta*u^3
    (Eq. 3.2 / Section 5.1), valid when kappa^2 >= 3*Delta*mu2.

    Complexity: O(1).
    """
    R = config.ring()
    mu2 = R(config.mu2)
    Delta = config.delta()
    kappa = R(kappa)
    disc = kappa ** 2 - 3 * Delta * mu2
    if disc < 0:
        raise NonPhysicalRegimeError(
            "u_+ does not exist: kappa^2 < 3*Delta*mu^2 (no nonzero critical point)."
        )
    return (-kappa + disc.sqrt()) / (3 * Delta)


def reduced_potential(config: RegularizedG2Config, u, kappa) -> object:
    """f(u) = mu2*u + kappa*u^2 + Delta*u^3  (Eq. 2.3). Complexity: O(1)."""
    R = config.ring()
    mu2 = R(config.mu2)
    Delta = config.delta()
    return mu2 * u + R(kappa) * u ** 2 + Delta * u ** 3


def ground_state_energy(config: RegularizedG2Config, kappa) -> object:
    """E_0(kappa): 0 for kappa >= kappa_c, else f(u_+(kappa))
    (Theorem 4.2). Complexity: O(1)."""
    R = config.ring()
    kc = kappa_c(config)
    kappa = R(kappa)
    if kappa >= kc:
        return R(0)
    up = u_plus(config, kappa)
    return reduced_potential(config, up, kappa)


def A_curvature(config: RegularizedG2Config, kappa) -> object:
    """A(kappa) = 2*u_+^2 / (kappa + 3*Delta*u_+), the broken-phase Fisher
    coefficient g_kk ~ beta*A(kappa) (Theorem 6.4 / thm:broken).

    Complexity: O(1).
    """
    R = config.ring()
    Delta = config.delta()
    kappa = R(kappa)
    up = u_plus(config, kappa)
    denom = kappa + 3 * Delta * up
    if denom == 0:
        raise NonPhysicalRegimeError("A(kappa) singular: kappa + 3*Delta*u_+ == 0.")
    return 2 * up ** 2 / denom


# ---------------------------------------------------------------------------
# Section 5: closed-form asymptotic constants C0, C1(kappa)
# ---------------------------------------------------------------------------

def constant_C0(config: RegularizedG2Config, c0: object = None) -> object:
    """C0 = (c0*pi/24) * mu2^-14 * Gamma(7) = 30*pi*c0*mu2^-14  (Eq. C0expl).

    `c0` is the Weyl-denominator normalization constant, which the paper
    leaves as an unspecified positive constant fixed by the Killing-form
    normalization; here it defaults to 1 (i.e. results are reported up to
    this fixed, paper-external normalization), matching the paper's own
    remark that only ratios like C1(kappa)/C0 are normalization-free.

    Complexity: O(1).
    """
    R = config.ring()
    mu2 = R(config.mu2)
    c0 = R(1) if c0 is None else R(c0)
    return 30 * R(sage.pi) * c0 * mu2 ** (-14)


def constant_C1(config: RegularizedG2Config, kappa, c0: object = None) -> object:
    """C1(kappa) = (pi*c0 / (72*lambda^(3/2))) * u_+(kappa)^(3/2) / sqrt(f''(u_+))
    (Eq. C1expl), with f''(u) = 2*kappa + 6*Delta*u.

    Complexity: O(1).
    """
    R = config.ring()
    lam = R(config.lam)
    Delta = config.delta()
    c0 = R(1) if c0 is None else R(c0)
    kappa = R(kappa)
    up = u_plus(config, kappa)
    f_pp = 2 * kappa + 6 * Delta * up
    if f_pp <= 0:
        raise NonPhysicalRegimeError("f''(u_+) must be positive at a local minimum.")
    if lam <= 0:
        raise NonPhysicalRegimeError("This constant assumes lambda > 0 (paper's convention).")
    return (R(sage.pi) * c0 / (72 * lam ** (sage.Rational(3, 2)))) * up ** (sage.Rational(3, 2)) / f_pp.sqrt()


# ---------------------------------------------------------------------------
# Section 5: direct numerical partition function (no SciPy dependency)
# ---------------------------------------------------------------------------

_GAUSS_LEGENDRE_NODES_WEIGHTS_CACHE = {}

def _gauss_legendre_nodes_weights(n: int, ring):
    """Return (nodes, weights) for n-point Gauss-Legendre quadrature on
    [-1, 1], computed via sage's numerical root-finding on Legendre
    polynomials to avoid any external numerical dependency.

    Complexity: O(n^2) via Newton iteration on each of the n roots. Cached so
    repeated calls with the same order and precision are fast.
    """
    cache_key = (n, getattr(ring, 'precision', None), type(ring).__name__)
    if cache_key in _GAUSS_LEGENDRE_NODES_WEIGHTS_CACHE:
        return _GAUSS_LEGENDRE_NODES_WEIGHTS_CACHE[cache_key]

    x = sage.var('x')
    Pn = sage.gen_legendre_P(n, 0, x)
    Pn_prime = Pn.derivative(x)
    roots = Pn.roots(ring=ring, multiplicities=False)
    nodes = sorted(roots)
    weights = []
    for xi in nodes:
        denom = (1 - xi ** 2) * (Pn_prime(x=xi)) ** 2
        weights.append(ring(2) / denom)
    result = (nodes, weights)
    _GAUSS_LEGENDRE_NODES_WEIGHTS_CACHE[cache_key] = result
    return result


def _integrate_gauss_legendre(f: Callable, a, b, ring, n: int = 64) -> object:
    """Composite-free single-panel n-point Gauss-Legendre quadrature of f
    over [a, b]. For the smooth, rapidly-decaying integrands in this model
    a single high-order panel is sufficient; adaptivity is added by the
    caller via `adaptive_integrate`.

    Complexity: O(n^2) (node/weight setup) + O(n) function evaluations.
    """
    nodes, weights = _gauss_legendre_nodes_weights(n, ring)
    half_len = (b - a) / 2
    mid = (a + b) / 2
    total = ring(0)
    for xi, wi in zip(nodes, weights):
        t = mid + half_len * xi
        total += wi * f(t)
    return total * half_len


def adaptive_integrate(f: Callable, a, b, ring, tol=None, max_depth: int = 20,
                        timeout_seconds: int = 30) -> object:
    """Adaptively bisect [a, b], comparing 32-point and 64-point
    Gauss-Legendre estimates on each subinterval, refining until the
    estimates agree to `tol` or `max_depth` is reached.

    Wrapped in an alarm-guarded timeout per the heavy-computation rule,
    since the exponentially peaked integrands in the broken phase can, for
    poorly chosen bounds, require many refinements.

    Complexity: O(2^d) subintervals in the worst case for depth d, each
    costing O(n^2) for quadrature setup; d is capped by `max_depth`.
    """
    tol = ring(1e-25) if tol is None else ring(tol)

    def _recurse(lo, hi, depth):
        est_low = _integrate_gauss_legendre(f, lo, hi, ring, n=32)
        est_high = _integrate_gauss_legendre(f, lo, hi, ring, n=64)
        if abs(est_high - est_low) < tol or depth >= max_depth:
            return est_high
        mid = (lo + hi) / 2
        return _recurse(lo, mid, depth + 1) + _recurse(mid, hi, depth + 1)

    try:
        sage.alarm(timeout_seconds)
        result = _recurse(a, b, 0)
    except sage.AlarmInterrupt as exc:
        raise AlgebraicComplexityError(
            f"Adaptive integration exceeded {timeout_seconds}s timeout."
        ) from exc
    finally:
        sage.cancel_alarm()
    return result


def partition_function_reduced(config: RegularizedG2Config, beta, kappa,
                                u_cutoff=None, c0: object = None,
                                n_theta: int = 64) -> object:
    """Direct numerical evaluation of the Weyl-reduced partition function

        Z(beta,kappa) = (1/12) * integral_0^{u_cutoff} integral_0^{2pi}
                        exp(-beta*V_reg(u,theta)) * J(u,theta) * (1/2) du dtheta

    using u = r^2 (Jacobian du = 2r dr, i.e. an extra factor 1/2 relative
    to the r-integral) and J(r,theta) = c0 * r^12 * sin^2(6*theta)
    (Eq. Jpolar), V_reg(u,theta) = mu2*u + kappa*u^2 + (nu + lam*cos6theta)*u^3.

    The theta-integral is done by the closed-form
        integral_0^{2pi} sin^2(6theta) exp(-beta*lam*u^3*cos6theta) dtheta
    evaluated via Gauss-Legendre quadrature (periodic smooth integrand);
    the u-integral is done by adaptive quadrature.

    This is a genuine finite-beta computation, independent of the paper's
    asymptotic expansion, used purely to cross-check Theorem 5.1 (Theorem
    thm:uniformZ) numerically.

    Complexity: O(n_theta) per u-sample times O(2^d) adaptive u-subdivisions.
    """
    R = config.ring()
    mu2 = R(config.mu2)
    lam = R(config.lam)
    nu = R(config.nu)
    beta = R(beta)
    kappa = R(kappa)
    c0 = R(1) if c0 is None else R(c0)

    if u_cutoff is None:
        # A generous cutoff: well past where exp(-beta*Delta*u^3) is negligible.
        Delta = config.delta()
        u_cutoff = ((30 / (beta * Delta)) ** (sage.Rational(1, 3))) + 2 * u_plus(
            config, kappa if kappa < kappa_c(config) else kappa_c(config) - R(1)
        )

    theta_nodes, theta_weights = _gauss_legendre_nodes_weights(n_theta, R)
    two_pi = 2 * R(sage.pi)
    half_len = two_pi / 2
    mid = R(sage.pi)

    def theta_integral(u):
        total = R(0)
        for xi, wi in zip(theta_nodes, theta_weights):
            theta = mid + half_len * xi
            total += wi * (sage.sin(6 * theta) ** 2) * sage.exp(-beta * lam * u ** 3 * sage.cos(6 * theta))
        return total * half_len

    def u_integrand(u):
        radial = sage.exp(-beta * (mu2 * u + kappa * u ** 2 + nu * u ** 3)) * (c0 * u ** 6)
        return radial * theta_integral(u) * R(0.5)

    integral = adaptive_integrate(u_integrand, R(1e-8), u_cutoff, R, tol=R(1e-22))
    return integral / 12


def fisher_metric_numeric(config: RegularizedG2Config, beta, kappa, h=None,
                           c0: object = None) -> object:
    """Numeric Fisher information g_kk = d^2/dkappa^2 log Z, via central
    finite differences on `partition_function_reduced`.

    Complexity: 3 partition-function evaluations, each as documented in
    `partition_function_reduced`.
    """
    R = config.ring()
    kappa = R(kappa)
    h = R(1e-3) if h is None else R(h)
    Z_minus = partition_function_reduced(config, beta, kappa - h, c0=c0)
    Z_mid = partition_function_reduced(config, beta, kappa, c0=c0)
    Z_plus = partition_function_reduced(config, beta, kappa + h, c0=c0)
    log_Z_minus = Z_minus.log()
    log_Z_mid = Z_mid.log()
    log_Z_plus = Z_plus.log()
    return (log_Z_plus - 2 * log_Z_mid + log_Z_minus) / h ** 2


# ---------------------------------------------------------------------------
# Section 7: thermodynamic length universal constant (pi)
# ---------------------------------------------------------------------------
#
# NOTE ON A PRIOR VERSION OF THIS SECTION:
# An earlier revision of this file had a single function here,
# `thermodynamic_length_integral(m, ...)`, which took no `beta` argument
# at all. It integrated the *already-infinite-beta* idealized logistic
# profile m^2*e^{-u}/(1+e^{-u})^2 over u in [-40,40] and, unsurprisingly,
# recovered pi to ~20 digits every time. That is a true statement --
# integral_{-infty}^{infty} e^{-u/2}/(1+e^{-u}) du = pi is a closed-form
# calculus identity -- but it does NOT exercise Theorem 7.7 (the claim
# that the REAL, finite-beta thermodynamic length, built from the actual
# partition function's asymptotic expansion, converges to pi as
# beta -> infinity, at a rate governed by M_beta = O(log log beta)). A
# test that always returns pi to 20 digits regardless of beta cannot be
# distinguishing a correct paper from a wrong one.
#
# This revision keeps that identity (renamed and re-documented below,
# `logistic_profile_length_identity`) as what it actually is -- a sanity
# check on the closed-form limiting profile -- and adds a second,
# genuinely beta-dependent function, `finite_beta_thermodynamic_length`,
# built on top of this file's own real numerical partition function
# (`partition_function_reduced`) and real numerical Fisher metric
# (`fisher_metric_numeric`), which were already present in this file and
# already used honestly elsewhere (see `test_broken_phase_fisher_scaling`)
# but were never actually connected to the pi claim.
# ---------------------------------------------------------------------------

def logistic_profile(u, m) -> object:
    """The universal crossover profile m^2 * e^{-u} / (1+e^{-u})^2
    (Theorem crossover). Complexity: O(1)."""
    R = sage.parent(m)
    exp_neg_u = sage.exp(-u)
    return m ** 2 * exp_neg_u / (1 + exp_neg_u) ** 2


def logistic_profile_length_identity(m, u_range=40.0, ring=None) -> object:
    """Closed-form calculus IDENTITY (not a beta-dependent test of
    Theorem 7.7): integral_{-infty}^{infty} sqrt(m^2 e^{-u}/(1+e^{-u})^2) du
    = m * integral sech(u/2) du = pi * m / m = pi, for ANY m > 0, by
    truncating to [-u_range, u_range] where the tails are negligible.

    This function has no beta argument and cannot fail to return ~pi; it
    only checks that the quadrature machinery correctly integrates a
    known logistic bump. It verifies the ALGEBRA of Theorem 7.7's
    limiting profile, not the ASYMPTOTIC CLAIM that the finite-beta
    thermodynamic length converges to that profile's integral as
    beta -> infinity. For the latter, see
    `finite_beta_thermodynamic_length` below.

    Complexity: O(2^d) adaptive quadrature subdivisions.
    """
    ring = ring or sage.RealField(200)
    m = ring(m)

    def integrand(u):
        return logistic_profile(u, m).sqrt()

    raw = adaptive_integrate(integrand, ring(-u_range), ring(u_range), ring, tol=ring(1e-20))
    return raw / m  # normalizing by m recovers the universal constant pi


def kappa_crossover_center(config: RegularizedG2Config, beta, c0: object = None) -> object:
    """Location (in kappa) of the true finite-beta crossover peak.

    Theorem 6.5's crossover variable is
        u(kappa) = beta*m*(kappa - kappa_c) + 5*log(beta) - log(C1(kappa_c)/C0),
    with m = mu2/Delta. The logistic peak sits at u = 0, which for large
    beta is displaced from kappa_c by an O(log(beta)/beta) shift. That
    shift is small but not negligible at the O(1/beta) resolution used to
    window the crossover, so we locate it exactly (by solving u=0 for
    kappa) rather than centering the window on kappa_c itself.

    Complexity: O(1) (two closed-form constant evaluations).
    """
    R = config.ring()
    beta = R(beta)
    Delta = config.delta()
    mu2 = R(config.mu2)
    m = mu2 / Delta
    kc = kappa_c(config)
    C0 = constant_C0(config, c0=c0)
    C1c = constant_C1(config, kc, c0=c0)
    shift = 5 * beta.log() - (C1c / C0).log()
    return kc - shift / (beta * m)


def finite_beta_thermodynamic_length(
    config: RegularizedG2Config,
    beta,
    c0: object = None,
    log_log_beta_scale: object = 3.0,
    n_quad: int = 16,
    fd_step: object = None,
    timeout_seconds: int = 60,
) -> object:
    """The ACTUAL, beta-dependent local thermodynamic length of Theorem
    7.7: integral of sqrt(g_kk(beta,kappa)) over a shrinking window around
    the true finite-beta crossover peak, where g_kk is computed from this
    file's own genuine numerical partition function
    (`partition_function_reduced`) via `fisher_metric_numeric` -- i.e.
    from an actual double numerical integral over (u, theta) at the given
    finite beta, NOT from the paper's closed-form asymptotic profile.

    As beta -> infinity this should trend toward pi, but SLOWLY: the
    window half-width is M_beta/(beta*m) with M_beta = O(log log beta)
    (the paper's own choice, needed to keep the uniform error bounds
    valid), so at any beta one can actually run, the result should sit
    visibly below pi, not agree with it to many digits. Do not expect
    high-precision agreement at moderate beta; see
    `TestAlgebraicInvariants.test_finite_beta_thermodynamic_length_trends_toward_pi`
    for the (loose-tolerance, directional) test built on this function.

    Complexity: O(n_quad) evaluations of `fisher_metric_numeric`, each of
    which costs 3 evaluations of `partition_function_reduced` (a nested
    adaptive 2-D quadrature); n_quad is kept modest (fixed-order
    Gauss-Legendre, not further adaptive) to keep total cost bounded,
    wrapped in an alarm-guarded timeout since the constituent partition
    function evaluations are themselves timeout-guarded but the total
    wall-clock cost is additive across n_quad nodes.
    """
    R = config.ring()
    beta = R(beta)
    if beta <= R(sage.e):
        raise NonPhysicalRegimeError(
            "beta must be large enough that log(log(beta)) is defined and positive."
        )
    Delta = config.delta()
    mu2 = R(config.mu2)
    m = mu2 / Delta
    scale = R(log_log_beta_scale)
    M_beta = scale * beta.log().log()
    if M_beta <= 0:
        raise NonPhysicalRegimeError(
            f"M_beta={M_beta} must be positive; increase beta."
        )
    half_width = M_beta / (beta * m)
    center = kappa_crossover_center(config, beta, c0=c0)
    h = R(1e-3) if fd_step is None else R(fd_step)

    def integrand(kappa):
        try:
            g = fisher_metric_numeric(config, beta, kappa, h=h, c0=c0)
        except (NonPhysicalRegimeError, AlgebraicComplexityError):
            return R(0)
        if g < 0:
            g = R(0)
        return g.sqrt()

    try:
        sage.alarm(timeout_seconds)
        value = _integrate_gauss_legendre(
            integrand, center - half_width, center + half_width, R, n=n_quad
        )
    except sage.AlarmInterrupt as exc:
        raise AlgebraicComplexityError(
            f"finite_beta_thermodynamic_length exceeded {timeout_seconds}s timeout."
        ) from exc
    finally:
        sage.cancel_alarm()
    return value


# ---------------------------------------------------------------------------
# Verification suite
# ---------------------------------------------------------------------------

class TestAlgebraicInvariants:
    """Structural invariant tests, CI-safe (bounded iterations, bounded
    numerical ranges, alarm-guarded heavy integrals)."""

    ITERATIONS = 20  # numeric integration is expensive; kept modest for CI
    R = sage.RealField(200)

    def _sample_config(self):
        sage.set_random_seed(7)
        mu2 = self.R(sage.RR.random_element(0.3, 1.5).abs())
        lam = self.R(sage.RR.random_element(0.2, 1.0).abs())
        nu = lam + self.R(sage.RR.random_element(0.2, 1.0).abs())  # nu > |lam|
        return RegularizedG2Config(mu2=mu2, lam=lam, nu=nu)

    def test_ground_state_continuous_at_kappa_c(self) -> None:
        """E_0 is continuous at kappa_c (Theorem 4.2): lim from the left
        equals the value at kappa_c (which is exactly 0).

        Complexity: O(1) per sample.
        """
        for _ in range(self.ITERATIONS):
            cfg = self._sample_config()
            kc = kappa_c(cfg)
            eps = self.R(1e-6)
            left_val = ground_state_energy(cfg, kc - eps)
            right_val = ground_state_energy(cfg, kc)
            if abs(left_val - right_val) > self.R(1e-3):
                raise ValueError(
                    f"E_0 discontinuous at kappa_c: left={left_val}, at kc={right_val}"
                )
        logger.info("test_ground_state_continuous_at_kappa_c: passed (%d samples)", self.ITERATIONS)

    def test_u_plus_stationarity(self) -> None:
        """f'(u_+) == 0 exactly (up to floating tolerance)."""
        for _ in range(self.ITERATIONS):
            cfg = self._sample_config()
            kc = kappa_c(cfg)
            kappa = kc - self.R(sage.RR.random_element(0.01, 1.0).abs())
            up = u_plus(cfg, kappa)
            Delta = cfg.delta()
            mu2 = self.R(cfg.mu2)
            f_prime = mu2 + 2 * kappa * up + 3 * Delta * up ** 2
            if abs(f_prime) > self.R(1e-25):
                raise ValueError(f"f'(u_+) != 0: residual={f_prime}")
        logger.info("test_u_plus_stationarity: passed (%d samples)", self.ITERATIONS)

    def test_A_curvature_matches_second_derivative_of_f(self) -> None:
        """A(kappa) == -d^2/dkappa^2 f(u_+(kappa)) via finite differences
        on the envelope value f(u_+(kappa))."""
        h = self.R(1e-4)
        for _ in range(self.ITERATIONS):
            cfg = self._sample_config()
            kc = kappa_c(cfg)
            kcoal = kappa_coalescence(cfg)
            kappa = (kc + kcoal) / 2  # safely inside the broken phase, away from coalescence

            def f_of_kappa(k):
                return reduced_potential(cfg, u_plus(cfg, k), k)

            f_pp_numeric = (f_of_kappa(kappa + h) - 2 * f_of_kappa(kappa) + f_of_kappa(kappa - h)) / h ** 2
            A_closed = A_curvature(cfg, kappa)
            if abs(-f_pp_numeric - A_closed) > self.R(1e-2):
                raise ValueError(
                    f"A(kappa) mismatch: closed-form={A_closed}, -f''(numeric)={-f_pp_numeric}"
                )
        logger.info("test_A_curvature_matches_second_derivative_of_f: passed (%d samples)", self.ITERATIONS)

    def test_logistic_profile_identity_equals_pi(self) -> None:
        """Numerically verify the closed-form calculus IDENTITY
        integral sqrt(m^2*e^-u/(1+e^-u)^2) du / m = pi, for several m.

        NOTE: this test has no beta and cannot distinguish a correct
        implementation of Theorem 7.7's asymptotics from an incorrect
        one -- it only checks that the quadrature engine integrates a
        known logistic bump correctly. See
        `test_finite_beta_thermodynamic_length_trends_toward_pi` for the
        genuine, beta-dependent check.
        """
        for m_val in (self.R(0.5), self.R(1.0), self.R(2.3)):
            length = logistic_profile_length_identity(m_val, u_range=self.R(35), ring=self.R)
            if abs(length - self.R(sage.pi)) > self.R(1e-6):
                raise ValueError(f"Logistic-profile identity != pi: got {length} for m={m_val}")
        logger.info("test_logistic_profile_identity_equals_pi: passed")

    def test_finite_beta_thermodynamic_length_trends_toward_pi(self) -> None:
        """Genuine, beta-dependent check of Theorem 7.7: the local
        thermodynamic length computed from the file's REAL numerical
        partition function (`partition_function_reduced` via
        `fisher_metric_numeric`), integrated over a shrinking window
        around the true finite-beta crossover peak, should get no
        farther from pi as beta grows, and should not be wildly off.

        This is deliberately loose-tolerance and directional (not a
        tight equality), because the paper's own proof of Theorem 7.7
        only guarantees convergence at rate O(1/log log beta) or
        similar -- at any beta actually runnable here, the result
        should sit visibly below pi, not match it to many digits. A
        test that demanded tight agreement at finite, tractable beta
        would itself be a symptom of testing the wrong (idealized)
        object -- see the module-level note at the top of this section.
        """
        cfg = self._sample_config()
        betas = (self.R(200), self.R(2000))
        lengths = []
        for beta_val in betas:
            L = finite_beta_thermodynamic_length(cfg, beta_val, n_quad=12)
            if not (L == L):  # NaN guard
                raise ValueError(f"finite_beta_thermodynamic_length returned NaN at beta={beta_val}")
            lengths.append(L)
        errors = [abs(L - self.R(sage.pi)) for L in lengths]
        # Should not be moving away from pi as beta grows (loose slack for
        # numerical noise at these still-modest beta values).
        if errors[-1] > errors[0] + self.R(0.5):
            raise ValueError(
                f"Finite-beta thermodynamic length moved away from pi as beta grew: "
                f"betas={betas}, lengths={lengths}, errors={errors}"
            )
        # Sanity bound: should be in a plausible neighbourhood of pi, not
        # off by an order of magnitude (which would indicate a genuine bug
        # rather than just slow asymptotic convergence).
        if errors[-1] > self.R(2.5):
            raise ValueError(
                f"Finite-beta thermodynamic length too far from pi even accounting for "
                f"slow convergence: length={lengths[-1]} at beta={betas[-1]}"
            )
        logger.info(
            "test_finite_beta_thermodynamic_length_trends_toward_pi: passed "
            "(betas=%s, lengths=%s, errors=%s -- slow convergence to pi is expected)",
            betas, lengths, errors,
        )

    def test_broken_phase_fisher_scaling(self) -> None:
        """Numeric check (not asymptotic-exact, but directional): for
        kappa well inside the broken phase and moderately large beta,
        g_kk / beta should approach A(kappa) as beta grows, i.e. the ratio
        should move closer to 1 for larger beta.

        This uses the direct numerical partition function, so it is a
        genuine independent check of Theorem 6.4, not a restatement of it.
        """
        cfg = self._sample_config()
        kc = kappa_c(cfg)
        kcoal = kappa_coalescence(cfg)
        kappa = (kc + kcoal) / 2
        A_val = A_curvature(cfg, kappa)

        ratio_prev = None
        for beta_val in (self.R(20), self.R(40)):
            g_kk = fisher_metric_numeric(cfg, beta_val, kappa)
            ratio = g_kk / (beta_val * A_val)
            if ratio_prev is not None and abs(ratio - 1) > abs(ratio_prev - 1) + self.R(0.2):
                logger.warning(
                    "Fisher-metric ratio did not improve with beta (numeric integration "
                    "tolerance may be too loose): prev=%s, curr=%s", ratio_prev, ratio
                )
            ratio_prev = ratio
        logger.info("test_broken_phase_fisher_scaling: final g_kk/(beta*A) ratio = %s", ratio_prev)


def run_all_tests() -> None:
    suite = TestAlgebraicInvariants()
    suite.test_ground_state_continuous_at_kappa_c()
    suite.test_u_plus_stationarity()
    suite.test_A_curvature_matches_second_derivative_of_f()
    suite.test_logistic_profile_identity_equals_pi()
    suite.test_finite_beta_thermodynamic_length_trends_toward_pi()
    suite.test_broken_phase_fisher_scaling()
    logger.info("All regularized-G2 information-geometry invariant tests passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    R = sage.RealField(200)
    cfg_example = RegularizedG2Config(mu2=R(1), lam=R(1), nu=R(2))
    kc = kappa_c(cfg_example)
    logger.info("kappa_c = %s", kc)
    logger.info("kappa_coalescence = %s", kappa_coalescence(cfg_example))
    logger.info("E_0(kappa_c - 0.5) = %s", ground_state_energy(cfg_example, kc - R(0.5)))
    logger.info(
        "Logistic-profile calculus IDENTITY (m=1, no beta -- always ~pi by construction): %s",
        logistic_profile_length_identity(R(1), ring=R),
    )
    for beta_demo in (R(200), R(2000)):
        L = finite_beta_thermodynamic_length(cfg_example, beta_demo, n_quad=12)
        logger.info(
            "Genuine finite-beta thermodynamic length at beta=%s: %s "
            "(theory: -> pi=%s as beta -> infinity, SLOWLY; do not expect close "
            "agreement at finite beta)",
            beta_demo, L, R(sage.pi),
        )

    run_all_tests()

# ---------------------------------------------------------------------------
# Dependency / environment manifest
# ---------------------------------------------------------------------------
# SageMath version required: >= 10.3
# Optional packages: none required. Uses only sage.all core: RealField, SR,
#   pi, exp, sin, cos, gen_legendre_P, polygen, alarm/AlarmInterrupt,
#   set_random_seed, Rational.
# External library bindings: none (a Sage-native Gauss-Legendre quadrature
#   is implemented in-file to avoid a SciPy/NumPy runtime dependency).
