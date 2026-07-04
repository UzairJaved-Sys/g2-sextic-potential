"""
paper1_g2_information_geometry.py

Production implementation of:
"Information Geometry of a Coercively Regularized G2-Invariant Sextic Potential"
(M. U. Javed, 2026).

Implements, with full type hints and structured logging:
  * The reduced cubic potential f(u) = mu2*u + kappa*u**2 + Delta*u**3
    and its critical points u_pm(kappa).
  * The exact ground-state energy E0(kappa) and the first-order
    transition point kappa_c = -2*sqrt(Delta*mu2)  (Theorem 3.2 / Sec. 4).
  * The rigorous uniform low-temperature asymptotic expansion of the
    partition function Z(beta, kappa) (Theorem 5.1 / Corollary 5.1).
  * The Fisher information g_{kappa,kappa}(beta, kappa) in its three
    regimes: broken phase, unbroken phase, and the logistic crossover
    window (Theorems 6.1-6.4).
  * The Kullback-Leibler divergence between two Gibbs states in the
    broken, unbroken, and cross-phase regimes (Theorems 7.1-7.3).
  * The universal local thermodynamic length of the crossover, proven
    in the paper to equal pi exactly (Theorem 7.4).

Backend selection
-----------------
All quantities here are closed-form scalar/vector algebraic expressions
evaluated on 1-D real grids; no automatic differentiation or GPU
acceleration is required by the theory (there are no gradients of a
loss function to back-propagate, only explicit analytic formulas and
one 1-D numerical integral for the thermodynamic length). We therefore
use `numpy` at float64 precision throughout, per the architectural
pillars. `scipy.integrate.quad` is used for the single non-elementary
integral (the thermodynamic length), which is a plain deterministic
1-D quadrature, not a differentiable computational graph.

Normalization note (Stage 0):
    The Weyl-integration Jacobian constant c0 in J(r,theta) =
    c0 * r**12 * sin(6 theta)**2 is, per the paper itself, immaterial
    to every reported observable (Fisher metric, KL divergence,
    thermodynamic length) because it cancels in all ratios C1/C0 that
    appear. We fix c0 = 1.0 and expose it as a configurable constant
    for transparency.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Tuple

import numpy as np
from scipy import integrate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Immutable physical configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class G2PotentialConfig:
    """Immutable physical parameters of the regularized G2 sextic model.

    Attributes:
        mu2: The mass-squared parameter mu^2 (paper requires mu2 > 0).
        lambda_: The sextic anisotropy coupling lambda (paper uses lambda;
            escaped with trailing underscore to avoid the Python keyword).
        nu: The stabilizing coupling nu (paper requires nu > |lambda_|).
        c0: Jacobian normalization constant (see module docstring). Cancels
            in every reported observable; default 1.0.

    Complexity:
        Time: O(1) to construct and validate.
        Space: O(1).
    """

    mu2: float
    lambda_: float
    nu: float
    c0: float = 1.0

    def __post_init__(self) -> None:
        if self.mu2 <= 0.0:
            raise ValueError(f"mu2 must be > 0 (coercivity requires mu2>0); got {self.mu2}")
        if self.nu <= abs(self.lambda_):
            raise ValueError(
                "nu must exceed |lambda_| for coercivity (Delta = nu - |lambda_| > 0); "
                f"got nu={self.nu}, lambda_={self.lambda_}"
            )
        if self.c0 <= 0.0:
            raise ValueError(f"c0 (Jacobian normalization) must be > 0; got {self.c0}")

    @property
    def Delta(self) -> float:  # noqa: N802 - paper symbol Delta
        """Delta := nu - |lambda_| > 0, the effective sextic stiffness."""
        return self.nu - abs(self.lambda_)

    @property
    def kappa_c(self) -> float:
        """First-order transition point kappa_c = -2*sqrt(Delta*mu2)."""
        return -2.0 * math.sqrt(self.Delta * self.mu2)

    @property
    def kappa_coal(self) -> float:
        """Saddle-coalescence point kappa_coal = -sqrt(3*Delta*mu2)."""
        return -math.sqrt(3.0 * self.Delta * self.mu2)


class Phase(str, Enum):
    """Which side of the first-order transition kappa lies on."""

    BROKEN = "broken"          # kappa < kappa_c : nonzero orbit is global min
    UNBROKEN = "unbroken"      # kappa > kappa_c : origin is global min
    CROSSOVER = "crossover"    # kappa within O(1/beta) of kappa_c


# ---------------------------------------------------------------------------
# 2. Reduced potential f(u) and its critical points
# ---------------------------------------------------------------------------
def f_reduced(u: np.ndarray, cfg: G2PotentialConfig, kappa: float) -> np.ndarray:
    """Reduced one-variable potential f(u) = mu2*u + kappa*u^2 + Delta*u^3.

    Args:
        u: array of u = r^2 >= 0 values.
        cfg: model configuration.
        kappa: quartic coupling.

    Complexity:
        Time: O(len(u)). Space: O(len(u)).
    """
    u = np.asarray(u, dtype=np.float64)
    if np.any(u < 0.0):
        raise ValueError("u = r^2 must be non-negative.")
    return cfg.mu2 * u + kappa * u**2 + cfg.Delta * u**3


def u_plus(cfg: G2PotentialConfig, kappa: float) -> float:
    """Positive local-minimizer branch u_+(kappa) of f'(u)=0 (Eq. 3.1).

    Raises:
        ValueError: if kappa^2 < 3*Delta*mu2 (no real positive branch exists,
            i.e. kappa is above the saddle-coalescence point).

    Complexity:
        Time: O(1). Space: O(1).
    """
    disc = kappa**2 - 3.0 * cfg.Delta * cfg.mu2
    if disc < 0.0:
        raise ValueError(
            f"No real critical branch: kappa^2={kappa**2:.6g} < 3*Delta*mu2="
            f"{3.0 * cfg.Delta * cfg.mu2:.6g}."
        )
    return (-kappa + math.sqrt(disc)) / (3.0 * cfg.Delta)


def ground_state_energy(cfg: G2PotentialConfig, kappa: float) -> float:
    """Exact global minimum energy E0(kappa) (Theorem 3.2).

    E0(kappa) = f(u_+(kappa)) for kappa < kappa_c, else 0.

    Complexity:
        Time: O(1). Space: O(1).
    """
    if not math.isfinite(kappa):
        raise ValueError(f"kappa must be finite; got {kappa}")
    if kappa >= cfg.kappa_c:
        return 0.0
    up = u_plus(cfg, kappa)
    return float(f_reduced(np.array([up]), cfg, kappa)[0])


def phase_of(cfg: G2PotentialConfig, kappa: float, beta: float, window_scale: float = 10.0) -> Phase:
    """Classify kappa into BROKEN / UNBROKEN / CROSSOVER at inverse temperature beta.

    The crossover window has width O(1/beta) around kappa_c (Sec. 6.4);
    `window_scale` sets how many such widths count as "crossover".

    Complexity:
        Time: O(1). Space: O(1).
    """
    m = cfg.mu2 / cfg.Delta
    half_width = window_scale / (beta * m)
    center = kappa_crossover_center(cfg, beta)
    if abs(kappa - center) <= half_width:
        return Phase.CROSSOVER
    return Phase.BROKEN if kappa < cfg.kappa_c else Phase.UNBROKEN


# ---------------------------------------------------------------------------
# 3. Rigorous uniform asymptotic expansion of Z(beta, kappa)  (Theorem 5.1)
# ---------------------------------------------------------------------------
def _f_second_derivative_at_uplus(cfg: G2PotentialConfig, kappa: float, up: float) -> float:
    """f''(u_+) = 2*kappa + 6*Delta*u_+."""
    return 2.0 * kappa + 6.0 * cfg.Delta * up


def C0_constant(cfg: G2PotentialConfig) -> float:
    """Leading origin-term coefficient C0 = (c0*pi/24) * mu2^-14 * Gamma(7) (Eq. C0expl).

    Complexity:
        Time: O(1). Space: O(1).
    """
    gamma7 = math.gamma(7)  # = 6! = 720
    return (cfg.c0 * math.pi / 24.0) * cfg.mu2 ** (-14) * gamma7


def C1_of_kappa(cfg: G2PotentialConfig, kappa: float) -> float:
    """Orbit-term coefficient C1(kappa) (Eq. C1expl), valid for kappa < kappa_coal.

    C1(kappa) = (pi*c0)/(72*lambda_^{3/2}) * u_+^{3/2} / sqrt(f''(u_+)).

    Requires lambda_ > 0, matching the paper's definiteness convention
    (Sec. 5.1: "without loss of generality take lambda_ > 0").

    Complexity:
        Time: O(1). Space: O(1).
    """
    if cfg.lambda_ <= 0.0:
        raise ValueError(
            "C1_of_kappa assumes lambda_ > 0 per the paper's wall-at-theta=pi/6 convention "
            f"(Sec. 5.1); got lambda_={cfg.lambda_}."
        )
    up = u_plus(cfg, kappa)
    fpp = _f_second_derivative_at_uplus(cfg, kappa, up)
    if fpp <= 0.0:
        raise ValueError(
            f"f''(u_+)={fpp:.6g} <= 0: u_+ is not a non-degenerate local minimum at kappa={kappa}."
        )
    return (math.pi * cfg.c0) / (72.0 * cfg.lambda_ ** 1.5) * up ** 1.5 / math.sqrt(fpp)


def kappa_crossover_center(cfg: G2PotentialConfig, beta: float) -> float:
    """Location where the crossover variable u(kappa) of Theorem 6.4 vanishes.

    Theorem 6.4 defines u = beta*m*(kappa - kappa_c) + 5*log(beta) - log(C1(kappa_c)/C0),
    with m = mu2/Delta. The logistic peak (and hence the true center of the
    finite-beta crossover window) sits at u=0, which for large beta is
    displaced from kappa_c by an O(log(beta)/beta) shift. This displacement
    is small but, at the resolution of the O(1/beta) window used elsewhere
    in this module, it is *not* negligible, so we compute it exactly rather
    than approximating the window center by kappa_c itself.

    Complexity:
        Time: O(1). Space: O(1).
    """
    m = cfg.mu2 / cfg.Delta
    C0 = C0_constant(cfg)
    C1c = C1_of_kappa(cfg, cfg.kappa_c)
    shift = 5.0 * math.log(beta) - math.log(C1c / C0)
    # u(kappa) = beta*m*(kappa-kappa_c) + shift; setting u=0 gives kappa = kappa_c - shift/(beta*m).
    return cfg.kappa_c - shift / (beta * m)


def partition_function_asymptotic(
    cfg: G2PotentialConfig, beta: float, kappa: float
) -> float:
    """Uniform low-temperature asymptotic expansion of Z(beta, kappa) (Theorem 5.1).

        Z ~ beta^-7 * C0 + beta^-2 * C1(kappa) * exp(-beta*f(u_+(kappa)))

    Valid for beta large and kappa in a compact set to the right of
    kappa_coal (per Theorem 5.1's hypotheses); the O(beta^-8) and
    O(exp(-c*beta)) remainders are dropped (leading order only).

    Complexity:
        Time: O(1). Space: O(1).
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0; got {beta}")
    C0 = C0_constant(cfg)
    origin_term = beta ** (-7) * C0
    if kappa > cfg.kappa_coal:
        try:
            up = u_plus(cfg, kappa)
            C1 = C1_of_kappa(cfg, kappa)
            fval = float(f_reduced(np.array([up]), cfg, kappa)[0])
            orbit_term = beta ** (-2) * C1 * math.exp(-beta * fval)
        except ValueError as exc:
            logger.warning("Orbit branch unavailable at kappa=%.6g: %s", kappa, exc)
            orbit_term = 0.0
    else:
        orbit_term = 0.0
    return origin_term + orbit_term


# ---------------------------------------------------------------------------
# 4. Fisher information g_{kappa,kappa}(beta, kappa)   (Theorems 6.1-6.4)
# ---------------------------------------------------------------------------
def curvature_A(cfg: G2PotentialConfig, kappa: float) -> float:
    """Broken-phase curvature coefficient A(kappa) = 2*u_+^2 / (kappa + 3*Delta*u_+).

    This is exactly -d^2/dkappa^2 f(u_+(kappa)) (envelope theorem, Theorem 6.1).

    Complexity:
        Time: O(1). Space: O(1).
    """
    up = u_plus(cfg, kappa)
    denom = kappa + 3.0 * cfg.Delta * up
    if denom == 0.0:
        raise ValueError("Degenerate denominator in A(kappa); kappa at coalescence boundary.")
    return 2.0 * up**2 / denom


def fisher_metric_kappakappa(cfg: G2PotentialConfig, beta: float, kappa: float) -> float:
    """Leading-order Fisher information g_{kappa,kappa}(beta, kappa).

    Dispatches on phase (Theorems 6.1, 6.2, 6.3):
        broken:    g ~ beta * A(kappa)
        unbroken:  g = O(beta^-2)  (returns the explicit O(beta^-2) magnitude bound is not
                   a value; we return the theorem's leading finite-beta estimate 0, since
                   the theorem only proves a *bound*, not a closed leading term - see note)
        crossover: g ~ beta^2 * m^2 * logistic(u), per Theorem 6.4.

    Note: In the unbroken phase the paper (Theorem 6.2) proves only the
    bound |g_{kappa,kappa}| <= C*beta^-2, not an explicit leading
    coefficient; we therefore report the rigorous upper bound rather
    than fabricate a nonexistent closed form, flagged via logger.

    Complexity:
        Time: O(1). Space: O(1).
    """
    phase = phase_of(cfg, kappa, beta)
    m = cfg.mu2 / cfg.Delta

    if phase is Phase.BROKEN:
        A = curvature_A(cfg, kappa)
        return beta * A

    if phase is Phase.UNBROKEN:
        logger.info(
            "Unbroken phase at kappa=%.6g: paper proves only |g_kappakappa| <= C*beta^-2 "
            "(Theorem 6.2); returning 0.0 as the leading-order point estimate.",
            kappa,
        )
        return 0.0

    # CROSSOVER: Theorem 6.4 logistic profile.
    C0 = C0_constant(cfg)
    C1c = C1_of_kappa(cfg, cfg.kappa_c)
    c_shift = math.log(C1c / C0)
    u = beta * m * (kappa - cfg.kappa_c) + 5.0 * math.log(beta) - c_shift
    logistic = math.exp(-u) / (1.0 + math.exp(-u)) ** 2
    return beta**2 * m**2 * logistic


# ---------------------------------------------------------------------------
# 5. Kullback-Leibler divergence   (Theorems 7.1-7.3)
# ---------------------------------------------------------------------------
def kl_divergence(
    cfg: G2PotentialConfig, beta: float, kappa: float, kappa_prime: float
) -> float:
    """Leading-order KL(p_{beta,kappa} || p_{beta,kappa'}) (Theorems 7.1-7.3).

    Dispatches on the relative location of kappa, kappa' to kappa_c:
        both broken:   0.5*beta*A(kappa)*(kappa'-kappa)^2               (Thm 7.1)
        both unbroken: O(beta^-2), reported as 0.0 (bound only)          (Thm 7.2)
        cross-phase (kappa<kappa_c<kappa'):
            beta*f_{kappa'}(u_+(kappa)) - 5*log(beta) + log(C0/C1(kappa)) (Thm 7.3)

    Complexity:
        Time: O(1). Space: O(1).
    """
    kc = cfg.kappa_c
    both_broken = kappa < kc and kappa_prime < kc
    both_unbroken = kappa > kc and kappa_prime > kc

    if both_broken:
        A = curvature_A(cfg, kappa)
        return 0.5 * beta * A * (kappa_prime - kappa) ** 2

    if both_unbroken:
        logger.info(
            "Both parameters unbroken (kappa=%.6g, kappa'=%.6g): Theorem 7.2 gives only "
            "|KL| <= C*beta^-2; returning 0.0 as leading-order point estimate.",
            kappa, kappa_prime,
        )
        return 0.0

    if kappa < kc < kappa_prime:
        up = u_plus(cfg, kappa)
        f_kprime_at_up = float(
            cfg.mu2 * up + kappa_prime * up**2 + cfg.Delta * up**3
        )
        C0 = C0_constant(cfg)
        C1 = C1_of_kappa(cfg, kappa)
        return beta * f_kprime_at_up - 5.0 * math.log(beta) + math.log(C0 / C1)

    raise ValueError(
        "kl_divergence requires kappa < kappa' with a well-defined phase ordering; "
        f"got kappa={kappa}, kappa'={kappa_prime}, kappa_c={kc}."
    )


# ---------------------------------------------------------------------------
# 6. Local thermodynamic length of the crossover  (Theorem 7.4: exactly pi)
# ---------------------------------------------------------------------------
def local_thermodynamic_length(
    cfg: G2PotentialConfig, beta: float, log_log_beta_scale: float = 3.0
) -> float:
    """Numerically integrate sqrt(g_kappakappa) over an expanding crossover window.

    Reproduces Theorem 7.4: as beta -> infinity, this integral -> pi,
    independent of all model parameters. The window half-width is
    delta_beta = M_beta / beta with M_beta = log_log_beta_scale * log(log(beta))
    (paper's suggested choice), which grows slower than sqrt(beta) as required.

    Complexity:
        Time: O(N) for an N-point adaptive quadrature (effectively O(1)
              function evaluations dominated by scipy's adaptive quad).
        Space: O(1).
    """
    if beta <= math.e:
        raise ValueError("beta must be large enough that log(log(beta)) is defined and positive.")
    M_beta = log_log_beta_scale * math.log(math.log(beta))
    if M_beta <= 0.0:
        raise ValueError(f"M_beta={M_beta:.6g} must be positive; increase beta.")

    m = cfg.mu2 / cfg.Delta
    # Window width in kappa corresponding to a width M_beta in the u-variable
    # of Theorem 6.4 (dv/dkappa = beta*m), centered on the true crossover
    # peak (see kappa_crossover_center), not on kappa_c itself.
    delta_beta = M_beta / (beta * m)
    center = kappa_crossover_center(cfg, beta)

    def integrand(kappa: float) -> float:
        try:
            g = fisher_metric_kappakappa(cfg, beta, kappa)
        except ValueError:
            return 0.0
        return math.sqrt(max(g, 0.0))

    lo, hi = center - delta_beta, center + delta_beta
    value, _abserr = integrate.quad(
        integrand, lo, hi, limit=400, epsabs=1e-10, epsrel=1e-8
    )
    return float(value)


# ---------------------------------------------------------------------------
# 7. Module self-check entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cfg = G2PotentialConfig(mu2=1.0, lambda_=1.0, nu=2.0)
    logger.info("kappa_c = %.6f, kappa_coal = %.6f", cfg.kappa_c, cfg.kappa_coal)

    beta = 5.0e4
    for kappa in (cfg.kappa_c - 1.0, cfg.kappa_c, cfg.kappa_c + 1.0):
        phase = phase_of(cfg, kappa, beta)
        E0 = ground_state_energy(cfg, kappa)
        g = fisher_metric_kappakappa(cfg, beta, kappa)
        logger.info(
            "kappa=%.4f  phase=%s  E0=%.6f  g_kappakappa=%.6g", kappa, phase.value, E0, g
        )

    length = local_thermodynamic_length(cfg, beta=1.0e6)
    logger.info("Local thermodynamic length at crossover ~ %.6f (theory: pi=%.6f)", length, math.pi)


# requirements.txt
# numpy>=1.26.0
# scipy>=1.11.0
# hypothesis>=6.100.0
# pytest>=8.0.0
