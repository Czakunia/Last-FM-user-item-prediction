"""Binary pair measures: orthonormal HCR a11 vs LEGACY_BINARY_ASSOC_V1.

Canonical HCR for binary–binary is the single non-constant orthonormal
coefficient a11 = E[φ1(X) φ1(Y)].  The historical scalar phi**2 + MI is kept
only as ``legacy_binary_assoc`` (protocol LEGACY_BINARY_ASSOC_V1) and must not
be named HCR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEGACY_PROTOCOL = "LEGACY_BINARY_ASSOC_V1"
HCR_PROTOCOL = "LASTFM_ORTHONORMAL_HCR_V2"

# Ordered block for PairMLP: signed a11 and energy are separate channels.
# Mask is included so the encoder can down-weight invalid pairs.
COMPACT10_ORDER = (
    "p11",
    "conditional_y_given_x",
    "risk_difference",
    "scaled_log_odds_ratio",
    "hcr_a11",
    "hcr_energy",
    "mutual_information",
    "support",
    "uncertainty",
    "valid_mask",
)
# Task-A-style 8-d without energy/mask (ablation / HT2).
COMPACT8_ORDER = (
    "p11",
    "conditional_y_given_x",
    "risk_difference",
    "scaled_log_odds_ratio",
    "hcr_a11",
    "mutual_information",
    "support",
    "uncertainty",
)


@dataclass(frozen=True)
class BinaryPairMeasures:
    p11: float
    conditional_y_given_x: float
    risk_difference: float
    scaled_log_odds_ratio: float

    hcr_a11: float
    hcr_energy: float
    mutual_information: float

    support: float
    uncertainty: float
    valid_mask: float

    n11: int
    n10: int
    n01: int
    n00: int

    # Compatibility / Stage B–F baseline only — never call this HCR.
    legacy_binary_assoc: float

    # Extra diagnostics used by older stages
    lift: float
    odds_ratio: float
    npmi: float
    # Alias of hcr_a11 (signed phi / orthonormal coefficient)
    phi: float

    def compact10(self) -> np.ndarray:
        """Primary model block: keeps sign(a11) and energy as separate channels."""

        return np.asarray(
            [
                self.p11,
                self.conditional_y_given_x,
                self.risk_difference,
                self.scaled_log_odds_ratio,
                self.hcr_a11,
                self.hcr_energy,
                self.mutual_information,
                self.support,
                self.uncertainty,
                self.valid_mask,
            ],
            dtype=np.float32,
        )

    def compact8(self) -> np.ndarray:
        """Task-A compact without energy/mask (ablation HT2 / H3)."""

        return np.asarray(
            [
                self.p11,
                self.conditional_y_given_x,
                self.risk_difference,
                self.scaled_log_odds_ratio,
                self.hcr_a11,
                self.mutual_information,
                self.support,
                self.uncertainty,
            ],
            dtype=np.float32,
        )

    def feature_block(self) -> np.ndarray:
        """Default block fed to PairMLP (compact10)."""

        return self.compact10()


# Backward-compatible name used by older imports.
BinaryMeasures = BinaryPairMeasures


def binary_contrast_phi(x: np.ndarray, p: float, *, eps: float = 1e-12) -> np.ndarray:
    """Orthonormal non-constant contrast φ1 for a binary variable."""

    p = float(np.clip(p, eps, 1.0 - eps))
    values = np.asarray(x, dtype=np.float64)
    return (values - p) / np.sqrt(p * (1.0 - p))


def a11_from_basis(
    x: np.ndarray,
    y: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """Reference a11 = mean(φ1(x) φ1(y)) on raw empirical margins."""

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("x and y must be non-empty and aligned")
    p_x = float(np.clip(x.mean(), eps, 1.0 - eps))
    p_y = float(np.clip(y.mean(), eps, 1.0 - eps))
    if p_x <= eps or p_x >= 1.0 - eps or p_y <= eps or p_y >= 1.0 - eps:
        return 0.0
    return float(np.mean(binary_contrast_phi(x, p_x, eps=eps) * binary_contrast_phi(y, p_y, eps=eps)))


def a11_from_contingency(
    n11: int,
    n10: int,
    n01: int,
    n00: int,
    *,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """Fast a11 from contingency; returns (a11, valid_mask).

    Uses raw counts (no Jeffreys smoothing) so orthonormality holds w.r.t.
    the empirical train distribution.
    """

    n11_f = float(n11)
    n10_f = float(n10)
    n01_f = float(n01)
    n00_f = float(n00)
    n_total = n11_f + n10_f + n01_f + n00_f
    n1_dot = n11_f + n10_f
    n0_dot = n01_f + n00_f
    n_dot1 = n11_f + n01_f
    n_dot0 = n10_f + n00_f
    denom = np.sqrt(n1_dot * n0_dot * n_dot1 * n_dot0)
    if n_total <= 0 or denom <= eps:
        return 0.0, 0.0
    a11 = (n_total * n11_f - n1_dot * n_dot1) / denom
    return float(a11), 1.0


def a11_energy_from_n11_batch(
    n11: np.ndarray,
    pop_j: np.ndarray,
    pop_i: int,
    n_users: int,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized raw a11 and energy=a11² from n11 / popularity (no smoothing).

    Matches ``contingency_from_pop`` + ``a11_from_contingency`` elementwise.
    """

    n11_f = np.asarray(n11, dtype=np.float64).reshape(-1)
    pop_j_f = np.asarray(pop_j, dtype=np.float64).reshape(-1)
    if n11_f.shape != pop_j_f.shape:
        raise ValueError("n11 and pop_j must share shape")
    pop_i_f = float(pop_i)
    n_users_f = float(n_users)
    n10 = np.maximum(pop_j_f - n11_f, 0.0)
    n01 = np.maximum(pop_i_f - n11_f, 0.0)
    n00 = np.maximum(n_users_f - pop_j_f - pop_i_f + n11_f, 0.0)
    n_total = n11_f + n10 + n01 + n00
    n1_dot = n11_f + n10
    n0_dot = n01 + n00
    n_dot1 = n11_f + n01
    n_dot0 = n10 + n00
    denom = np.sqrt(n1_dot * n0_dot * n_dot1 * n_dot0)
    a11 = np.zeros(n11_f.shape[0], dtype=np.float64)
    ok = (n_total > 0.0) & (denom > float(eps))
    a11[ok] = (n_total[ok] * n11_f[ok] - n1_dot[ok] * n_dot1[ok]) / denom[ok]
    energy = a11 * a11
    return a11, energy


def a11_energy_from_n11_matrix(
    n11: np.ndarray,
    pop_j: np.ndarray,
    pop_i: np.ndarray,
    n_users: int,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """HxC batch of raw a11 / energy; same contingency rules as 1-d batch helper."""

    n11_f = np.asarray(n11, dtype=np.float64)
    if n11_f.ndim != 2:
        raise ValueError("n11 must be 2-d (H×C)")
    pop_j_f = np.asarray(pop_j, dtype=np.float64).reshape(-1, 1)
    pop_i_f = np.asarray(pop_i, dtype=np.float64).reshape(1, -1)
    if pop_j_f.shape[0] != n11_f.shape[0] or pop_i_f.shape[1] != n11_f.shape[1]:
        raise ValueError("popularity shapes must broadcast to n11")
    n_users_f = float(n_users)
    n10 = np.maximum(pop_j_f - n11_f, 0.0)
    n01 = np.maximum(pop_i_f - n11_f, 0.0)
    n00 = np.maximum(n_users_f - pop_j_f - pop_i_f + n11_f, 0.0)
    n_total = n11_f + n10 + n01 + n00
    n1_dot = n11_f + n10
    n0_dot = n01 + n00
    n_dot1 = n11_f + n01
    n_dot0 = n10 + n00
    denom = np.sqrt(n1_dot * n0_dot * n_dot1 * n_dot0)
    a11 = np.zeros_like(n11_f, dtype=np.float64)
    ok = (n_total > 0.0) & (denom > float(eps))
    a11[ok] = (n_total[ok] * n11_f[ok] - n1_dot[ok] * n_dot1[ok]) / denom[ok]
    energy = a11 * a11
    return a11, energy


def cooc_from_n11_matrix(
    n11: np.ndarray,
    pop_j: np.ndarray | None = None,
    pop_i: np.ndarray | None = None,
) -> np.ndarray:
    """HxC raw common support: score = n11 (no normalization)."""

    n11_f = np.asarray(n11, dtype=np.float64)
    if n11_f.ndim != 2:
        raise ValueError("n11 must be 2-d (H×C)")
    return n11_f.copy()


def jaccard_from_n11_matrix(
    n11: np.ndarray,
    pop_j: np.ndarray,
    pop_i: np.ndarray,
) -> np.ndarray:
    """HxC set-Jaccard on binary user-incidence: |U_j∩U_i| / |U_j∪U_i|.

    Empty union → 0.0 (never NaN/Inf).
    """

    n11_f = np.asarray(n11, dtype=np.float64)
    if n11_f.ndim != 2:
        raise ValueError("n11 must be 2-d (H×C)")
    pop_j_f = np.asarray(pop_j, dtype=np.float64).reshape(-1, 1)
    pop_i_f = np.asarray(pop_i, dtype=np.float64).reshape(1, -1)
    union = pop_j_f + pop_i_f - n11_f
    out = np.zeros_like(n11_f, dtype=np.float64)
    ok = union > 0.0
    out[ok] = n11_f[ok] / union[ok]
    return out


def dice_from_n11_matrix(
    n11: np.ndarray,
    pop_j: np.ndarray,
    pop_i: np.ndarray,
) -> np.ndarray:
    """HxC Dice = 2 n11 / (pop_j + pop_i). Empty denom → 0. Not in main race."""

    n11_f = np.asarray(n11, dtype=np.float64)
    if n11_f.ndim != 2:
        raise ValueError("n11 must be 2-d (H×C)")
    pop_j_f = np.asarray(pop_j, dtype=np.float64).reshape(-1, 1)
    pop_i_f = np.asarray(pop_i, dtype=np.float64).reshape(1, -1)
    denom = pop_j_f + pop_i_f
    out = np.zeros_like(n11_f, dtype=np.float64)
    ok = denom > 0.0
    out[ok] = (2.0 * n11_f[ok]) / denom[ok]
    return out


def cosine_from_n11_matrix(
    n11: np.ndarray,
    pop_j: np.ndarray,
    pop_i: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """HxC cosine on binary incidence vectors: n11 / sqrt(|U_j|·|U_i|).

    Zero-vector (pop_j=0 or pop_i=0) → 0.0.
    """

    n11_f = np.asarray(n11, dtype=np.float64)
    if n11_f.ndim != 2:
        raise ValueError("n11 must be 2-d (H×C)")
    pop_j_f = np.asarray(pop_j, dtype=np.float64).reshape(-1, 1)
    pop_i_f = np.asarray(pop_i, dtype=np.float64).reshape(1, -1)
    denom = np.sqrt(np.maximum(pop_j_f, 0.0) * np.maximum(pop_i_f, 0.0))
    out = np.zeros_like(n11_f, dtype=np.float64)
    ok = denom > float(eps)
    out[ok] = n11_f[ok] / denom[ok]
    return out


def _mi_terms_from_cells(
    n00: np.ndarray,
    n01: np.ndarray,
    n10: np.ndarray,
    n11: np.ndarray,
    *,
    smoothing: float,
    eps: float,
) -> np.ndarray:
    s = float(smoothing)
    c00 = n00 + s
    c01 = n01 + s
    c10 = n10 + s
    c11 = n11 + s
    tot = np.maximum(c00 + c01 + c10 + c11, eps)
    p00, p01, p10, p11 = c00 / tot, c01 / tot, c10 / tot, c11 / tot
    px0, px1 = p00 + p01, p10 + p11
    py0, py1 = p00 + p10, p01 + p11
    if s == 0.0:
        mi = np.zeros_like(n11, dtype=np.float64)
        for pij, pi, pj in (
            (p00, px0, py0),
            (p01, px0, py1),
            (p10, px1, py0),
            (p11, px1, py1),
        ):
            mask = (pij > 0.0) & (pi > 0.0) & (pj > 0.0)
            if not np.any(mask):
                continue
            ratio = pij[mask] / np.maximum(pi[mask] * pj[mask], eps)
            mi[mask] += pij[mask] * np.log(ratio)
        return mi
    return (
        p00 * np.log(p00 / np.maximum(px0 * py0, eps))
        + p01 * np.log(p01 / np.maximum(px0 * py1, eps))
        + p10 * np.log(p10 / np.maximum(px1 * py0, eps))
        + p11 * np.log(p11 / np.maximum(px1 * py1, eps))
    )


def mi_from_n11_matrix(
    n11: np.ndarray,
    pop_j: np.ndarray,
    pop_i: np.ndarray,
    n_users: int,
    *,
    smoothing: float = 0.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """HxC pure Mutual Information (nats) — full binary I(C;H).

    Default ``smoothing=0`` (measure-race freeze): no Jeffreys additive.
    Zero cells: contribution 0 (0·log convention). Natural log.

    This is **not** legacy_binary_assoc (a11²+MI).
    Pass ``smoothing=0.5`` only for legacy channel parity.
    """

    from src.lastfm_lp.binary.contingency import contingency_batch_from_n11_matrix

    batch = contingency_batch_from_n11_matrix(n11, pop_j, pop_i, n_users)
    return _mi_terms_from_cells(
        batch.d, batch.c, batch.b, batch.a, smoothing=smoothing, eps=eps
    )


def npmi11_from_n11_matrix(
    n11: np.ndarray,
    pop_j: np.ndarray,
    pop_i: np.ndarray,
    n_users: int,
    *,
    eps: float = 1e-12,
    zero_n11_score: float = -1.0,
) -> np.ndarray:
    """HxC NPMI on the 11-cell only (not full MI).

    PMI_11 = log(p11 / (p1· p·1)); NPMI_11 = PMI_11 / (-log p11).
    If n11==0: ``zero_n11_score`` (default -1.0). No additive smoothing.
    """

    from src.lastfm_lp.binary.contingency import contingency_batch_from_n11_matrix

    batch = contingency_batch_from_n11_matrix(n11, pop_j, pop_i, n_users)
    n = float(batch.N)
    out = np.full_like(batch.a, float(zero_n11_score), dtype=np.float64)
    if n <= 0:
        return out
    p11 = batch.a / n
    p1dot = (batch.a + batch.b) / n
    pdot1 = (batch.a + batch.c) / n
    ok = (batch.a > 0.0) & (p1dot > 0.0) & (pdot1 > 0.0)
    denom = -np.log(np.maximum(p11, eps))
    pmi = np.log(np.maximum(p11, eps) / np.maximum(p1dot * pdot1, eps))
    npmi = np.zeros_like(p11)
    valid_den = ok & (denom > eps)
    npmi[valid_den] = pmi[valid_den] / denom[valid_den]
    npmi = np.clip(npmi, -1.0, 1.0)
    out[ok] = npmi[ok]
    return out


def g2_from_n11_matrix(
    n11: np.ndarray,
    pop_j: np.ndarray,
    pop_i: np.ndarray,
    n_users: int,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """HxC G² / LLR (unsigned) + association direction.

    G2 = 2 Σ O_ij log(O_ij/E_ij); O=0 → contribution 0.
    direction = sign(n11/N - p_h * p_c). MAIN RACE uses unsigned G2.
    """

    from src.lastfm_lp.binary.contingency import contingency_batch_from_n11_matrix

    batch = contingency_batch_from_n11_matrix(n11, pop_j, pop_i, n_users)
    n = float(batch.N)
    g2 = np.zeros_like(batch.a, dtype=np.float64)
    direction = np.zeros_like(batch.a, dtype=np.float64)
    if n <= 0:
        return g2, direction

    hs = batch.a + batch.b
    cs = batch.a + batch.c
    row1, row0 = hs, n - hs
    col1, col0 = cs, n - cs
    cells = (
        (batch.a, row1 * col1),
        (batch.b, row1 * col0),
        (batch.c, row0 * col1),
        (batch.d, row0 * col0),
    )
    for o, e_num in cells:
        e = e_num / n
        mask = (o > 0.0) & (e > float(eps))
        if not np.any(mask):
            continue
        g2[mask] += o[mask] * np.log(o[mask] / e[mask])
    g2 = 2.0 * g2
    direction = np.sign(batch.a / n - (hs / n) * (cs / n))
    return g2, direction


def g2_positive_from_n11_matrix(
    n11: np.ndarray,
    pop_j: np.ndarray,
    pop_i: np.ndarray,
    n_users: int,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Optional: G2 if n11 > E[n11] else 0. NOT the main-race default."""

    g2, direction = g2_from_n11_matrix(n11, pop_j, pop_i, n_users, eps=eps)
    return np.where(direction > 0, g2, 0.0)


def odds_ratio_from_n11_matrix(
    n11: np.ndarray,
    pop_j: np.ndarray,
    pop_i: np.ndarray,
    n_users: int,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Diagnostic OR = (a·d)/(b·c). Supplementary only."""

    from src.lastfm_lp.binary.contingency import contingency_batch_from_n11_matrix

    batch = contingency_batch_from_n11_matrix(n11, pop_j, pop_i, n_users)
    denom = batch.b * batch.c
    out = np.full_like(batch.a, np.nan, dtype=np.float64)
    ok = denom > float(eps)
    out[ok] = (batch.a[ok] * batch.d[ok]) / denom[ok]
    return out


def phi_coefficient_from_n11_matrix(
    n11: np.ndarray,
    pop_j: np.ndarray,
    pop_i: np.ndarray,
    n_users: int,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Diagnostic phi (= a11). Supplementary only."""

    a11, _ = a11_energy_from_n11_matrix(n11, pop_j, pop_i, n_users, eps=eps)
    return a11


def legacy_assoc_from_n11_matrix(
    n11: np.ndarray,
    pop_j: np.ndarray,
    pop_i: np.ndarray,
    n_users: int,
    *,
    smoothing: float = 0.5,
    eps: float = 1e-12,
) -> np.ndarray:
    """HxC LEGACY_BINARY_ASSOC_V1 = a11² + MI (smoothed MI, raw a11)."""

    n11_f = np.asarray(n11, dtype=np.float64)
    if n11_f.ndim != 2:
        raise ValueError("n11 must be 2-d (H×C)")
    pop_j_f = np.asarray(pop_j, dtype=np.float64).reshape(-1, 1)
    pop_i_f = np.asarray(pop_i, dtype=np.float64).reshape(1, -1)
    _a11, energy = a11_energy_from_n11_matrix(
        n11_f, pop_j_f.ravel(), pop_i_f.ravel(), n_users, eps=eps
    )
    mi = mi_from_n11_matrix(
        n11_f, pop_j_f.ravel(), pop_i_f.ravel(), n_users, smoothing=smoothing, eps=eps
    )
    return energy + mi


def legacy_assoc_from_n11_batch(
    n11: np.ndarray,
    pop_j: np.ndarray,
    pop_i: int,
    n_users: int,
    *,
    smoothing: float = 0.5,
    eps: float = 1e-12,
) -> np.ndarray:
    """1-d legacy assoc vs one candidate (matches matrix helper)."""

    n11_f = np.asarray(n11, dtype=np.float64).reshape(-1, 1)
    pop_j_f = np.asarray(pop_j, dtype=np.float64)
    pop_i_f = np.asarray([pop_i], dtype=np.float64)
    return legacy_assoc_from_n11_matrix(
        n11_f, pop_j_f, pop_i_f, n_users, smoothing=smoothing, eps=eps
    ).reshape(-1)


def measures_from_counts(
    n11: int,
    n10: int,
    n01: int,
    n00: int,
    *,
    smoothing: float = 0.5,
    eps: float = 1e-12,
    uncertainty: float = 0.0,
) -> BinaryPairMeasures:
    """Build compact measures; a11 is raw, other channels may use smoothing."""

    n_total = int(n11 + n10 + n01 + n00)
    a11, valid = a11_from_contingency(n11, n10, n01, n00, eps=eps)
    hcr_energy = float(a11 * a11)

    # Raw p11 / support (manifest: support = log1p(n11))
    p11_raw = float(n11 / max(n_total, 1))
    support = float(np.log1p(n11))

    # Smoothed table for unstable channels only (not for a11).
    counts = np.array([[n00, n01], [n10, n11]], dtype=np.float64) + float(smoothing)
    probs = counts / counts.sum()
    p00, p01 = probs[0]
    p10, p11_s = probs[1]
    p_x1 = p10 + p11_s
    p_x0 = p00 + p01
    p_y1 = p01 + p11_s
    p_y0 = p00 + p10

    cond = float(p11_s / max(p_x1, eps))
    cond_given_x0 = float(p01 / max(p_x0, eps))
    risk_difference = float(cond - cond_given_x0)
    odds = float((p11_s * p00) / max(p10 * p01, eps))
    log_or = float(np.log(max(odds, eps)))
    scaled_log_or = float(np.tanh(log_or / 4.0))

    lift = float(cond / max(p_y1, eps))
    pmi = np.log(max(p11_s, eps) / max(p_x1 * p_y1, eps))
    npmi = float(np.clip(pmi / max(-np.log(max(p11_s, eps)), eps), -1.0, 1.0))

    mi = 0.0
    px = np.array([p_x0, p_x1])
    py = np.array([p_y0, p_y1])
    for i in range(2):
        for j in range(2):
            pij = probs[i, j]
            mi += float(pij * np.log(pij / max(px[i] * py[j], eps)))

    # LEGACY_BINARY_ASSOC_V1 — historical baseline only.
    legacy_binary_assoc = float(a11 * a11 + mi)

    return BinaryPairMeasures(
        p11=p11_raw,
        conditional_y_given_x=cond,
        risk_difference=risk_difference,
        scaled_log_odds_ratio=scaled_log_or,
        hcr_a11=float(a11),
        hcr_energy=hcr_energy,
        mutual_information=float(mi),
        support=support,
        uncertainty=float(uncertainty),
        valid_mask=float(valid),
        n11=int(n11),
        n10=int(n10),
        n01=int(n01),
        n00=int(n00),
        legacy_binary_assoc=legacy_binary_assoc,
        lift=lift,
        odds_ratio=odds,
        npmi=npmi,
        phi=float(a11),
    )
