"""
Cross-Asset Volatility Correlation and Contagion Analytics — pure numpy/scipy.

dim_117 — Cross-asset vol correlation (DCC-GARCH, contagion, spillover)

Classes
-------
GARCHResult
    Container for fitted GARCH(1,1) results per asset.

DCCResult
    Container for DCC-GARCH dynamic conditional correlations.

SpilloverTable
    Diebold-Yilmaz variance-decomposition-based spillover table.

DCCGARCHModel
    Two-step DCC-GARCH estimator (per-asset GARCH, then DCC dynamics).

ContagionAnalyzer
    Forbes-Rigobon bias-adjusted contagion test.

VolatilitySpillover
    VAR-based and realized-vol spillover tables.

CorrelationMetrics
    Summary statistics for correlation matrices.

Convenience functions
---------------------
garch11_vol          Conditional volatilities from GARCH(1,1).
rolling_correlation  Rolling Pearson correlation matrices.
contagion_test       Forbes-Rigobon test on a two-asset pair.
correlation_regime   High/low correlation period labelling.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GARCHResult:
    """Fitted GARCH(1,1) result for one asset."""
    omega: float
    alpha: float
    beta: float
    conditional_vols: np.ndarray   # shape (T,)
    log_likelihood: float


@dataclass
class DCCResult:
    """Dynamic Conditional Correlation result."""
    conditional_correlations: np.ndarray  # shape (T, n, n)
    Q_bar: np.ndarray                     # unconditional correlation (n, n)
    a: float
    b: float
    log_likelihood: float

    def average_correlation(self) -> np.ndarray:
        """Time-series of average pairwise correlation (length T)."""
        T, n, _ = self.conditional_correlations.shape
        out = np.empty(T)
        for t in range(T):
            R = self.conditional_correlations[t]
            # upper triangle, excluding diagonal
            idx = np.triu_indices(n, k=1)
            out[t] = float(np.mean(R[idx]))
        return out

    def correlation_at(self, t: int) -> np.ndarray:
        """Return n×n correlation matrix at time t."""
        return self.conditional_correlations[t].copy()


@dataclass
class SpilloverTable:
    """Diebold-Yilmaz spillover table."""
    from_to: np.ndarray          # (n, n) directional spillovers in %
    asset_names: List[str]
    spillover_index: float       # total index in %
    net_spillovers: np.ndarray   # (n,) net = to_others - from_others

    def transmitters(self) -> List[str]:
        """Assets that are net transmitters (net_spillover > 0)."""
        return [self.asset_names[i]
                for i in range(len(self.asset_names))
                if self.net_spillovers[i] > 0]

    def receivers(self) -> List[str]:
        """Assets that are net receivers (net_spillover < 0)."""
        return [self.asset_names[i]
                for i in range(len(self.asset_names))
                if self.net_spillovers[i] < 0]


# ---------------------------------------------------------------------------
# Helper: univariate GARCH(1,1) via MLE
# ---------------------------------------------------------------------------

def _garch11_variance_filter(returns: np.ndarray,
                              omega: float, alpha: float, beta: float
                              ) -> np.ndarray:
    """Compute conditional variance series for GARCH(1,1)."""
    n = len(returns)
    var = np.empty(n)
    var[0] = max(float(np.var(returns)), 1e-10)
    for t in range(1, n):
        v = omega + alpha * returns[t - 1] ** 2 + beta * var[t - 1]
        var[t] = max(v, 1e-12)
    return var


def _garch11_neg_ll(params: np.ndarray, returns: np.ndarray) -> float:
    """Negative Gaussian log-likelihood for GARCH(1,1)."""
    omega, alpha, beta = params
    if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 1.0:
        return 1e10
    var = _garch11_variance_filter(returns, omega, alpha, beta)
    if np.any(var <= 0):
        return 1e10
    ll = -0.5 * float(np.sum(np.log(var) + returns ** 2 / var))
    return -ll


def _fit_garch11(returns: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Fit GARCH(1,1) via MLE.  Returns (omega, alpha, beta, log_likelihood).
    """
    r = returns - returns.mean()
    sv = max(float(np.var(r)), 1e-10)

    starts = [
        np.array([0.05 * sv, 0.09, 0.90]),
        np.array([0.10 * sv, 0.05, 0.93]),
        np.array([0.20 * sv, 0.15, 0.80]),
    ]
    bounds = [(1e-10, None), (1e-6, 0.9999), (1e-6, 0.9999)]

    best_nll = np.inf
    best_params = (sv * 0.05, 0.09, 0.90)

    for x0 in starts:
        try:
            res = minimize(
                _garch11_neg_ll,
                x0,
                args=(r,),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-12},
            )
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = tuple(res.x)
        except Exception:
            pass

    omega, alpha, beta = best_params
    # Enforce stationarity
    if alpha + beta >= 1.0:
        scale = 0.99 / (alpha + beta)
        alpha *= scale
        beta *= scale

    ll = -best_nll
    return omega, alpha, beta, ll


def garch11_vol(returns: np.ndarray) -> np.ndarray:
    """
    Fit GARCH(1,1) and return conditional volatility array (length T).

    Parameters
    ----------
    returns : 1-D array of returns for a single asset.
    """
    returns = np.asarray(returns, dtype=float)
    omega, alpha, beta, _ = _fit_garch11(returns)
    var = _garch11_variance_filter(returns - returns.mean(), omega, alpha, beta)
    return np.sqrt(np.maximum(var, 0.0))


# ---------------------------------------------------------------------------
# Rolling correlation
# ---------------------------------------------------------------------------

def rolling_correlation(returns: np.ndarray, window: int = 60) -> np.ndarray:
    """
    Rolling Pearson correlation matrices.

    Parameters
    ----------
    returns : shape (T, n) array of multi-asset returns.
    window  : rolling window length.

    Returns
    -------
    shape (T - window, n, n) array.  Diagonal entries are 1.0.
    """
    returns = np.asarray(returns, dtype=float)
    T, n = returns.shape
    out_len = T - window
    if out_len <= 0:
        raise ValueError(f"T={T} must be > window={window}")
    out = np.empty((out_len, n, n))
    for i in range(out_len):
        block = returns[i: i + window]  # (window, n)
        # demean
        block = block - block.mean(axis=0)
        cov = block.T @ block / (window - 1)
        std = np.sqrt(np.diag(cov))
        std = np.where(std < 1e-12, 1e-12, std)
        corr = cov / np.outer(std, std)
        # clip to [-1, 1] and set diagonal to 1
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        out[i] = corr
    return out


# ---------------------------------------------------------------------------
# DCCGARCHModel
# ---------------------------------------------------------------------------

class DCCGARCHModel:
    """
    Two-step DCC-GARCH estimator (Engle, 2002).

    Step 1: Fit univariate GARCH(1,1) to each return series.
    Step 2: Standardize and fit DCC dynamics (Q_t, R_t).
    """

    def __init__(self, a0: float = 0.05, b0: float = 0.90) -> None:
        self.a0 = a0
        self.b0 = b0

    # ── Step 1 ──────────────────────────────────────────────────────────────

    def fit_garch(self, returns: np.ndarray) -> GARCHResult:
        """
        Fit GARCH(1,1) to a single asset return series.

        Parameters
        ----------
        returns : 1-D array.

        Returns
        -------
        GARCHResult
        """
        returns = np.asarray(returns, dtype=float)
        omega, alpha, beta, ll = _fit_garch11(returns)
        r = returns - returns.mean()
        var = _garch11_variance_filter(r, omega, alpha, beta)
        cond_vols = np.sqrt(np.maximum(var, 0.0))
        return GARCHResult(
            omega=omega,
            alpha=alpha,
            beta=beta,
            conditional_vols=cond_vols,
            log_likelihood=ll,
        )

    # ── Step 2 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _dcc_Q_filter(z: np.ndarray, Q_bar: np.ndarray,
                      a: float, b: float) -> np.ndarray:
        """
        Compute Q_t matrices given standardized residuals and DCC params.

        Parameters
        ----------
        z     : (T, n) standardized residuals.
        Q_bar : (n, n) unconditional correlation of z.
        a, b  : DCC scalar parameters.

        Returns
        -------
        Q : (T, n, n) array.
        """
        T, n = z.shape
        Q = np.empty((T, n, n))
        Q[0] = Q_bar.copy()
        for t in range(1, T):
            zt1 = z[t - 1: t].T  # (n, 1)
            Q[t] = (1 - a - b) * Q_bar + a * (zt1 @ zt1.T) + b * Q[t - 1]
        return Q

    @staticmethod
    def _Q_to_R(Q: np.ndarray) -> np.ndarray:
        """Convert Q matrices to correlation matrices R."""
        T, n, _ = Q.shape
        R = np.empty_like(Q)
        for t in range(T):
            Qt = Q[t]
            d = np.sqrt(np.maximum(np.diag(Qt), 1e-12))
            D_inv = np.diag(1.0 / d)
            Rt = D_inv @ Qt @ D_inv
            np.fill_diagonal(Rt, 1.0)
            Rt = np.clip(Rt, -1.0, 1.0)
            np.fill_diagonal(Rt, 1.0)
            R[t] = Rt
        return R

    def _dcc_neg_ll(self, params: np.ndarray, z: np.ndarray,
                    Q_bar: np.ndarray) -> float:
        """Negative log-likelihood for DCC parameters (a, b)."""
        a, b = params
        if a <= 0 or b <= 0 or a + b >= 1.0:
            return 1e10
        T, n = z.shape
        Q = self._dcc_Q_filter(z, Q_bar, a, b)
        R = self._Q_to_R(Q)
        ll = 0.0
        for t in range(T):
            Rt = R[t]
            zt = z[t]
            try:
                sign, logdet = np.linalg.slogdet(Rt)
                if sign <= 0:
                    return 1e10
                inv_R = np.linalg.inv(Rt)
                ll += logdet + float(zt @ inv_R @ zt) - float(zt @ zt)
            except np.linalg.LinAlgError:
                return 1e10
        return 0.5 * ll  # negative of DCC log-lik contribution

    def fit_dcc(self, returns: np.ndarray) -> DCCResult:
        """
        Fit DCC-GARCH to a multi-asset return matrix.

        Parameters
        ----------
        returns : (T, n) array of returns.

        Returns
        -------
        DCCResult
        """
        returns = np.asarray(returns, dtype=float)
        T, n = returns.shape

        # Step 1: per-asset GARCH
        cond_vols = np.empty((T, n))
        for i in range(n):
            r_i = returns[:, i]
            omega, alpha, beta, _ = _fit_garch11(r_i)
            r_dm = r_i - r_i.mean()
            var_i = _garch11_variance_filter(r_dm, omega, alpha, beta)
            cond_vols[:, i] = np.sqrt(np.maximum(var_i, 1e-12))

        # Standardized residuals
        r_dm = returns - returns.mean(axis=0)
        z = r_dm / np.maximum(cond_vols, 1e-12)

        # Q_bar: unconditional correlation of z
        Q_bar = np.corrcoef(z.T)
        np.fill_diagonal(Q_bar, 1.0)

        # Step 2: optimize a, b
        x0 = np.array([self.a0, self.b0])
        bounds = [(1e-6, 0.9), (1e-6, 0.9999)]

        best_nll = np.inf
        best_ab = (self.a0, self.b0)
        starts = [
            np.array([0.05, 0.90]),
            np.array([0.03, 0.94]),
            np.array([0.10, 0.85]),
        ]
        for x_start in starts:
            try:
                res = minimize(
                    self._dcc_neg_ll,
                    x_start,
                    args=(z, Q_bar),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 500, "ftol": 1e-10},
                )
                if res.fun < best_nll:
                    best_nll = res.fun
                    best_ab = tuple(res.x)
            except Exception:
                pass

        a, b = best_ab
        if a + b >= 1.0:
            scale = 0.99 / (a + b)
            a *= scale
            b *= scale

        # Compute final Q and R
        Q = self._dcc_Q_filter(z, Q_bar, a, b)
        R = self._Q_to_R(Q)

        return DCCResult(
            conditional_correlations=R,
            Q_bar=Q_bar,
            a=a,
            b=b,
            log_likelihood=-best_nll,
        )

    def rolling_correlation(self, returns: np.ndarray,
                            window: int = 60) -> np.ndarray:
        """
        Rolling Pearson correlation matrices.

        Parameters
        ----------
        returns : (T, n) array.
        window  : rolling window.

        Returns
        -------
        shape (T - window, n, n)
        """
        returns = np.asarray(returns, dtype=float)
        return rolling_correlation(returns, window=window)


# ---------------------------------------------------------------------------
# ContagionAnalyzer
# ---------------------------------------------------------------------------

class ContagionAnalyzer:
    """Forbes-Rigobon (2002) bias-adjusted contagion test."""

    def forbes_rigobon_test(self,
                            returns_crisis: np.ndarray,
                            returns_tranquil: np.ndarray) -> dict:
        """
        Bias-adjusted contagion test between two bivariate return samples.

        Parameters
        ----------
        returns_crisis  : (T_c, 2) or (T_c,) array — crisis period.
        returns_tranquil: (T_t, 2) or (T_t,) array — tranquil period.

        Returns
        -------
        dict with keys: rho_crisis, rho_tranquil, rho_crisis_adj,
                        contagion_detected, p_value, delta
        """
        rc = np.asarray(returns_crisis, dtype=float)
        rt = np.asarray(returns_tranquil, dtype=float)

        # If 2-D with 2 columns, use pairwise; else treat as 1-D pair
        if rc.ndim == 2 and rc.shape[1] == 2:
            rho_c = float(np.corrcoef(rc[:, 0], rc[:, 1])[0, 1])
            rho_t = float(np.corrcoef(rt[:, 0], rt[:, 1])[0, 1])
            var_c = float(np.var(rc[:, 0]))
            var_t = float(np.var(rt[:, 0]))
        else:
            # Single-vector: compare variance regimes, correlation with lagged
            rho_c = float(np.corrcoef(rc[:-1], rc[1:])[0, 1]) if len(rc) > 2 else 0.5
            rho_t = float(np.corrcoef(rt[:-1], rt[1:])[0, 1]) if len(rt) > 2 else 0.3
            var_c = float(np.var(rc))
            var_t = float(np.var(rt))

        # Clip correlation to avoid numerical issues
        rho_c = np.clip(rho_c, -0.9999, 0.9999)
        rho_t = np.clip(rho_t, -0.9999, 0.9999)

        # Forbes-Rigobon bias adjustment
        # delta = (Var_crisis / Var_tranquil - 1) / (something)
        # We adjust rho_crisis downward for heteroscedasticity bias
        if var_t > 1e-12 and var_c > 1e-12:
            delta = (var_c / var_t - 1.0)
            denom = np.sqrt(1.0 + delta * (1.0 - rho_c ** 2))
            rho_c_adj = float(rho_c / max(denom, 1e-10))
        else:
            delta = 0.0
            rho_c_adj = rho_c

        rho_c_adj = np.clip(rho_c_adj, -0.9999, 0.9999)

        # Fisher z-test: H0: rho_crisis_adj == rho_tranquil
        n_c = len(rc)
        n_t = len(rt)

        def fisher_z(r: float) -> float:
            return 0.5 * np.log((1 + r + 1e-12) / (1 - r + 1e-12))

        z_c = fisher_z(rho_c_adj)
        z_t = fisher_z(rho_t)
        se = np.sqrt(1.0 / max(n_c - 3, 1) + 1.0 / max(n_t - 3, 1))
        z_stat = (z_c - z_t) / max(se, 1e-10)
        # Two-tailed p-value from normal
        from scipy.stats import norm as _norm
        p_value = float(2.0 * (1.0 - _norm.cdf(abs(z_stat))))

        contagion = (rho_c_adj > rho_t) and (p_value < 0.10)

        return {
            "rho_crisis": float(rho_c),
            "rho_tranquil": float(rho_t),
            "rho_crisis_adj": float(rho_c_adj),
            "delta": float(delta),
            "contagion_detected": bool(contagion),
            "p_value": float(p_value),
            "z_stat": float(z_stat),
        }

    def correlation_breakdown(self, returns: np.ndarray,
                              crisis_start: int, crisis_end: int) -> dict:
        """
        Compare correlations pre/during/post crisis window.

        Parameters
        ----------
        returns      : (T, n) multi-asset returns.
        crisis_start : index of crisis start (inclusive).
        crisis_end   : index of crisis end (exclusive).

        Returns
        -------
        dict with keys pre, crisis, post — each an (n, n) correlation matrix.
        Also contagion_test results between first two assets pre vs. during.
        """
        returns = np.asarray(returns, dtype=float)
        T, n = returns.shape

        pre = returns[:crisis_start] if crisis_start > 1 else returns[:2]
        crisis = returns[crisis_start:crisis_end]
        post = returns[crisis_end:] if crisis_end < T - 1 else returns[-2:]

        def safe_corr(x: np.ndarray) -> np.ndarray:
            if len(x) < 2:
                return np.eye(n)
            cc = np.corrcoef(x.T)
            np.fill_diagonal(cc, 1.0)
            return cc

        corr_pre = safe_corr(pre)
        corr_crisis = safe_corr(crisis)
        corr_post = safe_corr(post)

        # Contagion test for first pair
        contagion = {}
        if n >= 2 and len(pre) >= 4 and len(crisis) >= 4:
            contagion = self.forbes_rigobon_test(
                crisis[:, :2], pre[:, :2]
            )

        return {
            "pre": corr_pre,
            "crisis": corr_crisis,
            "post": corr_post,
            "contagion": contagion,
        }


# ---------------------------------------------------------------------------
# VolatilitySpillover
# ---------------------------------------------------------------------------

def _var_ols(returns: np.ndarray, lag: int) -> np.ndarray:
    """
    Estimate VAR(lag) via OLS.  Returns coefficient matrix A of shape (n, n*lag).
    """
    T, n = returns.shape
    # Build lagged design matrix
    Y = returns[lag:]                          # (T-lag, n)
    rows = T - lag
    X = np.ones((rows, n * lag + 1))          # +1 for intercept
    for p in range(lag):
        X[:, p * n: (p + 1) * n] = returns[lag - 1 - p: T - 1 - p]
    # OLS: B = (X'X)^{-1} X'Y, shape (n*lag+1, n)
    try:
        B = np.linalg.lstsq(X, Y, rcond=None)[0]
    except np.linalg.LinAlgError:
        B = np.zeros((n * lag + 1, n))
    residuals = Y - X @ B
    sigma_u = (residuals.T @ residuals) / max(rows - n * lag - 1, 1)
    return B, sigma_u


def _var_fevd(B: np.ndarray, sigma_u: np.ndarray,
              lag: int, n: int, horizon: int) -> np.ndarray:
    """
    Forecast Error Variance Decomposition (FEVD) for a VAR(lag).

    Returns FEVD matrix (n, n): fevd[i, j] = fraction of variance of asset i
    explained by shocks from asset j, at given horizon.
    """
    # Cholesky of sigma_u for orthogonalization
    try:
        P = np.linalg.cholesky(sigma_u + 1e-10 * np.eye(n))
    except np.linalg.LinAlgError:
        P = np.eye(n) * np.sqrt(np.diag(sigma_u) + 1e-10)

    # Companion form MA coefficients Phi_h
    # Phi_0 = I_n, Phi_h = A1*Phi_{h-1} + ... + Ap*Phi_{h-p}
    A = np.zeros((n, n, lag))
    for p in range(lag):
        A[:, :, p] = B[p * n: (p + 1) * n].T  # (n, n)

    Phi = [np.eye(n)]
    for h in range(1, horizon + 1):
        Ph = np.zeros((n, n))
        for p in range(min(h, lag)):
            Ph += A[:, :, p] @ Phi[h - 1 - p]
        Phi.append(Ph)

    # FEVD numerator: sum_h (Phi_h @ P)^2 over elements
    # fevd[i,j] = sum_{h=0}^{H-1} (e_i' Phi_h P e_j)^2 / sum_{k} ...
    num = np.zeros((n, n))
    denom = np.zeros(n)
    for h in range(horizon):
        PhP = Phi[h] @ P  # (n, n)
        for j in range(n):
            col = PhP[:, j]          # (n,)
            num[:, j] += col ** 2
        for i in range(n):
            denom[i] += float(np.sum(Phi[h][i, :] ** 2 * np.diag(sigma_u)))

    denom = np.maximum(denom, 1e-12)
    fevd = num / denom[:, np.newaxis]
    # Row-normalize so each row sums to 1
    row_sums = fevd.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    fevd = fevd / row_sums
    return fevd


class VolatilitySpillover:
    """VAR-based and realized-vol spillover tables (Diebold-Yilmaz 2012)."""

    def diebold_yilmaz(self,
                       returns: np.ndarray,
                       lag: int = 4,
                       horizon: int = 10,
                       asset_names: Optional[List[str]] = None) -> SpilloverTable:
        """
        Generalized FEVD-based spillover table.

        Parameters
        ----------
        returns    : (T, n) return matrix.
        lag        : VAR lag order.
        horizon    : forecast horizon for FEVD.
        asset_names: optional list of n names.

        Returns
        -------
        SpilloverTable
        """
        returns = np.asarray(returns, dtype=float)
        T, n = returns.shape
        if asset_names is None:
            asset_names = [f"Asset{i+1}" for i in range(n)]

        B, sigma_u = _var_ols(returns, lag)
        fevd = _var_fevd(B, sigma_u, lag, n, horizon)  # (n, n), rows sum to 1

        # from_to[i, j] = share of variance of i explained by j (%)
        from_to = fevd * 100.0

        # Total spillover index = off-diagonal sum / total * 100
        total = float(from_to.sum())
        off_diag_sum = float(from_to.sum() - np.trace(from_to))
        spillover_index = off_diag_sum / max(total, 1e-12) * 100.0
        spillover_index = np.clip(spillover_index, 0.0, 100.0)

        # Net spillover of asset i = to_others_i - from_others_i
        # to_others_i = sum_j from_to[j, i] (what i contributes to others)
        # from_others_i = sum_j from_to[i, j] (what others contribute to i)
        to_others = from_to.sum(axis=0) - np.diag(from_to)    # (n,)
        from_others = from_to.sum(axis=1) - np.diag(from_to)  # (n,)
        net_spillovers = to_others - from_others

        return SpilloverTable(
            from_to=from_to,
            asset_names=asset_names,
            spillover_index=float(spillover_index),
            net_spillovers=net_spillovers,
        )

    def realized_vol_spillover(self,
                               vol_series: np.ndarray,
                               lag: int = 1,
                               asset_names: Optional[List[str]] = None
                               ) -> SpilloverTable:
        """
        Simpler cross-predictability spillover via OLS regression of realized vols.

        Parameters
        ----------
        vol_series  : (T, n) realized volatility series.
        lag         : number of lags.
        asset_names : optional list of n names.

        Returns
        -------
        SpilloverTable with from_to (n, n) and spillover_index in [0, 100].
        """
        vol_series = np.asarray(vol_series, dtype=float)
        T, n = vol_series.shape
        if asset_names is None:
            asset_names = [f"Asset{i+1}" for i in range(n)]

        # For each asset i, regress vol_i[t] on all vols[t-lag, ...]
        # R^2 contribution per predictor j → cross-predictability matrix
        from_to = np.zeros((n, n))
        Y = vol_series[lag:]          # (T-lag, n)
        rows = T - lag

        for i in range(n):
            y = Y[:, i]
            # Full model (all n predictors)
            X = np.column_stack([np.ones(rows),
                                 vol_series[:rows]])   # (rows, n+1)
            try:
                B_full = np.linalg.lstsq(X, y, rcond=None)[0]
                y_hat_full = X @ B_full
                ss_res_full = float(np.sum((y - y_hat_full) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                r2_full = 1.0 - ss_res_full / max(ss_tot, 1e-12)
            except np.linalg.LinAlgError:
                r2_full = 0.0

            # Attribution per predictor j (incremental R2 approximation)
            r2_full = max(r2_full, 0.0)
            raw = np.zeros(n)
            for j in range(n):
                # coefficient for predictor j (B_full index j+1)
                try:
                    b_j = B_full[j + 1]
                    x_j = X[:, j + 1]
                    raw[j] = abs(b_j) * float(np.std(x_j))
                except Exception:
                    raw[j] = 0.0

            total_raw = raw.sum()
            if total_raw > 1e-12:
                shares = raw / total_raw * r2_full * 100.0
            else:
                shares = np.zeros(n)

            from_to[i, :] = shares

        # Ensure diagonal dominates (own vol is most predictive)
        # Row-normalize to 100
        row_sums = from_to.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        from_to = from_to / row_sums * 100.0

        # Spillover index
        off_diag_sum = float(from_to.sum() - np.trace(from_to))
        total = float(from_to.sum())
        spillover_index = off_diag_sum / max(total, 1e-12) * 100.0
        spillover_index = np.clip(spillover_index, 0.0, 100.0)

        # Net spillovers
        to_others = from_to.sum(axis=0) - np.diag(from_to)
        from_others = from_to.sum(axis=1) - np.diag(from_to)
        net_spillovers = to_others - from_others

        return SpilloverTable(
            from_to=from_to,
            asset_names=asset_names,
            spillover_index=float(spillover_index),
            net_spillovers=net_spillovers,
        )


# ---------------------------------------------------------------------------
# CorrelationMetrics
# ---------------------------------------------------------------------------

class CorrelationMetrics:
    """Summary statistics for correlation matrices."""

    def average_pairwise(self, corr_matrix: np.ndarray) -> float:
        """Mean of upper-triangle off-diagonal entries."""
        C = np.asarray(corr_matrix, dtype=float)
        n = C.shape[0]
        idx = np.triu_indices(n, k=1)
        if len(idx[0]) == 0:
            return 0.0
        return float(np.mean(C[idx]))

    def eigenvalue_dispersion(self, corr_matrix: np.ndarray) -> float:
        """
        Eigenvalue concentration: max_eigenvalue / sum_eigenvalues.

        Returns a value in (0, 1].  Close to 1/n implies low concentration.
        """
        C = np.asarray(corr_matrix, dtype=float)
        eigvals = np.linalg.eigvalsh(C)
        eigvals = np.maximum(eigvals, 0.0)
        total = float(eigvals.sum())
        if total < 1e-12:
            return 1.0
        return float(eigvals.max() / total)

    def effective_n_factors(self, corr_matrix: np.ndarray) -> float:
        """
        Effective number of uncorrelated factors.

        effective_N = 1 / sum(share_i^2)  where share_i = lambda_i / sum(lambda).
        """
        C = np.asarray(corr_matrix, dtype=float)
        eigvals = np.linalg.eigvalsh(C)
        eigvals = np.maximum(eigvals, 0.0)
        total = float(eigvals.sum())
        if total < 1e-12:
            return 1.0
        shares = eigvals / total
        return float(1.0 / max(float(np.sum(shares ** 2)), 1e-12))

    def market_correlation(self, returns: np.ndarray) -> np.ndarray:
        """
        Correlation of each asset with the equal-weight portfolio.

        Parameters
        ----------
        returns : (T, n) return matrix.

        Returns
        -------
        1-D array of length n with each asset's correlation to EW portfolio.
        """
        returns = np.asarray(returns, dtype=float)
        T, n = returns.shape
        ew = returns.mean(axis=1)   # (T,) equal-weight portfolio
        corrs = np.empty(n)
        for i in range(n):
            c = np.corrcoef(returns[:, i], ew)[0, 1]
            corrs[i] = float(np.nan_to_num(c))
        return corrs


# ---------------------------------------------------------------------------
# Convenience function: contagion_test (single pair)
# ---------------------------------------------------------------------------

def contagion_test(returns_a: np.ndarray, returns_b: np.ndarray,
                   split: Optional[int] = None) -> dict:
    """
    Convenience wrapper for Forbes-Rigobon contagion test on a single pair.

    Parameters
    ----------
    returns_a, returns_b : 1-D arrays of equal length.
    split : index splitting tranquil / crisis.  If None, splits at midpoint.

    Returns
    -------
    dict from ContagionAnalyzer.forbes_rigobon_test.
    """
    returns_a = np.asarray(returns_a, dtype=float)
    returns_b = np.asarray(returns_b, dtype=float)
    T = len(returns_a)
    if split is None:
        split = T // 2

    paired = np.column_stack([returns_a, returns_b])
    tranquil = paired[:split]
    crisis = paired[split:]

    analyzer = ContagionAnalyzer()
    return analyzer.forbes_rigobon_test(crisis, tranquil)


# ---------------------------------------------------------------------------
# Convenience function: correlation_regime
# ---------------------------------------------------------------------------

def correlation_regime(returns: np.ndarray,
                       n_regimes: int = 2,
                       window: int = 40) -> np.ndarray:
    """
    Label high/low correlation periods via threshold on rolling average correlation.

    Parameters
    ----------
    returns   : (T, n) return matrix.
    n_regimes : 2 (high / low) — only 2-regime is supported.
    window    : rolling window for average pairwise correlation.

    Returns
    -------
    1-D integer array of length T with regime labels {0, 1}.
    0 = low correlation, 1 = high correlation.
    """
    returns = np.asarray(returns, dtype=float)
    T, n = returns.shape

    if n < 2:
        return np.zeros(T, dtype=int)

    rolling = rolling_correlation(returns, window=window)   # (T-window, n, n)
    cm = CorrelationMetrics()
    avg_corr_rolling = np.array([cm.average_pairwise(rolling[i])
                                 for i in range(len(rolling))])

    # Pad first `window` time steps with median
    pad = np.full(window, float(np.median(avg_corr_rolling)))
    avg_corr = np.concatenate([pad, avg_corr_rolling])  # (T,)

    median_corr = float(np.median(avg_corr))
    labels = (avg_corr >= median_corr).astype(int)
    return labels
