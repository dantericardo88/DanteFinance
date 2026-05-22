"""
ML Alpha Signal Generation via Gradient Boosting — pure numpy, zero external ML deps.

dim_129 — ML alpha signals (gradient boosting / feature engineering)

Implements a full ML alpha pipeline:
  - CART regression trees from scratch
  - Gradient Boosting Regressor (XGBoost-style MSE objective)
  - Financial feature engineering (momentum, vol, RSI, volume ratio, price/SMA)
  - Signal evaluation (IC, IR, Sharpe, hit-rate, turnover)
  - End-to-end MLAlphaPipeline with backtest

Classes
-------
TreeNode
    Node container for a CART decision tree.

DecisionTree
    CART regression tree: greedy MSE split search, recursive build.
    .fit()               → DecisionTree
    .predict()           → np.ndarray
    .feature_importance()→ np.ndarray (sum=1 if any splits)

GradientBoostingRegressor
    Gradient boosting ensemble (MSE loss) built from DecisionTrees.
    .fit()               → GradientBoostingRegressor
    .predict()           → np.ndarray
    .feature_importance()→ np.ndarray
    .staged_predict()    → List[np.ndarray]

AlphaFeatureEngine
    Financial feature construction from price / volume arrays.
    .momentum()          → np.ndarray
    .volatility()        → np.ndarray
    .rsi()               → np.ndarray
    .volume_ratio()      → np.ndarray
    .price_to_sma()      → np.ndarray
    .build_feature_matrix() → (X, feature_names)

SignalEvaluator
    Evaluation metrics for alpha signals.
    .information_coefficient() → np.ndarray   rolling IC series
    .information_ratio()       → float
    .sharpe()                  → float
    .hit_rate()                → float
    .turnover()                → float

MLAlphaPipeline
    Full train-predict-backtest pipeline.
    .build_dataset()     → (X, y, feature_names)
    .train_test_split()  → (X_tr, X_te, y_tr, y_te)
    .fit()               → MLAlphaPipeline
    .predict_alpha()     → np.ndarray
    .backtest()          → dict

Convenience functions
---------------------
gradient_boosting_alpha   Fit GB model on prices, return predictions + metrics.
compute_ic                Pearson correlation between signal and forward returns.
compute_ir                IC mean / IC std.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# CART Decision Tree
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """Single node in a CART regression tree."""
    feature_idx: int = -1
    threshold: float = 0.0
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    value: float = 0.0      # leaf prediction
    is_leaf: bool = False
    impurity_reduction: float = 0.0  # gain stored for feature importance


class DecisionTree:
    """
    CART regression tree with greedy MSE split search.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth (default 3).
    min_samples_split : int
        Minimum samples required to split an internal node (default 10).
    n_features : int or None
        Number of candidate features per split (None = all features).
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 10,
        n_features: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.seed = seed
        self.root_: Optional[TreeNode] = None
        self.n_features_in_: int = 0
        self._importance: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mse(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        return float(np.var(y))

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> Tuple[int, float, float]:
        """Return (feature_idx, threshold, impurity_reduction)."""
        n_samples, n_features = X.shape
        parent_mse = self._mse(y) * n_samples

        n_cand = self.n_features or n_features
        feature_indices = rng.choice(n_features, size=min(n_cand, n_features), replace=False)

        best_gain = -np.inf
        best_feat = -1
        best_thresh = 0.0

        for feat in feature_indices:
            x_col = X[:, feat]
            thresholds = np.unique(x_col)
            if len(thresholds) <= 1:
                continue
            # Midpoints between sorted unique values
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2.0

            for thresh in thresholds:
                left_mask = x_col <= thresh
                right_mask = ~left_mask
                n_left = left_mask.sum()
                n_right = right_mask.sum()
                if n_left == 0 or n_right == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]
                child_mse = self._mse(y_left) * n_left + self._mse(y_right) * n_right
                gain = parent_mse - child_mse
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh, max(0.0, best_gain)

    def _build(
        self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.Generator
    ) -> TreeNode:
        node = TreeNode()
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or np.all(y == y[0])
        ):
            node.is_leaf = True
            node.value = float(np.mean(y))
            return node

        feat, thresh, gain = self._best_split(X, y, rng)
        if feat == -1 or gain <= 0.0:
            node.is_leaf = True
            node.value = float(np.mean(y))
            return node

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        node.feature_idx = feat
        node.threshold = thresh
        node.impurity_reduction = gain
        self._importance[feat] += gain  # type: ignore[index]
        node.left = self._build(X[left_mask], y[left_mask], depth + 1, rng)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1, rng)
        return node

    def _predict_one(self, x: np.ndarray, node: TreeNode) -> float:
        if node.is_leaf:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)  # type: ignore[arg-type]
        return self._predict_one(x, node.right)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self._importance = np.zeros(self.n_features_in_)
        rng = np.random.default_rng(self.seed)
        self.root_ = self._build(X, y, depth=0, rng=rng)
        # Normalise importance
        total = self._importance.sum()
        if total > 0:
            self._importance /= total
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root_) for x in X])  # type: ignore[arg-type]

    def feature_importance(self) -> np.ndarray:
        if self._importance is None:
            return np.array([])
        return self._importance.copy()


# ---------------------------------------------------------------------------
# Gradient Boosting Regressor
# ---------------------------------------------------------------------------

class GradientBoostingRegressor:
    """
    Gradient Boosting for regression (MSE loss).

    Uses the negative gradient (residuals) as pseudo-targets at each stage.
    Equivalent to simplified XGBoost with MSE objective.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds (trees).
    learning_rate : float
        Shrinkage applied to each tree's contribution.
    max_depth : int
        Max depth of individual trees.
    min_samples_split : int
        Minimum samples to split a node.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 10,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.seed = seed
        self.trees_: List[DecisionTree] = []
        self.init_value_: float = 0.0
        self._importance: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_features = X.shape[1]
        self._importance = np.zeros(n_features)

        self.init_value_ = float(np.mean(y))
        F = np.full(len(y), self.init_value_)
        self.trees_ = []

        for m in range(self.n_estimators):
            # Negative gradient (residuals for MSE)
            residuals = y - F
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                seed=self.seed + m,
            )
            tree.fit(X, residuals)
            predictions = tree.predict(X)
            F = F + self.learning_rate * predictions
            self.trees_.append(tree)
            self._importance += tree.feature_importance()

        # Normalise importance across all trees
        total = self._importance.sum()
        if total > 0:
            self._importance /= total
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.init_value_)
        for tree in self.trees_:
            F = F + self.learning_rate * tree.predict(X)
        return F

    def feature_importance(self) -> np.ndarray:
        if self._importance is None:
            return np.array([])
        imp = self._importance.copy()
        total = imp.sum()
        if total > 0:
            imp /= total
        return imp

    def staged_predict(self, X: np.ndarray) -> List[np.ndarray]:
        """Yield predictions after each boosting stage."""
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.init_value_)
        results = []
        for tree in self.trees_:
            F = F + self.learning_rate * tree.predict(X)
            results.append(F.copy())
        return results


# ---------------------------------------------------------------------------
# Financial Feature Engineering
# ---------------------------------------------------------------------------

class AlphaFeatureEngine:
    """
    Compute financial alpha factors from price and volume arrays.

    Parameters
    ----------
    prices : np.ndarray
        Daily close prices, shape (T,).
    volumes : np.ndarray or None
        Daily traded volumes, shape (T,). Optional.
    """

    def __init__(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> None:
        self.prices = np.asarray(prices, dtype=float)
        self.volumes = np.asarray(volumes, dtype=float) if volumes is not None else None
        self._returns = np.diff(np.log(np.clip(self.prices, 1e-10, None)))
        # Pad with NaN to keep length = T
        self._log_returns = np.concatenate([[np.nan], self._returns])

    # ------------------------------------------------------------------
    # Individual factors — all return arrays of length T
    # ------------------------------------------------------------------

    def momentum(self, lookback: int = 21) -> np.ndarray:
        """Return over past `lookback` trading days (log-return)."""
        T = len(self.prices)
        mom = np.full(T, np.nan)
        for t in range(lookback, T):
            mom[t] = np.log(self.prices[t] / self.prices[t - lookback])
        return mom

    def volatility(self, window: int = 21) -> np.ndarray:
        """Rolling standard deviation of daily log-returns."""
        T = len(self.prices)
        vol = np.full(T, np.nan)
        for t in range(window, T):
            vol[t] = float(np.std(self._returns[t - window: t], ddof=1))
        return vol

    def rsi(self, period: int = 14) -> np.ndarray:
        """Relative Strength Index (0-100)."""
        T = len(self.prices)
        rsi_arr = np.full(T, np.nan)
        rets = np.diff(self.prices)
        gains = np.where(rets > 0, rets, 0.0)
        losses = np.where(rets < 0, -rets, 0.0)

        for t in range(period, T):
            # t-th price corresponds to rets index t-1 (diff is length T-1)
            start = t - period
            avg_gain = np.mean(gains[start:t])
            avg_loss = np.mean(losses[start:t])
            if avg_loss < 1e-12:
                rsi_arr[t] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_arr[t] = 100.0 - 100.0 / (1.0 + rs)
        return rsi_arr

    def volume_ratio(self, window: int = 21) -> np.ndarray:
        """Current volume / rolling average volume."""
        if self.volumes is None:
            return np.full(len(self.prices), np.nan)
        T = len(self.prices)
        vr = np.full(T, np.nan)
        for t in range(window, T):
            avg_vol = np.mean(self.volumes[t - window: t])
            if avg_vol > 1e-12:
                vr[t] = self.volumes[t] / avg_vol
        return vr

    def price_to_sma(self, window: int = 50) -> np.ndarray:
        """Price divided by its simple moving average."""
        T = len(self.prices)
        ratio = np.full(T, np.nan)
        for t in range(window - 1, T):
            sma = np.mean(self.prices[t - window + 1: t + 1])
            if sma > 1e-12:
                ratio[t] = self.prices[t] / sma
        return ratio

    # ------------------------------------------------------------------
    # Combined feature matrix
    # ------------------------------------------------------------------

    def build_feature_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Build aligned feature matrix.

        Returns
        -------
        X : np.ndarray, shape (T_valid, n_features)
            Rows with no NaN across all features.
        feature_names : List[str]
        """
        features: Dict[str, np.ndarray] = {
            "momentum_1m": self.momentum(21),
            "momentum_3m": self.momentum(63),
            "volatility_21d": self.volatility(21),
            "price_to_sma_50": self.price_to_sma(50),
            "rsi_14": self.rsi(14),
        }
        if self.volumes is not None:
            features["volume_ratio_21d"] = self.volume_ratio(21)

        feature_names = list(features.keys())
        raw = np.column_stack([features[k] for k in feature_names])  # (T, n_feat)
        # Keep only rows with no NaN
        valid = ~np.any(np.isnan(raw), axis=1)
        return raw[valid], feature_names


# ---------------------------------------------------------------------------
# Signal Evaluator
# ---------------------------------------------------------------------------

class SignalEvaluator:
    """Compute standard quantitative metrics for alpha signals."""

    @staticmethod
    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        """Pearson correlation with NaN safety."""
        if len(a) < 2:
            return np.nan
        std_a = np.std(a, ddof=1)
        std_b = np.std(b, ddof=1)
        if std_a < 1e-12 or std_b < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def information_coefficient(
        self,
        signals: np.ndarray,
        forward_returns: np.ndarray,
        window: int = 60,
    ) -> np.ndarray:
        """
        Rolling IC between signal and forward returns.

        Returns
        -------
        ic : np.ndarray, length = len(signals) - window + 1
        """
        signals = np.asarray(signals, dtype=float)
        forward_returns = np.asarray(forward_returns, dtype=float)
        T = len(signals)
        ic_series = []
        for t in range(window, T + 1):
            ic_series.append(
                self._safe_corr(signals[t - window: t], forward_returns[t - window: t])
            )
        return np.array(ic_series)

    def information_ratio(
        self, signals: np.ndarray, forward_returns: np.ndarray, window: int = 60
    ) -> float:
        """IR = mean(IC) / std(IC)."""
        ic = self.information_coefficient(signals, forward_returns, window=window)
        ic = ic[~np.isnan(ic)]
        if len(ic) < 2:
            return 0.0
        std_ic = float(np.std(ic, ddof=1))
        if std_ic < 1e-12:
            return 0.0
        return float(np.mean(ic) / std_ic)

    def sharpe(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
        annualize: float = 252.0,
    ) -> float:
        """
        Sharpe of a signal-driven strategy.
        Position = sign(signal); P&L = position * next_return.
        """
        signals = np.asarray(signals, dtype=float)
        returns = np.asarray(returns, dtype=float)
        n = min(len(signals), len(returns))
        pos = np.sign(signals[:n])
        pnl = pos * returns[:n]
        std_pnl = float(np.std(pnl, ddof=1))
        if std_pnl < 1e-12:
            return 0.0
        return float(np.mean(pnl) / std_pnl * np.sqrt(annualize))

    def hit_rate(self, signals: np.ndarray, returns: np.ndarray) -> float:
        """Fraction of trades where sign(signal) == sign(return)."""
        signals = np.asarray(signals, dtype=float)
        returns = np.asarray(returns, dtype=float)
        n = min(len(signals), len(returns))
        mask = signals[:n] != 0
        if mask.sum() == 0:
            return 0.5
        correct = np.sign(signals[:n][mask]) == np.sign(returns[:n][mask])
        return float(np.mean(correct))

    def turnover(self, signals: np.ndarray) -> float:
        """Average absolute daily change in signal (normalised by max magnitude)."""
        signals = np.asarray(signals, dtype=float)
        if len(signals) < 2:
            return 0.0
        changes = np.abs(np.diff(signals))
        max_val = np.abs(signals).max()
        if max_val < 1e-12:
            return 0.0
        return float(np.mean(changes) / max_val)


# ---------------------------------------------------------------------------
# End-to-end ML Alpha Pipeline
# ---------------------------------------------------------------------------

class MLAlphaPipeline:
    """
    Full ML alpha pipeline: feature engineering → gradient boosting → backtest.

    Parameters
    ----------
    prices : np.ndarray
        Daily prices, shape (T,).
    volumes : np.ndarray or None
        Daily volumes, shape (T,).
    """

    def __init__(
        self, prices: np.ndarray, volumes: Optional[np.ndarray] = None
    ) -> None:
        self.prices = np.asarray(prices, dtype=float)
        self.volumes = volumes
        self._engine = AlphaFeatureEngine(self.prices, self.volumes)
        self._evaluator = SignalEvaluator()
        self._model: Optional[GradientBoostingRegressor] = None
        self._X_train: Optional[np.ndarray] = None
        self._X_test: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None
        self._feature_names: List[str] = []

    def build_dataset(
        self, forward_days: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build (X, y) where y = forward log-return over `forward_days`.

        Returns
        -------
        X : np.ndarray
        y : np.ndarray
        feature_names : List[str]
        """
        X_full, feature_names = self._engine.build_feature_matrix()
        T = len(self.prices)

        # Align: we need forward returns for each valid feature row.
        # Re-build validity mask to map rows back to original time indices.
        X_all, _ = self._engine.build_feature_matrix()
        feature_names_out = feature_names

        # Compute forward returns for all T steps
        fwd_returns = np.full(T, np.nan)
        log_prices = np.log(np.clip(self.prices, 1e-10, None))
        for t in range(T - forward_days):
            fwd_returns[t] = log_prices[t + forward_days] - log_prices[t]

        # Re-derive valid mask using raw feature matrix
        features_dict = {
            "momentum_1m": self._engine.momentum(21),
            "momentum_3m": self._engine.momentum(63),
            "volatility_21d": self._engine.volatility(21),
            "price_to_sma_50": self._engine.price_to_sma(50),
            "rsi_14": self._engine.rsi(14),
        }
        if self.volumes is not None:
            features_dict["volume_ratio_21d"] = self._engine.volume_ratio(21)

        raw = np.column_stack([features_dict[k] for k in feature_names])
        valid = ~np.any(np.isnan(raw), axis=1) & ~np.isnan(fwd_returns)

        X = raw[valid]
        y = fwd_returns[valid]
        self._feature_names = feature_names
        return X, y, feature_names

    def train_test_split(
        self, X: np.ndarray, y: np.ndarray, test_frac: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)
        split = int(n * (1 - test_frac))
        return X[:split], X[split:], y[:split], y[split:]

    def fit(self, n_estimators: int = 30) -> "MLAlphaPipeline":
        X, y, _ = self.build_dataset()
        X_tr, X_te, y_tr, y_te = self.train_test_split(X, y)
        self._X_train = X_tr
        self._X_test = X_te
        self._y_train = y_tr
        self._y_test = y_te
        self._model = GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=0.05, max_depth=3,
            min_samples_split=5, seed=42
        )
        self._model.fit(X_tr, y_tr)
        return self

    def predict_alpha(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call .fit() first.")
        return self._model.predict(X)

    def backtest(self) -> dict:
        """
        Run backtest on held-out test set.

        Returns
        -------
        dict with keys: sharpe, ir, hit_rate, turnover, feature_importance
        """
        if self._model is None or self._X_test is None:
            raise RuntimeError("Call .fit() first.")
        signals = self._model.predict(self._X_test)
        returns = self._y_test  # type: ignore[assignment]
        ev = self._evaluator
        return {
            "sharpe": ev.sharpe(signals, returns),
            "ir": ev.information_ratio(signals, returns, window=min(60, len(signals) // 3 or 5)),
            "hit_rate": ev.hit_rate(signals, returns),
            "turnover": ev.turnover(signals),
            "feature_importance": self._model.feature_importance().tolist(),
            "feature_names": self._feature_names,
        }


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def gradient_boosting_alpha(
    prices: np.ndarray,
    volumes: Optional[np.ndarray] = None,
    n_estimators: int = 30,
) -> dict:
    """
    Fit gradient boosting alpha model on price (and optional volume) data.

    Parameters
    ----------
    prices : np.ndarray
        Daily prices.
    volumes : np.ndarray or None
        Daily volumes (optional).
    n_estimators : int
        Boosting rounds.

    Returns
    -------
    dict with keys: predictions, feature_importance, sharpe, ir
    """
    pipeline = MLAlphaPipeline(prices, volumes)
    pipeline.fit(n_estimators=n_estimators)
    result = pipeline.backtest()
    predictions = pipeline.predict_alpha(pipeline._X_test)  # type: ignore[arg-type]
    return {
        "predictions": predictions,
        "feature_importance": result["feature_importance"],
        "sharpe": result["sharpe"],
        "ir": result["ir"],
    }


def compute_ic(signals: np.ndarray, forward_returns: np.ndarray) -> float:
    """
    Pearson Information Coefficient between signals and forward returns.

    Parameters
    ----------
    signals : np.ndarray
        Alpha signal values.
    forward_returns : np.ndarray
        Realized forward returns.

    Returns
    -------
    float
        Pearson correlation (IC).
    """
    signals = np.asarray(signals, dtype=float)
    forward_returns = np.asarray(forward_returns, dtype=float)
    mask = ~(np.isnan(signals) | np.isnan(forward_returns))
    if mask.sum() < 2:
        return 0.0
    s, r = signals[mask], forward_returns[mask]
    std_s = np.std(s, ddof=1)
    std_r = np.std(r, ddof=1)
    if std_s < 1e-12 or std_r < 1e-12:
        return 0.0
    return float(np.corrcoef(s, r)[0, 1])


def compute_ir(ic_series: np.ndarray) -> float:
    """
    Information Ratio = mean(IC) / std(IC).

    Parameters
    ----------
    ic_series : np.ndarray
        Time series of Information Coefficients.

    Returns
    -------
    float
        Information Ratio.
    """
    ic_series = np.asarray(ic_series, dtype=float)
    ic_series = ic_series[~np.isnan(ic_series)]
    if len(ic_series) < 2:
        return 0.0
    std_ic = float(np.std(ic_series, ddof=1))
    if std_ic < 1e-12:
        return 0.0
    return float(np.mean(ic_series) / std_ic)
