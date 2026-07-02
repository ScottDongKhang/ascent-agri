"""ascentagri/regime/model.py — ported near-verbatim from Ascent Capital.

K==2  -> MarkovRegression primary, HMM fallback
K>=3  -> GaussianHMM primary (multi-restart), MarkovRegression fallback

Changes from the source: debug print() calls converted to logging, and the
"spy_returns" argument of walk_forward_model_select renamed to
"benchmark_returns" (here: coffee daily returns).
"""
from __future__ import annotations
import logging, pickle, warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .types import RegimeLabel, RegimeScorecard

log = logging.getLogger(__name__)

try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False
    log.warning("regime.model: statsmodels not available")

try:
    from hmmlearn.hmm import GaussianHMM
    _HAS_HMMLEARN = True
except ImportError:
    _HAS_HMMLEARN = False
    log.warning("regime.model: hmmlearn not available")

MIN_TRAIN_DAYS = 252
DEGENERATE_STATE_THRESHOLD = 0.03
MAX_TRANSITION_SELF = 0.995


class _MarkovModel:
    def __init__(self, k_regimes: int, switching_variance: bool = True):
        self.k_regimes = k_regimes
        self.switching_variance = switching_variance
        self._result = None
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, signal: np.ndarray) -> bool:
        if len(signal) < MIN_TRAIN_DAYS or not _HAS_STATSMODELS:
            return False
        x = self._scaler.fit_transform(signal.reshape(-1, 1)).ravel()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = MarkovRegression(x, k_regimes=self.k_regimes,
                                       switching_variance=self.switching_variance,
                                       switching_exog=False)
                self._result = mod.fit(disp=False, maxiter=1000, em_iter=100,
                                       search_reps=15, search_iter=50)
            if not self._sanity_check():
                self._result = None
                return False
            self._fitted = True
            fracs = np.array(self._result.smoothed_marginal_probabilities).mean(axis=0)
            log.info(f"regime.model[Markov] K={self.k_regimes}: AIC={self.aic:.1f} fracs={np.round(fracs,3)}")
            return True
        except Exception as exc:
            log.warning(f"regime.model[Markov] K={self.k_regimes} failed: {exc}")
            return False

    def _sanity_check(self) -> bool:
        if self._result is None:
            return False
        fracs = np.array(self._result.smoothed_marginal_probabilities).mean(axis=0)
        if np.any(fracs < DEGENERATE_STATE_THRESHOLD):
            log.warning(f"regime.model[Markov]: degenerate fracs {np.round(fracs,3)}")
            return False
        tm = None
        for attr in ("transition", "regime_transition", "transition_matrix"):
            if hasattr(self._result, attr):
                tm = getattr(self._result, attr)
                break
        if tm is not None:
            try:
                tm = np.atleast_2d(tm).reshape(self.k_regimes, self.k_regimes)
                if np.any(np.diag(tm) > MAX_TRANSITION_SELF):
                    return False
            except Exception:
                pass
        return True

    def filtered_probs(self, signal: np.ndarray) -> np.ndarray:
        """Causal filtered probabilities for `signal` using the train-fitted
        parameters. (The source returned the TRAIN filtered probabilities
        regardless of input — latent bug, exposed here because the walk-forward
        runner scores train+test spans with a train-fitted model.)"""
        if not self._fitted or self._result is None:
            return np.full((len(signal), self.k_regimes), 1.0 / self.k_regimes)
        x = self._scaler.transform(np.asarray(signal, dtype=float).reshape(-1, 1)).ravel()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = MarkovRegression(x, k_regimes=self.k_regimes,
                                       switching_variance=self.switching_variance,
                                       switching_exog=False)
                res = mod.filter(self._result.params)
            return np.array(res.filtered_marginal_probabilities)
        except Exception as exc:
            log.warning(f"regime.model[Markov] filter on new data failed: {exc}")
            return np.full((len(signal), self.k_regimes), 1.0 / self.k_regimes)

    def smoothed_probs(self) -> np.ndarray:
        if not self._fitted or self._result is None:
            return np.array([])
        return np.array(self._result.smoothed_marginal_probabilities)

    @property
    def aic(self): return self._result.aic if self._result else np.inf
    @property
    def bic(self): return self._result.bic if self._result else np.inf
    @property
    def loglike(self): return self._result.llf if self._result else -np.inf

    def transition_matrix(self) -> Optional[np.ndarray]:
        if self._result is None:
            return None
        for attr in ("transition", "regime_transition", "transition_matrix"):
            if hasattr(self._result, attr):
                try:
                    return np.atleast_2d(getattr(self._result, attr)).reshape(self.k_regimes, self.k_regimes)
                except Exception:
                    pass
        return None

    def state_means(self) -> Optional[np.ndarray]:
        if self._result is None:
            return None
        try:
            k = self.k_regimes
            n_trans = k * (k - 1)
            params = np.asarray(self._result.params)
            return params[n_trans: n_trans + k].astype(float)
        except Exception:
            return None

    def state_vols(self) -> Optional[np.ndarray]:
        if self._result is None:
            return None
        try:
            k = self.k_regimes
            n_trans = k * (k - 1)
            params = np.asarray(self._result.params)
            if self.switching_variance:
                vols = params[n_trans + k: n_trans + k + k]
                return np.sqrt(np.abs(vols)).astype(float)
            else:
                v = params[n_trans + k]
                return np.full(k, float(np.sqrt(abs(v))))
        except Exception:
            return None


class _HMMModel:
    """GaussianHMM primary for K>=3. Multi-restart, diag covariance."""
    def __init__(self, k_regimes: int, n_iter: int = 300, n_restarts: int = 10):
        self.k_regimes = k_regimes
        self.n_iter = n_iter
        self.n_restarts = n_restarts
        self._model = None
        self._scaler = StandardScaler()
        self._imputer_medians = None
        self._fitted = False
        self._train_len = 0
        # store the actual fitted training matrix so aic/loglike
        # score on real data instead of a zero array
        self._train_matrix: Optional[np.ndarray] = None

    def _impute(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        # only fit imputer medians on training data (fit=True);
        # when scoring test/prediction data use the already-fitted medians
        x = np.where(np.isfinite(x), x, np.nan)   # ±inf → NaN → imputed
        if fit or self._imputer_medians is None:
            self._imputer_medians = np.nanmedian(x, axis=0)
        inds = np.where(np.isnan(x))
        if inds[0].size > 0:
            x = x.copy()
            x[inds] = np.take(self._imputer_medians, inds[1])
        return x

    def fit(self, features: np.ndarray) -> bool:
        if not _HAS_HMMLEARN or len(features) < MIN_TRAIN_DAYS:
            return False
        x = self._scaler.fit_transform(features)
        x = self._impute(x, fit=True)   # fit imputer on training data only
        best_model, best_score = None, -np.inf
        for seed in range(self.n_restarts):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = GaussianHMM(n_components=self.k_regimes, covariance_type="diag",
                                    n_iter=self.n_iter, random_state=seed * 7,
                                    tol=1e-5, init_params="stmc")
                    m.fit(x)
                sc = m.score(x)
                if sc > best_score:
                    best_score, best_model = sc, m
            except Exception:
                continue
        if best_model is None:
            log.warning(f"regime.model[HMM] K={self.k_regimes}: all restarts failed")
            return False
        _, state_seq = best_model.decode(x, algorithm="viterbi")
        fracs = np.bincount(state_seq, minlength=self.k_regimes) / len(state_seq)
        log.info(f"regime.model[HMM] K={self.k_regimes}: loglike={best_score:.1f} fracs={np.round(fracs,3)}")
        n_degen = int(np.sum(fracs < DEGENERATE_STATE_THRESHOLD))
        if n_degen > 1 or (n_degen == 1 and self.k_regimes == 2):
            log.warning(f"regime.model[HMM] K={self.k_regimes}: {n_degen} degenerate states, rejecting")
            return False
        if n_degen == 1:
            log.info("regime.model[HMM]: one low-occupancy state, accepting with warning")
        self._model = best_model
        self._train_len = len(x)
        self._train_matrix = x
        self._fitted = True
        return True

    def filtered_probs(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted or self._model is None:
            return np.full((len(features), self.k_regimes), 1.0 / self.k_regimes)
        x = self._scaler.transform(features)
        x = self._impute(x, fit=False)  # use train-fitted imputer
        try:
            _, posteriors = self._model.score_samples(x)
            return posteriors
        except Exception as exc:
            log.warning(f"regime.model[HMM] score_samples failed: {exc}")
            return np.full((len(features), self.k_regimes), 1.0 / self.k_regimes)

    def state_means_pc1(self) -> Optional[np.ndarray]:
        return self._model.means_[:, 0] if self._fitted and self._model else None

    def state_vols_pc1(self) -> Optional[np.ndarray]:
        if not self._fitted or self._model is None:
            return None
        try:
            c = self._model.covars_
            if c.ndim == 3:
                return np.sqrt(np.array([c[i, 0, 0] for i in range(self.k_regimes)]))
            elif c.ndim == 2:
                return np.sqrt(c[:, 0])
            else:
                return np.ones(self.k_regimes)
        except Exception:
            return np.ones(self.k_regimes)

    @property
    def aic(self):
        if not self._fitted or self._model is None or self._train_matrix is None:
            return np.inf
        n_params = (self.k_regimes ** 2 +
                    self.k_regimes * self._model.means_.shape[1] * 2)
        ll = self._model.score(self._train_matrix) * self._train_len
        return 2 * n_params - 2 * ll

    @property
    def bic(self): return np.inf

    @property
    def loglike(self):
        if not self._fitted or self._model is None or self._train_matrix is None:
            return -np.inf
        return float(self._model.score(self._train_matrix) * self._train_len)


class _PCATransformer:
    """Fit PCA and imputer on training data only, then reuse on test data —
    fitting them independently per split is leakage (test PCA would see
    test-period variance)."""
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self._pca = None
        self._imputer_medians = None
        self._fitted = False

    def fit_transform(self, features: pd.DataFrame) -> np.ndarray:
        from sklearn.decomposition import PCA
        x = features.values.astype(float)
        x = np.where(np.isfinite(x), x, np.nan)   # ±inf → NaN → imputed
        self._imputer_medians = np.nanmedian(x, axis=0)
        inds = np.where(np.isnan(x))
        if inds[0].size > 0:
            x = x.copy()
            x[inds] = np.take(self._imputer_medians, inds[1])
        if x.shape[1] == 0:
            raise ValueError("regime.model: empty feature panel")
        n_comp = min(self.n_components, x.shape[1], x.shape[0] - 1)
        self._pca = PCA(n_components=n_comp, random_state=42)
        result = self._pca.fit_transform(x)
        self._fitted = True
        return result

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Apply train-fitted PCA to test/prediction features."""
        if not self._fitted or self._pca is None:
            raise RuntimeError("_PCATransformer.fit_transform() must be called first")
        x = features.values.astype(float)
        x = np.where(np.isfinite(x), x, np.nan)   # ±inf → NaN → imputed
        inds = np.where(np.isnan(x))
        if inds[0].size > 0:
            x = x.copy()
            x[inds] = np.take(self._imputer_medians, inds[1])
        if x.shape[1] != self._pca.n_features_in_:
            raise ValueError(
                f"Feature count mismatch: train had {self._pca.n_features_in_}, "
                f"test has {x.shape[1]}"
            )
        return self._pca.transform(x)


def _feature_to_signal(features: pd.DataFrame, n_components: int = 3) -> np.ndarray:
    """Standalone helper — fits PCA fresh. Only use for training data."""
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median")
    x = imp.fit_transform(features.values)
    if x.shape[1] == 0:
        raise ValueError("regime.model: empty feature panel")
    n_comp = min(n_components, x.shape[1], x.shape[0] - 1)
    return PCA(n_components=n_comp, random_state=42).fit_transform(x)


def _evaluate_candidate(k, train_feat, test_feat, test_ret):
    """Fit on train only; PCA/imputer/scaler reused on test (no leakage)."""
    try:
        if k == 2 and _HAS_STATSMODELS:
            pca = _PCATransformer(n_components=1)
            sig_tr = pca.fit_transform(train_feat).ravel()
            sig_te = pca.transform(test_feat).ravel()
            m = _MarkovModel(k_regimes=k)
            if not m.fit(sig_tr):
                return None
            probs_test = m.filtered_probs(sig_te)
            dominant = np.argmax(probs_test, axis=1)
            aic_val, loglike_val = m.aic, m.loglike
            tm = m.transition_matrix()
            ts_val = float(np.std(tm)) if tm is not None else 0.0
        else:
            if not _HAS_HMMLEARN:
                return None
            hmm = _HMMModel(k_regimes=k, n_restarts=5)
            if not hmm.fit(train_feat.values):
                return None
            probs_test = hmm.filtered_probs(test_feat.values)
            dominant = np.argmax(probs_test, axis=1)
            aic_val, loglike_val, ts_val = hmm.aic, hmm.loglike, 0.0
        runs, cur = [], 1
        for i in range(1, len(dominant)):
            if dominant[i] == dominant[i - 1]:
                cur += 1
            else:
                runs.append(cur); cur = 1
        runs.append(cur)
        persistence = float(np.mean(runs)) if runs else 1.0
        n_switches = int(np.sum(np.diff(dominant) != 0))
        false_switch_rate = n_switches / max(1, len(dominant) / 21)
        state_fracs = np.array([np.mean(dominant == s) for s in range(k)])
        tiny_state_penalty = float(np.sum(state_fracs < DEGENERATE_STATE_THRESHOLD) > 0)
        regime_proxy = {s: (1.0 if state_fracs[s] == state_fracs.max() else 0.5) for s in range(k)}
        risk_mult = np.array([regime_proxy[d] for d in dominant])
        scaled_ret = test_ret.values[:len(risk_mult)] * risk_mult
        oos_sharpe = float(np.mean(scaled_ret) / (np.std(scaled_ret) + 1e-9) * np.sqrt(252))
        raw_sharpe = float(np.mean(test_ret.values) / (np.std(test_ret.values) + 1e-9) * np.sqrt(252))
        return {"aic": aic_val, "bic": np.inf, "loglike": loglike_val,
                "persistence": persistence, "false_switch_rate": false_switch_rate,
                "tiny_state_penalty": tiny_state_penalty,
                "oos_sharpe_impact": oos_sharpe - raw_sharpe,
                "transition_stability": ts_val}
    except Exception as exc:
        log.debug(f"regime.model: K={k} eval failed: {exc}")
        return None


def walk_forward_model_select(feature_panel, benchmark_returns, candidate_k=(2, 3, 4),
                              train_days=504, test_days=126, step_days=63, min_train=252):
    log.info(f"regime.model: walk-forward selection candidates={list(candidate_k)}")
    n = len(feature_panel)
    scorecards = {k: [] for k in candidate_k}
    train_start, n_windows = 0, 0
    while True:
        train_end = train_start + train_days
        test_end = train_end + test_days
        if test_end > n:
            break
        train_feat = feature_panel.iloc[train_start:train_end]
        test_feat = feature_panel.iloc[train_end:test_end]
        test_ret = benchmark_returns.iloc[train_end:test_end]
        for k in candidate_k:
            score = _evaluate_candidate(k, train_feat, test_feat, test_ret)
            if score is not None:
                scorecards[k].append(score)
        n_windows += 1
        train_start += step_days
        if train_start + train_days + test_days > n:
            break
    log.info(f"regime.model: {n_windows} windows, wins: { {k: len(scorecards[k]) for k in candidate_k} }")
    best_k = _select_best_k(scorecards, candidate_k)
    summary = _make_scorecard(scorecards[best_k], best_k, train_days)
    log.info(f"regime.model: selected K={best_k} composite={summary.composite_score:.4f}")
    return best_k, summary


def _select_best_k(scorecards, candidate_k):
    valid = {}
    for k in candidate_k:
        records = scorecards.get(k, [])
        if records:
            valid[k] = len(records)
    log.info("regime selection valid candidates: %s", valid)
    if not valid:
        log.warning("regime selection: no valid candidates, defaulting K=2")
        return 2
    best, best_score = None, -np.inf
    for k in valid:
        sc = _make_scorecard(scorecards[k], k, 252)
        log.info(f"regime selection K={k} composite={sc.composite_score:.4f}")
        if sc.composite_score > best_score:
            best_score, best = sc.composite_score, k
    log.info("regime selection: chose K=%s score=%.4f", best, best_score)
    return best


def _make_scorecard(records, k, lookback):
    if not records:
        return RegimeScorecard(k, lookback, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    def _m(key): return float(np.mean([r[key] for r in records]))
    return RegimeScorecard(
        n_regimes=k, lookback_days=lookback,
        aic_mean=_m("aic"), bic_mean=_m("bic"), loglike_mean=_m("loglike"),
        regime_persistence_mean=_m("persistence"),
        transition_stability=_m("transition_stability"),
        oos_sharpe_impact=_m("oos_sharpe_impact"),
        oos_drawdown_impact=0.0,
        false_switch_rate=_m("false_switch_rate"),
        tiny_state_penalty=_m("tiny_state_penalty"),
    )


def label_states(state_means, state_vols, k) -> Dict[int, RegimeLabel]:
    if state_means is None:
        return {i: RegimeLabel.UNCERTAIN for i in range(k)}
    vols = state_vols if state_vols is not None else np.ones(k)
    labels = {}
    if k == 2:
        order = np.argsort(state_means)[::-1]
        labels[order[0]] = RegimeLabel.CALM_BULL
        labels[order[1]] = RegimeLabel.CRISIS
        return labels
    if k == 3:
        order = np.argsort(state_means)[::-1]
        top_idx, mid_idx, bot_idx = order[0], order[1], order[2]
        median_vol = float(np.median(vols))
        if vols[top_idx] > median_vol * 1.2:
            labels[top_idx] = RegimeLabel.EUPHORIC
        else:
            labels[top_idx] = RegimeLabel.CALM_BULL
        used = set(labels.values())
        labels[mid_idx] = RegimeLabel.CALM_BULL if RegimeLabel.CALM_BULL not in used else RegimeLabel.STRESSED
        labels[bot_idx] = RegimeLabel.CRISIS
        for i in range(k):
            if i not in labels:
                labels[i] = RegimeLabel.UNCERTAIN
        return labels
    # k==4
    order = np.argsort(state_means)[::-1]
    top2, bot2 = order[:2], order[2:]
    top_vol_order = top2[np.argsort(vols[top2])]
    labels[top_vol_order[0]] = RegimeLabel.CALM_BULL
    labels[top_vol_order[1]] = RegimeLabel.EUPHORIC
    bot_vol_order = bot2[np.argsort(vols[bot2])]
    labels[bot_vol_order[0]] = RegimeLabel.STRESSED
    labels[bot_vol_order[1]] = RegimeLabel.CRISIS
    return labels


class RegimeModel:
    def __init__(self, k_regimes: int = 3, hmm_restarts: int = 10):
        self.k_regimes = k_regimes
        self.hmm_restarts = hmm_restarts
        self._markov = None
        self._hmm = None
        self._active_backend = "none"
        self._state_labels: Dict[int, RegimeLabel] = {}
        self._train_index = None
        self._fitted = False
        self._pca_transformer: Optional[_PCATransformer] = None

    def fit(self, feature_panel: pd.DataFrame) -> bool:
        if len(feature_panel) < MIN_TRAIN_DAYS:
            log.warning(f"regime.model: insufficient data ({len(feature_panel)} rows)")
            return False
        self._train_index = feature_panel.index
        log.info(f"regime.model: fitting K={self.k_regimes} rows={len(feature_panel)} features={feature_panel.shape[1]}")
        if self.k_regimes == 2 and _HAS_STATSMODELS:
            return self._fit_markov_primary(feature_panel)
        return self._fit_hmm_primary(feature_panel)

    def _fit_markov_primary(self, feature_panel):
        try:
            self._pca_transformer = _PCATransformer(n_components=1)
            signal = self._pca_transformer.fit_transform(feature_panel).ravel()
            m = _MarkovModel(k_regimes=self.k_regimes)
            if m.fit(signal):
                self._markov = m
                self._active_backend = "markov"
                self._state_labels = label_states(m.state_means(), m.state_vols(), self.k_regimes)
                self._fitted = True
                self._log_summary()
                return True
        except Exception as exc:
            log.warning(f"regime.model: Markov primary failed: {exc}")
        return self._fit_hmm_primary(feature_panel)

    def _fit_hmm_primary(self, feature_panel):
        try:
            hmm = _HMMModel(k_regimes=self.k_regimes, n_restarts=self.hmm_restarts)
            if hmm.fit(feature_panel.values):
                self._hmm = hmm
                self._active_backend = "hmm"
                self._state_labels = label_states(hmm.state_means_pc1(), hmm.state_vols_pc1(), self.k_regimes)
                self._fitted = True
                self._log_summary()
                return True
        except Exception as exc:
            log.warning(f"regime.model: HMM primary failed: {exc}")
        if _HAS_STATSMODELS:
            try:
                self._pca_transformer = _PCATransformer(n_components=1)
                signal = self._pca_transformer.fit_transform(feature_panel).ravel()
                m = _MarkovModel(k_regimes=self.k_regimes)
                if m.fit(signal):
                    self._markov = m
                    self._active_backend = "markov"
                    self._state_labels = label_states(m.state_means(), m.state_vols(), self.k_regimes)
                    self._fitted = True
                    self._log_summary()
                    return True
            except Exception as exc:
                log.warning(f"regime.model: Markov fallback failed: {exc}")
        log.error("regime.model: ALL fit paths failed")
        return False

    def _log_summary(self):
        log.info(f"REGIME FIT SUMMARY backend={self._active_backend} K={self.k_regimes} labels={self._state_labels}")
        if self._active_backend == "markov" and self._markov:
            tm = self._markov.transition_matrix()
            if tm is not None:
                log.info(f"  transition matrix:\n{np.round(tm, 3)}")
        elif self._active_backend == "hmm" and self._hmm and self._hmm._model:
            log.info(f"  transition matrix:\n{np.round(self._hmm._model.transmat_, 3)}")

    def predict_probs(self, feature_panel: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            return pd.DataFrame(
                np.full((len(feature_panel), self.k_regimes), 1.0 / self.k_regimes),
                index=feature_panel.index, columns=list(range(self.k_regimes)))
        if self._active_backend == "markov":
            # use stored PCA transformer to avoid re-fitting on prediction data
            if self._pca_transformer is not None:
                signal = self._pca_transformer.transform(feature_panel).ravel()
            else:
                signal = _feature_to_signal(feature_panel, 1).ravel()
            probs = self._markov.filtered_probs(signal)
        else:
            probs = self._hmm.filtered_probs(feature_panel.values)
        df = pd.DataFrame(probs, index=feature_panel.index, columns=list(range(self.k_regimes)))
        dominant = df.values.argmax(axis=1)
        occ = {}
        for i in range(self.k_regimes):
            lbl = self._state_labels.get(i, RegimeLabel.UNCERTAIN).value
            occ[lbl] = occ.get(lbl, 0) + int(np.sum(dominant == i))
        log.info(f"regime.model: predict_probs label occupancy: {occ}")
        return df

    @property
    def state_labels(self): return self._state_labels
    @property
    def active_backend(self): return self._active_backend

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"regime.model: saved to {path}")

    @classmethod
    def load(cls, path: Path):
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)
        log.info(f"regime.model: loaded from {path}")
        return obj
