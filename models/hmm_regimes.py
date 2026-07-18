from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class HMMRegimeResult:
    """Results from HMM regime detection"""
    model: hmm.GaussianHMM
    states: np.ndarray
    logprob: float
    transition_matrix: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    weights: np.ndarray
    scaler: StandardScaler
    regime_labels: np.ndarray


def train_hmm_with_embeddings(
    text_embeddings: np.ndarray,
    returns: np.ndarray,
    volatility: Optional[np.ndarray] = None,
    n_states: int = 3,
    n_iter: int = 1000,
    random_state: int = 42,
    embedding_weight: float = 0.5
) -> HMMRegimeResult:
    """
    Train Gaussian HMM on text embeddings + market features to detect sentiment regimes.
    
    Args:
        text_embeddings: [T, D] text embeddings from kernel aggregation
        returns: [T] price returns
        volatility: [T] realized volatility (optional)
        n_states: 3 for bullish, neutral, bearish
        n_iter: Number of EM iterations
        random_state: Random seed
        embedding_weight: Weight for embeddings vs returns in feature mix
    
    Returns:
        HMMRegimeResult with trained model and sentiment regime states
    """
    T = len(returns)
    
    # Reduce embeddings with PCA to avoid high dimensionality
    if text_embeddings.shape[1] > 32:
        pca = PCA(n_components=32, random_state=random_state)
        emb_reduced = pca.fit_transform(text_embeddings)
    else:
        emb_reduced = text_embeddings.copy()
    
    # Build feature matrix: [embedding features, returns, volatility]
    features_list = [emb_reduced, returns.reshape(-1, 1)]
    if volatility is not None:
        features_list.append(volatility.reshape(-1, 1))
    
    X_combined = np.hstack(features_list)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # Train HMM
    model = hmm.GaussianHMM(n_components=n_states, n_iter=n_iter, random_state=random_state)
    model.fit(X_scaled)
    states = model.predict(X_scaled)
    logprob = model.score(X_scaled)
    
    # Map states to sentiment labels based on mean returns
    mean_returns_by_state = np.array([returns[states == i].mean() for i in range(n_states)])
    state_order = np.argsort(mean_returns_by_state)
    
    # Create mapping: bearish (negative returns) -> neutral (middle) -> bullish (positive returns)
    sentiment_map = {state_order[0]: 0, state_order[1]: 1, state_order[2]: 2}  # 0=bearish, 1=neutral, 2=bullish
    regime_labels = np.array([sentiment_map[s] for s in states])
    
    return HMMRegimeResult(
        model=model,
        states=states,
        logprob=logprob,
        transition_matrix=model.transmat_,
        means=model.means_,
        variances=model.covars_,
        weights=model.startprob_,
        scaler=scaler,
        regime_labels=regime_labels
    )


def classify_sentiment_regimes(
    text_embeddings: np.ndarray,
    returns: np.ndarray,
    volatility: Optional[np.ndarray] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Classify into sentiment regimes: Bearish, Neutral, Bullish using embeddings.
    
    Args:
        text_embeddings: [T, D] text embeddings
        returns: [T] price returns
        volatility: [T] realized volatility (optional)
        random_state: Random seed
    
    Returns:
        Tuple of (regime_labels array, DataFrame with regime statistics)
    """
    result = train_hmm_with_embeddings(
        text_embeddings, returns, volatility, n_states=3, random_state=random_state
    )
    
    regime_stats = pd.DataFrame({
        'regime_id': range(3),
        'mean_return': [returns[result.regime_labels == i].mean() for i in range(3)],
        'volatility': [returns[result.regime_labels == i].std() for i in range(3)],
        'n_days': [(result.regime_labels == i).sum() for i in range(3)],
        'transition_prob_in': [result.transition_matrix[i, i] for i in range(3)]
    })
    
    regime_stats['label'] = ['Bearish', 'Neutral', 'Bullish']
    regime_stats = regime_stats.sort_values('mean_return')
    
    return result.regime_labels, regime_stats, result


def detect_regime_transitions(regime_labels: np.ndarray) -> pd.DataFrame:
    """
    Detect sentiment regime transitions (Bearish <-> Neutral <-> Bullish).
    
    Args:
        regime_labels: Array of regime labels (0=bearish, 1=neutral, 2=bullish)
    
    Returns:
        DataFrame of transitions with timing and sentiment
    """
    transitions = []
    sentiment_names = {0: 'Bearish', 1: 'Neutral', 2: 'Bullish'}
    
    for i in range(1, len(regime_labels)):
        if regime_labels[i] != regime_labels[i-1]:
            from_regime = regime_labels[i-1]
            to_regime = regime_labels[i]
            
            transitions.append({
                'idx': i,
                'from_regime': from_regime,
                'to_regime': to_regime,
                'from_sentiment': sentiment_names[from_regime],
                'to_sentiment': sentiment_names[to_regime],
                'direction': 'bullish_turn' if to_regime > from_regime else 'bearish_turn'
            })
    
    return pd.DataFrame(transitions) if transitions else pd.DataFrame()


def regime_detection_delay(
    true_regimes: np.ndarray,
    predicted_regimes: np.ndarray
) -> float:
    """
    Calculate average regime detection delay (in periods).
    
    Args:
        true_regimes: Ground truth regime labels
        predicted_regimes: Predicted regime labels
    
    Returns:
        Average delay in periods
    """
    true_transitions = np.where(np.diff(true_regimes) != 0)[0]
    pred_transitions = np.where(np.diff(predicted_regimes) != 0)[0]
    
    if len(true_transitions) == 0 or len(pred_transitions) == 0:
        return 0.0
    
    delays = []
    for true_t in true_transitions:
        closest_pred = np.argmin(np.abs(pred_transitions - true_t))
        delay = abs(pred_transitions[closest_pred] - true_t)
        delays.append(delay)
    
    return float(np.mean(delays)) if delays else 0.0


def regime_metrics(true_regimes: np.ndarray, predicted_regimes: np.ndarray) -> dict:
    """
    Calculate regime detection metrics.
    
    Args:
        true_regimes: Ground truth labels
        predicted_regimes: Predicted labels
    
    Returns:
        Dictionary with accuracy, F1-score, confusion matrix
    """
    accuracy = accuracy_score(true_regimes, predicted_regimes)
    f1 = f1_score(true_regimes, predicted_regimes, average='weighted', zero_division=0)
    cm = confusion_matrix(true_regimes, predicted_regimes)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm,
        'detection_delay': regime_detection_delay(true_regimes, predicted_regimes)
    }


def forecast_regime(
    transition_matrix: np.ndarray,
    current_regime: int,
    steps: int = 5
) -> np.ndarray:
    """
    Forecast future sentiment regime based on transition probabilities.
    
    Args:
        transition_matrix: [3, 3] transition probability matrix
        current_regime: Current regime label (0=bearish, 1=neutral, 2=bullish)
        steps: Number of steps to forecast
    
    Returns:
        Array of forecasted regime labels
    """
    forecast = np.zeros(steps, dtype=int)
    regime = current_regime
    
    for i in range(steps):
        regime = np.random.choice(
            np.arange(len(transition_matrix)),
            p=transition_matrix[regime]
        )
        forecast[i] = regime
    
    return forecast


def sentiment_regime_metrics(returns: np.ndarray, regime_labels: np.ndarray) -> dict:
    """
    Calculate sentiment regime performance metrics.
    
    Args:
        returns: [T] price returns
        regime_labels: [T] regime labels (0=bearish, 1=neutral, 2=bullish)
    
    Returns:
        Dictionary with performance by regime
    """
    sentiment_names = {0: 'Bearish', 1: 'Neutral', 2: 'Bullish'}
    
    metrics = {}
    for regime in range(3):
        mask = regime_labels == regime
        if mask.sum() > 0:
            regime_returns = returns[mask]
            metrics[sentiment_names[regime]] = {
                'n_days': mask.sum(),
                'avg_return': regime_returns.mean(),
                'std_return': regime_returns.std(),
                'sharpe': regime_returns.mean() / (regime_returns.std() + 1e-8) * np.sqrt(252),
                'win_rate': (regime_returns > 0).mean(),
                'max_drawdown': (regime_returns.cumsum().max() - regime_returns.cumsum()).max()
            }
    
    return metrics


def integrate_regimes_with_volatility(
    price_data: pd.DataFrame,
    z_context: Optional[np.ndarray] = None,
    window: int = 20
) -> pd.DataFrame:
    """
    Integrate HMM sentiment regimes with rolling volatility.
    
    Args:
        price_data: DataFrame with 'Close' column
        z_context: [T, D] text embeddings (optional, use price-only if None)
        window: Rolling window for volatility
    
    Returns:
        DataFrame with regimes and volatility metrics
    """
    returns = price_data['Close'].pct_change().dropna()
    rolling_vol = returns.rolling(window).std()
    rolling_vol_annualized = rolling_vol * np.sqrt(252)
    
    if z_context is not None:
        regime_labels, regime_stats, _ = classify_sentiment_regimes(
            z_context[:len(returns)], returns.values, rolling_vol_annualized.values[:len(returns)]
        )
        
        sentiment_names = {0: 'Bearish', 1: 'Neutral', 2: 'Bullish'}
        sentiment_labels = np.array([sentiment_names[r] for r in regime_labels])
    else:
        regime_labels = None
        sentiment_labels = None
    
    result_df = pd.DataFrame({
        'returns': returns.values,
        'rolling_vol': rolling_vol_annualized.values[:len(returns)],
        'sentiment_regime': sentiment_labels if sentiment_labels is not None else None,
        'regime_id': regime_labels if regime_labels is not None else None
    }, index=returns.index)
    
    return result_df


def integrate_sentiment_regimes(
    merged_df: pd.DataFrame,
    z_context_dict: dict,
    lambdas: list = [0.01, 0.05, 0.1]
) -> dict:
    """
    Integrate sentiment regimes across different kernel decay parameters.
    
    Args:
        merged_df: DataFrame with date, Close, vol_20d columns
        z_context_dict: Dict of {lambda: embeddings} from kernel aggregation
        lambdas: List of lambda values to evaluate
    
    Returns:
        Dictionary with regime results for each lambda
    """
    results = {}
    returns = merged_df['Close'].pct_change().dropna().values
    volatility = merged_df['vol_20d'].dropna().values
    
    # Ensure alignment
    min_len = min(len(returns), len(volatility))
    returns = returns[:min_len]
    volatility = volatility[:min_len]
    
    for lam in lambdas:
        z_context = z_context_dict[lam]
        # Ensure alignment
        z_context = z_context[:min_len]
        
        regime_labels, regime_stats, result = classify_sentiment_regimes(
            z_context, returns, volatility
        )
        
        metrics = sentiment_regime_metrics(returns, regime_labels)
        
        results[lam] = {
            'regime_labels': regime_labels,
            'regime_stats': regime_stats,
            'metrics': metrics,
            'model': result.model,
            'transition_matrix': result.transition_matrix
        }
    
    return results
