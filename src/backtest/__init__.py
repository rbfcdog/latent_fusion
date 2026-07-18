from .engine import BacktestConfig, BacktestEngine, BacktestResult, Strategy, Trade, decompose_alpha_beta
from src.strategy import (
    HMMRegimeStrategy,
    InstitutionalV3Strategy,
    IntensityGatedStrategy,
    MeanReversionStrategy,
    RegimeRouterStrategy,
    S1Hard70Strategy,
    SmaCrossStrategy,
    VwapReversionStrategy,
)
from .monte_carlo import MonteCarloConfig, MonteCarloResult, monte_carlo_simulate
from .optimizer import (
    GridSearchResult,
    ParamGrid,
    WalkForwardConfig,
    WalkForwardResult,
    grid_search,
    walk_forward_optimize,
)
from .stress_test import (
    STRESS_SCENARIOS,
    StressScenario,
    StressTestResult,
    run_stress_tests,
)
from .visualization import (
    plot_equity_curve,
    plot_monte_carlo_fan,
    plot_param_heatmap,
    plot_stress_test,
    plot_walk_forward,
)
