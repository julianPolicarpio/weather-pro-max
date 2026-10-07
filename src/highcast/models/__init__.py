from lightgbm import LGBMRegressor


def make_lgbm(alpha: float) -> LGBMRegressor:
    return LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=300,
                         learning_rate=0.05, num_leaves=15,
                         min_child_samples=20, verbose=-1)
