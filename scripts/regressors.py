"""The three manuscript regressors, preserving the original hyperparameters."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def build_model(method, device='cpu', seed=42):
    if device == 'gpu':
        import cupy
        if cupy.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('GPU requested, but no CUDA device is available')
    if method == 'SVR':
        if device == 'gpu':
            from cuml.svm import SVR as GPU_SVR
            estimator, backend = GPU_SVR, 'cuml.SVR'
        else:
            estimator, backend = SVR, 'sklearn.SVR'
        model = estimator(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale', cache_size=1024)
        return make_pipeline(StandardScaler(), model), backend
    if method == 'RF':
        settings = dict(n_estimators=100, max_features=0.3, random_state=seed)
        if device == 'gpu':
            from cuml.ensemble import RandomForestRegressor as GPU_RF
            return GPU_RF(**settings, n_streams=4), 'cuml.RandomForestRegressor'
        return RandomForestRegressor(**settings, n_jobs=-1), 'sklearn.RandomForestRegressor'
    if method == 'XGB':
        from xgboost import XGBRegressor
        settings = dict(num_parallel_tree=4, n_estimators=1000, learning_rate=0.25,
                        reg_alpha=0.01, reg_lambda=0.1, tree_method='hist', subsample=0.85,
                        colsample_bytree=0.3, colsample_bylevel=0.5, random_state=seed)
        if device == 'gpu':
            settings['device'] = 'cuda'
        else:
            settings['n_jobs'] = -1
        return XGBRegressor(**settings), f'xgboost.{device}'
    raise ValueError(method)


def metrics(true, pred):
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    return dict(MAE=float(np.mean(np.abs(true - pred))),
                R2=float(r2_score(true, pred)) if len(true) > 1 else np.nan)
