"""Baselines on D2 (rolling-origin, step 1, no refit). Usage: python base_d2.py MODEL seed H cond  (MODEL in stat, LSTM, DeepAR, TFT)"""
import sys, os, time, numpy as np, pandas as pd, warnings, logging
warnings.filterwarnings('ignore'); logging.getLogger('pytorch_lightning').setLevel(logging.ERROR); logging.getLogger('lightning').setLevel(logging.ERROR)
os.environ['NIXTLA_ID_AS_COL'] = '1'
mode, seed, H, COND = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
L, NT, NV, NTE = 32, 5311, 759, 1518
raw = np.loadtxt('../data/exchange_rate.txt.gz', delimiter=','); clean = raw.copy()
rng = np.random.default_rng(123)
if COND == 'missing':
    X = raw.copy()
    for j in range(X.shape[1]):
        miss = rng.random(len(X)) < 0.10; miss[[0, -1]] = False
        X[miss, j] = np.interp(np.where(miss)[0], np.where(~miss)[0], X[~miss, j])
    raw = X
elif COND == 'noise':
    raw = raw + rng.normal(0, 1, raw.shape) * 0.5 * np.diff(raw[:NT], axis=0).std(0)
mu, sd = raw[:NT].mean(0), raw[:NT].std(0); X = (raw - mu) / sd; Xt = (clean - mu) / sd
df = pd.concat([pd.DataFrame({'unique_id': j, 'ds': np.arange(len(X)), 'y': X[:, j]}) for j in range(8)])
origins = np.arange(NT + NV, len(X) - H + 1)

def to_arr(cv, col):
    cv = cv.sort_values(['cutoff', 'unique_id', 'ds'])
    return cv[col].values.reshape(len(origins), 8, H)

t0 = time.time(); out = {}
if mode == 'stat':
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, Naive
    sf = StatsForecast(models=[AutoARIMA(season_length=1), Naive()], freq=1, n_jobs=2)
    cv = sf.cross_validation(df=df, h=H, test_size=len(X) - (NT + NV), step_size=1, refit=False, level=[90])
    for c in ['AutoARIMA', 'Naive']:
        out[c] = to_arr(cv, c); out[c + '_lo'] = to_arr(cv, c + '-lo-90'); out[c + '_hi'] = to_arr(cv, c + '-hi-90')
else:
    import torch; torch.set_num_threads(2)
    from neuralforecast import NeuralForecast
    from neuralforecast.models import LSTM, DeepAR, TFT
    from neuralforecast.losses.pytorch import DistributionLoss, MAE
    common = dict(h=H, input_size=L, max_steps=300, val_check_steps=50, early_stop_patience_steps=4, scaler_type='standard',
                  learning_rate=1e-3, batch_size=8, windows_batch_size=256, random_seed=seed, enable_progress_bar=False,
                  enable_model_summary=False, logger=False, accelerator='cpu')
    if mode == 'LSTM': m = LSTM(encoder_hidden_size=64, decoder_hidden_size=64, encoder_n_layers=2, loss=MAE(), **common)
    if mode == 'DeepAR': m = DeepAR(lstm_hidden_size=64, lstm_n_layers=2, loss=DistributionLoss('StudentT', level=[90]), **common)
    if mode == 'TFT': m = TFT(hidden_size=64, n_head=4, loss=MAE(), **common)
    nf = NeuralForecast(models=[m], freq=1)
    cv = nf.cross_validation(df=df, val_size=NV, test_size=len(X) - (NT + NV), n_windows=None, step_size=1, refit=False)
    out['yhat'] = to_arr(cv, mode if mode in cv.columns else mode + '-median')
    if mode == 'DeepAR': out['lo'] = to_arr(cv, 'DeepAR-lo-90'); out['hi'] = to_arr(cv, 'DeepAR-hi-90')
ytrue = np.stack([Xt[t:t + H].T for t in origins])
np.savez_compressed(f'res/d2_H{H}_{COND}_{mode}_{seed}.npz', **out, ytrue=ytrue, time=time.time() - t0)
k = 'yhat' if 'yhat' in out else 'AutoARIMA'
print(mode, seed, H, COND, 'time %.1f' % (time.time() - t0), 'MAE %.4f' % np.abs(out[k] - ytrue).mean())
