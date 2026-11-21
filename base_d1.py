"""Baselines on D1. Usage: python base_d1.py stat [hist]  |  python base_d1.py MODEL seed [hist]   (MODEL in LSTM, DeepAR, TFT)"""
import sys, time, os, numpy as np, pandas as pd, logging, warnings
warnings.filterwarnings('ignore'); logging.getLogger('pytorch_lightning').setLevel(logging.ERROR); logging.getLogger('lightning').setLevel(logging.ERROR)
os.environ['NIXTLA_ID_AS_COL'] = '1'
from d1_prep import load
H, L = 8, 16
ids, X, Y = load()
mode = sys.argv[1]
hist = int(sys.argv[3]) if mode != 'stat' and len(sys.argv) > 3 else (int(sys.argv[2]) if mode == 'stat' and len(sys.argv) > 2 else 0)
if hist: X = [x[-hist - H:] if len(x) >= hist + H else x for x in X]
tag = f'_h{hist}' if hist else ''
_lock = f'res/lock_d1{tag}_{mode}_{sys.argv[2] if mode != "stat" else ""}'
if os.path.exists(_lock): sys.exit(0)
open(_lock, 'w').close()
df = pd.concat([pd.DataFrame({'unique_id': i, 'ds': np.arange(len(x)), 'y': x}) for i, x in enumerate(X)])

def to_mat(fc, col):
    fc = fc.sort_values(['unique_id', 'ds']); return fc[col].values.reshape(len(X), H)

if mode == 'stat':
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, Naive, SeasonalNaive
    t0 = time.time()
    sf = StatsForecast(models=[AutoARIMA(season_length=4), AutoETS(season_length=4), Naive(), SeasonalNaive(4)], freq=1, n_jobs=2)
    fc = sf.forecast(df=df, h=H, level=[90])
    out = {}
    for mname in ['AutoARIMA', 'AutoETS', 'Naive', 'SeasonalNaive']:
        out[mname] = to_mat(fc, mname); out[mname + '_lo'] = to_mat(fc, mname + '-lo-90'); out[mname + '_hi'] = to_mat(fc, mname + '-hi-90')
    np.savez_compressed(f'res/d1{tag}_stat.npz', **out, time=time.time() - t0); print('stat done', time.time() - t0)
else:
    import torch; torch.set_num_threads(2)
    from neuralforecast import NeuralForecast
    from neuralforecast.models import LSTM, DeepAR, TFT
    from neuralforecast.losses.pytorch import DistributionLoss, MAE
    seed = int(sys.argv[2])
    common = dict(h=H, input_size=L, max_steps=400, val_check_steps=50, early_stop_patience_steps=5, scaler_type='robust',
                  learning_rate=1e-3, batch_size=64, windows_batch_size=256, random_seed=seed, start_padding_enabled=True,
                  enable_progress_bar=False, enable_model_summary=False, logger=False, accelerator='cpu')
    if mode == 'LSTM': m = LSTM(encoder_hidden_size=64, decoder_hidden_size=64, encoder_n_layers=2, loss=MAE(), **common)
    if mode == 'DeepAR': m = DeepAR(lstm_hidden_size=64, lstm_n_layers=2, loss=DistributionLoss('StudentT', level=[90]), **common)
    if mode == 'TFT': m = TFT(hidden_size=64, n_head=4, loss=MAE(), **common)
    t0 = time.time(); nf = NeuralForecast(models=[m], freq=1); nf.fit(df=df, val_size=H); tt = time.time() - t0
    fc = nf.predict(); col = [c for c in fc.columns if c.startswith(mode)]
    out = {'yhat': to_mat(fc, mode if mode in fc.columns else mode + '-median'), 'train_time': tt}
    if mode == 'DeepAR': out['lo'] = to_mat(fc, 'DeepAR-lo-90'); out['hi'] = to_mat(fc, 'DeepAR-hi-90')
    np.savez_compressed(f'res/d1{tag}_{mode}_{seed}.npz', **out); print(mode, seed, 'train', round(tt, 1), col)
