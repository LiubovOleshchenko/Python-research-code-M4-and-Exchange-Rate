"""Aggregate results: metrics mean±sd over seeds, Wilcoxon / Diebold-Mariano tests with Holm correction, coverage."""
import numpy as np, glob, json, os
from scipy import stats
from d1_prep import load
from common import metrics
R = {}
_, X, Y = load(); n1 = len(X)
s1 = np.array([np.mean(np.abs(x)) for x in X])[:, None]                       # nMAE/nRMSE normalizer (mean |level| of training series)
mase_den = np.array([np.mean(np.abs(x[4:] - x[:-4])) if len(x) > 4 else np.mean(np.abs(np.diff(x))) for x in X])[:, None]

def m_d1(yh, Yt=Y):
    e = Yt - yh
    return dict(nMAE=np.mean(np.abs(e) / s1), nRMSE=np.mean(np.sqrt(np.mean((e / s1) ** 2, 1))),
                MAPE=100 * np.mean(np.abs(e) / np.abs(Yt)), sMAPE=200 * np.mean(np.abs(e) / (np.abs(Yt) + np.abs(yh))),
                MASE=np.mean(np.abs(e) / mase_den))
def per_series_smape(yh): return 200 * np.mean(np.abs(Y - yh) / (np.abs(Y) + np.abs(yh)), 1)

def collect(pattern, key='yhat'):
    fs = sorted(glob.glob(pattern)); return [np.load(f)[key] for f in fs], [np.load(f) for f in fs]

def summarize(preds, mfun):
    ms = [mfun(p) for p in preds]
    return {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms], ddof=1)) if len(ms) > 1 else 0.0) for k in ms[0]}, ms

def holm(ps):
    ks = list(ps); order = np.argsort([ps[k] for k in ks]); out, run = {}, 0
    for r, i in enumerate(order):
        adj = min(1.0, (len(ks) - r) * ps[ks[i]]); run = max(run, adj); out[ks[i]] = run
    return out

def dm_test(e1, e2, h):
    """Diebold-Mariano with HAC (Bartlett, h-1 lags) on absolute-error loss differential; e: (n,) per-origin mean abs error."""
    d = e1 - e2; n = len(d); dm = d - d.mean(); g0 = dm @ dm / n
    lrv = g0 + 2 * sum((1 - k / h) * (dm[k:] @ dm[:-k] / n) for k in range(1, h))
    stat = d.mean() / np.sqrt(lrv / n); return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))

# ---------------- D1 ----------------
def d1_block(tag=''):
    out = {}; per = {}
    st = np.load(f'res/d1{tag}_stat.npz') if os.path.exists(f'res/d1{tag}_stat.npz') else None
    Yt = Y
    for name in ['Naive', 'SeasonalNaive', 'AutoARIMA', 'AutoETS']:
        if st is not None:
            out[name] = summarize([st[name]], m_d1)[0]; per[name] = per_series_smape(st[name])
            out[name]['CR'] = (float(np.mean((Y >= st[name + '_lo']) & (Y <= st[name + '_hi']))), 0.0)
    for name in ['LSTM', 'DeepAR', 'TFT', 'full', 'noTDA', 'nodeonly', 'rawTDA']:
        preds, raw = collect(f'res/d1{tag}_{name}_[0-9].npz')
        if not preds: continue
        out[name], ms = summarize(preds, m_d1); out[name]['n_seeds'] = len(preds)
        out[name]['per_seed'] = ms
        per[name] = np.mean([per_series_smape(p) for p in preds], 0)
        if 'lo' in raw[0].files:
            crs = [np.mean((Y >= r['lo']) & (Y <= r['hi'])) for r in raw]
            out[name]['CR'] = (float(np.mean(crs)), float(np.std(crs, ddof=1)) if len(crs) > 1 else 0.0)
            out[name]['CR_h'] = np.mean([np.mean((Y >= r['lo']) & (Y <= r['hi']), 0) for r in raw], 0).tolist()
            out[name]['width'] = float(np.mean([np.mean((r['hi'] - r['lo']) / s1) for r in raw]))
        if 'train_time' in raw[0].files: out[name]['train_time'] = (float(np.mean([r['train_time'] for r in raw])), float(np.std([r['train_time'] for r in raw], ddof=1)))
    if 'full' in per:
        ps, med = {}, {}
        for k in per:
            if k == 'full': continue
            w = stats.wilcoxon(per['full'], per[k]); ps[k] = float(w.pvalue); med[k] = float(np.median(per['full'] - per[k]))
        ph = holm(ps); out['tests'] = {k: dict(p=ps[k], p_holm=ph[k], median_diff=med[k], wins=float(np.mean(per['full'] < per[k]))) for k in ps}
        # paired t-test over seeds (sMAPE) vs seed-matched neural baselines
        out['ttest_seeds'] = {}
        for k in ['LSTM', 'DeepAR', 'TFT', 'noTDA', 'nodeonly', 'rawTDA']:
            if k in out and 'per_seed' in out[k] and len(out[k]['per_seed']) == len(out['full']['per_seed']) > 1:
                a = [m['sMAPE'] for m in out['full']['per_seed']]; b = [m['sMAPE'] for m in out[k]['per_seed']]
                out['ttest_seeds'][k] = float(stats.ttest_rel(a, b).pvalue)
    # stratified by history length
    lens = np.array([len(x) for x in X]); strata = {'short (<40)': lens < 40, 'medium (40-99)': (lens >= 40) & (lens < 100), 'long (>=100)': lens >= 100}
    out['strata'] = {s: {k: float(np.mean(per[k][msk])) for k in per} | {'n': int(msk.sum())} for s, msk in strata.items()}
    return out
R['d1'] = d1_block(); R['d1_h20'] = d1_block('_h20')

# ---------------- D2 ----------------
def m_d2(yh, yt):
    e = yt - yh; sd = SD[None, :, None]
    raw_e = e * sd; raw_y = yt * sd + MU[None, :, None]
    return dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))), MAPE=float(100 * np.mean(np.abs(raw_e) / np.abs(raw_y))))
raw = np.loadtxt('../data/exchange_rate.txt.gz', delimiter=','); MU, SD = raw[:5311].mean(0), raw[:5311].std(0)
for H, cond in [(5, 'clean'), (20, 'clean'), (5, 'missing'), (5, 'noise')]:
    key = f'd2_H{H}_{cond}'; out = {}; per = {}
    sf = f'res/d2_H{H}_{cond}_stat_0.npz'
    if os.path.exists(sf):
        st = np.load(sf); yt = st['ytrue']
        for name in ['Naive', 'AutoARIMA']:
            out[name] = {k: (v, 0.0) for k, v in m_d2(st[name], yt).items()}; per[name] = np.abs(yt - st[name]).mean((1, 2))
            out[name]['CR'] = (float(np.mean((yt >= st[name + '_lo']) & (yt <= st[name + '_hi']))), 0.0)
    for name in ['LSTM', 'DeepAR', 'TFT', 'full', 'noTDA', 'noCG', 'noTE', 'nodeonly']:
        fs = sorted(glob.glob(f'res/d2_H{H}_{cond}_{name}_[0-9].npz'))
        if not fs: continue
        raws = [np.load(f) for f in fs]; ms = [m_d2(r['yhat'], r['ytrue']) for r in raws]
        out[name] = {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms], ddof=1)) if len(ms) > 1 else 0.0) for k in ms[0]}
        out[name]['n_seeds'] = len(ms); out[name]['per_seed'] = ms
        per[name] = np.mean([np.abs(r['ytrue'] - r['yhat']).mean((1, 2)) for r in raws], 0)
        if 'lo' in raws[0].files:
            crs = [np.mean((r['ytrue'] >= r['lo']) & (r['ytrue'] <= r['hi'])) for r in raws]
            out[name]['CR'] = (float(np.mean(crs)), float(np.std(crs, ddof=1)) if len(crs) > 1 else 0.0)
            out[name]['CR_h'] = np.mean([np.mean((r['ytrue'] >= r['lo']) & (r['ytrue'] <= r['hi']), (0, 1)) for r in raws], 0).tolist()
            out[name]['width'] = float(np.mean([np.mean(r['hi'] - r['lo']) for r in raws]))
        if 'train_time' in raws[0].files: out[name]['train_time'] = (float(np.mean([r['train_time'] for r in raws])), float(np.std([r['train_time'] for r in raws], ddof=1)))
    if 'full' in per:
        ps, dms, wp = {}, {}, {}
        for k in per:
            if k == 'full': continue
            dms[k] = dm_test(per['full'], per[k], H); ps[k] = dms[k][1]; wp[k] = float(stats.wilcoxon(per['full'], per[k]).pvalue)
        ph = holm(ps); pw = holm(wp)
        out['tests'] = {k: dict(DM=dms[k][0], p=ps[k], p_holm=ph[k], wilcoxon_p=wp[k], wilcoxon_holm=pw[k]) for k in ps}
        out['ttest_seeds'] = {}
        for k in ['LSTM', 'DeepAR', 'TFT', 'noTDA', 'noCG', 'noTE', 'nodeonly']:
            if k in out and len(out[k].get('per_seed', [])) == len(out['full']['per_seed']) > 1:
                out['ttest_seeds'][k] = float(stats.ttest_rel([m['MAE'] for m in out['full']['per_seed']], [m['MAE'] for m in out[k]['per_seed']]).pvalue)
    R[key] = out
json.dump(R, open('res/summary.json', 'w'), indent=1, default=float)

def fmt(v, d=3): return f'{v[0]:.{d}f} ± {v[1]:.{d}f}' if isinstance(v, (tuple, list)) else str(v)
for blk in R:
    print('=====', blk)
    for k, v in R[blk].items():
        if k in ('tests', 'strata', 'ttest_seeds'): print(k, json.dumps(v, default=float)[:1500]); continue
        print(f'{k:14s}', '  '.join(f'{m}={fmt(v[m])}' for m in v if m not in ('per_seed', 'CR_h', 'n_seeds', 'width', 'train_time')), ' w=%.3f' % v['width'] if 'width' in v else '')
