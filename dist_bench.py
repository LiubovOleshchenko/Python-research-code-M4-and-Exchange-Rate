"""Sequential vs worker-pool execution of the analytical pipeline (single node, process workers emulate Celery workers).
Workload A: 1,000 D1 series -> tau/m selection, PSR, TDA for all windows. Workload B: TE matrix (56 pairs x 101 estimates) on D2."""
import time, json, numpy as np, os
from concurrent.futures import ProcessPoolExecutor
os.environ['OMP_NUM_THREADS'] = '1'
from d1_prep import load, series_job
from common import transfer_entropy

def te_pair(a):
    i, j, seed = a; R = RR; rng = np.random.default_rng(seed)
    te = transfer_entropy(R[:, i], R[:, j])
    sur = [transfer_entropy(np.roll(R[:, i], rng.integers(20, len(R) - 20)), R[:, j]) for _ in range(100)]
    return i, j, te, (1 + np.sum(np.array(sur) >= te)) / 101

X0 = np.loadtxt('../data/exchange_rate.txt.gz', delimiter=',')[:5311]; RR = np.diff((X0 - X0.mean(0)) / X0.std(0), axis=0)
if __name__ == '__main__':
    _, X, _ = load(); recs = [(x,) for x in X[:1000]]
    pairs = [(i, j, 7 * i + j) for i in range(8) for j in range(8) if i != j]
    res = {}
    for name, fn, tasks in [('A', series_job, recs), ('B', te_pair, pairs)]:
        for mode in ['seq', 'w1', 'w2']:
            pts, lats = [], []
            for rep in range(5):
                t0 = time.perf_counter()
                if mode == 'seq':
                    lat = []
                    for t in tasks: s = time.perf_counter(); fn(t); lat.append(time.perf_counter() - s)
                else:
                    with ProcessPoolExecutor(int(mode[1])) as ex:
                        sub = time.perf_counter(); futs = [(ex.submit(fn, t), time.perf_counter()) for t in tasks]
                        lat = []
                        for f, ts in futs: f.result(); lat.append(time.perf_counter() - ts)
                pts.append(time.perf_counter() - t0); lats.append(np.mean(lat))
            res[f'{name}_{mode}'] = dict(PT_mean=np.mean(pts), PT_std=np.std(pts, ddof=1), LAT_mean=np.mean(lats), n=len(tasks), runs=pts)
            print(name, mode, round(np.mean(pts), 2), '+-', round(np.std(pts, ddof=1), 2), 'lat', round(np.mean(lats), 3), flush=True)
    json.dump(res, open('res/dist_bench.json', 'w'), indent=1)
