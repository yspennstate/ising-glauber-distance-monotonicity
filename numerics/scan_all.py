"""Double-precision scans of the path example: discrete, lazy and continuous time."""
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.special import expit


def chain(n, edges):
    spins = np.array(list(itertools.product((-1,1), repeat=n)))
    indices = {tuple(s): k for k,s in enumerate(spins)}
    p = np.zeros((2**n,2**n))
    for a,s in enumerate(spins):
        for site in range(n):
            field = sum(j*s[v] if u == site else j*s[u] if v == site else 0
                        for u,v,j in edges)
            plus = expit(2*field)
            for spin, probability in [(-1,1-plus),(1,plus)]:
                target = s.copy()
                target[site] = spin
                p[a,indices[tuple(target)]] += probability/n
    return p


def diameter(a):
    # Only a 16x16 or 32x32 temporary, no time-by-state-by-state array.
    return max(np.abs(a-a[k]).sum(axis=1).max()/2 for k in range(len(a)))


def scan(lazy=False):
    ps = [chain(4,[(0,1,2),(1,2,x),(2,3,4)]) for x in [1,2]]
    if lazy:
        ps = [(p+np.eye(16))/2 for p in ps]
    kernels = [np.eye(16),np.eye(16)]
    decreases, increases, ties = [], [], []
    samples = {}
    minimum = (1.,None)
    for t in range(1,3001):
        # Avoid the platform BLAS path; einsum performs these tiny contractions
        # directly and provides an additional check on the original matmul run.
        kernels = [np.einsum('ij,jk->ik',a,p,optimize=False) for a,p in zip(kernels,ps)]
        assert all(np.isfinite(a).all() for a in kernels)
        values = [diameter(a) for a in kernels]
        delta = values[1]-values[0]
        if delta < -1e-12:
            decreases.append(t)
            if abs(delta)<minimum[0]:
                minimum = (abs(delta),t)
        elif delta > 1e-12:
            increases.append(t)
        else:
            ties.append(t)
        if t in [1,2,3,4,5,6,7,8,12,13,14,15,30,144,699,700,701,1399,1400,3000]:
            samples[t] = {'dbar':values,'delta':delta}
    return {'decreasing_times':decreases,'increasing_times':increases,
            'numerical_ties_1e-12':ties,'smallest_decrease':minimum,'samples':samples,
            'max_stochastic_error': max(float(abs(a.sum(axis=1)-1).max()) for a in kernels)}


if __name__ == '__main__':
    report = {'ordinary': scan(), 'lazy': scan(True)}
    ps = [chain(4,[(0,1,2),(1,2,x),(2,3,4)]) for x in [1,2]]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        report['continuous_direct_expm'] = {
            str(t): [diameter(expm(4*t*(p-np.eye(16)))) for p in ps]
            for t in [5,10,20,30,40,60]}
    report['scipy_platform_warnings'] = sorted(set(str(w.message) for w in caught))
    assert all(np.isfinite(value) for pair in report['continuous_direct_expm'].values()
               for value in pair)
    Path('scan_all.json').write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['ordinary','lazy']:
        item=report[kind]
        ts=item['decreasing_times']
        print(kind,'decreases',ts[0],ts[-1], 'count',len(ts),
              'contiguous',ts==list(range(ts[0],ts[-1]+1)),
              'ties',item['numerical_ties_1e-12'],
              'smallest_decrease',item['smallest_decrease'],
              'max_stochastic_error',item['max_stochastic_error'])
    print('continuous_direct_expm',report['continuous_direct_expm'])
