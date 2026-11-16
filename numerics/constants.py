"""Spectral data for the path with couplings (2, x, 4), discrete-time heat-bath dynamics, x = 1 and x = 2:
lambda_2, the next eigenvalue, the constant C_x of the leading term, the crossing time of the leading terms,
and the pair attaining the maximum in dbar_x(t) at the times of Table 1. Multiple precision (mpmath, 60 digits);
these values are not enclosed.

    python constants.py
"""
import itertools

import mpmath as mp

mp.mp.dps = 60
N = 4
EDGES = [(0, 1), (1, 2), (2, 3)]
STATES = list(itertools.product((-1, 1), repeat=N))
IX = {s: i for i, s in enumerate(STATES)}


def kernel(x):
    J = {(0, 1): mp.mpf(2), (1, 2): mp.mpf(x), (2, 3): mp.mpf(4)}
    n = len(STATES)
    P = mp.zeros(n, n)
    for a, s in enumerate(STATES):
        for v in range(N):
            h = sum(J[e] * s[e[0] if e[1] == v else e[1]] for e in EDGES if v in e)
            up = 1 / (1 + mp.exp(-2 * h))
            for spin, prob in ((1, up), (-1, 1 - up)):
                t = list(s)
                t[v] = spin
                P[a, IX[tuple(t)]] += prob / N
    weight = [mp.exp(sum(J[e] * s[e[0]] * s[e[1]] for e in EDGES)) for s in STATES]
    Z = sum(weight)
    return P, [w / Z for w in weight]


def spectral(x):
    P, mu = kernel(x)
    n = len(STATES)
    r = [mp.sqrt(m) for m in mu]
    A = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            A[i, j] = r[i] * P[i, j] / r[j]          # symmetric, same spectrum as P
    A = (A + A.T) / 2
    E, Q = mp.eigsy(A)
    order = sorted(range(n), key=lambda k: -E[k])
    lam = [E[k] for k in order]
    v = [Q[i, order[1]] for i in range(n)]
    Pi = [[v[i] * v[j] * r[j] / r[i] for j in range(n)] for i in range(n)]   # projector onto the lambda_2 eigenspace
    C = max(sum(abs(Pi[a][k] - Pi[b][k]) for k in range(n)) / 2 for a in range(n) for b in range(a + 1, n))
    return P, lam, C


def argmax_pair(P, t):
    n = len(STATES)
    M = P ** t
    best = max(((sum(abs(M[a, k] - M[b, k]) for k in range(n)) / 2, a, b)
                for a in range(n) for b in range(a + 1, n)), key=lambda z: z[0])
    return STATES[best[1]], STATES[best[2]]


if __name__ == '__main__':
    out = {}
    for x in (1, 2):
        P, lam, C = spectral(x)
        assert lam[1] - lam[2] > mp.mpf('1e-6'), 'lambda_2 is not simple'
        out[x] = (lam, C)
        print('x =', x)
        print('  lambda_2 =', mp.nstr(lam[1], 20), ' next =', mp.nstr(lam[2], 5))
        print('  C        =', mp.nstr(C, 20))
        for t in (6, 7, 30, 144, 699, 700):
            print('  t = %3d  maximising pair' % t, argmax_pair(P, t))
    (l1, C1), (l2, C2) = out[1], out[2]
    print('crossing time', mp.nstr(mp.log(C1 / C2) / mp.log(l2[1] / l1[1]), 8))
