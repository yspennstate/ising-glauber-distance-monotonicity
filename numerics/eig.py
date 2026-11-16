"""Spectrum of the heat-bath kernel for the path (2, x, 4), x = 1, 2, at 50 digits; prints the four largest eigenvalues and
their multiplicities (to 1e-30), and the maximising pair of dbar(t) at t = 7, 30, 144.

    python eig.py
"""
import itertools

import mpmath as mp

mp.mp.dps = 50
S = list(itertools.product((-1, 1), repeat=4))
ix = {s: i for i, s in enumerate(S)}


def kernel(x):
    E = [(0, 1, 2), (1, 2, x), (2, 3, 4)]
    P = mp.zeros(16, 16)
    for a, s in enumerate(S):
        for v in range(4):
            h = sum(j * (s[u] if w == v else s[w]) for u, w, j in E if v in (u, w))
            pp = 1 / (1 + mp.e ** (-2 * h))
            for spin, q in ((1, pp), (-1, 1 - pp)):
                t = list(s)
                t[v] = spin
                P[a, ix[tuple(t)]] += q / 4
    return P


for x in (1, 2):
    P = kernel(x)
    w = [mp.exp(sum(j * s[u] * s[v] for u, v, j in [(0, 1, 2), (1, 2, x), (2, 3, 4)])) for s in S]
    r = [mp.sqrt(wi) for wi in w]
    A = mp.matrix(16, 16)
    for i in range(16):
        for k in range(16):
            A[i, k] = r[i] * P[i, k] / r[k]
    A = (A + A.T) / 2
    ev = sorted([mp.re(e) for e in mp.eigsy(A)[0]], reverse=True)
    print("x=%d top eigenvalues:" % x, [mp.nstr(e, 22) for e in ev[:5]], "min", mp.nstr(ev[-1], 8))
    for t in (7, 30, 144):
        M = P ** t
        best = max(((sum(abs(M[a, k] - M[b, k]) for k in range(16)) / 2, a, b) for a in range(16) for b in range(a)),
                   key=lambda z: z[0])
        print("  t=%d max pair %s %s dbar %s" % (t, S[best[1]], S[best[2]], mp.nstr(best[0], 22)))
