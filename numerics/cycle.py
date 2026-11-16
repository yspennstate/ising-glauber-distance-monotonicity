"""The 5-cycle with couplings (x, 2, 1/2, 4, 2) around the cycle, discrete-time heat-bath dynamics:
dbar(3) at x = 1/2 and x = 1, at 80 significant digits, maximised over all 496 pairs of states.

    python cycle.py
"""
import itertools

import mpmath as mp

mp.mp.dps = 80
N = 5
S = list(itertools.product((-1, 1), repeat=N))
IX = {s: i for i, s in enumerate(S)}


def kernel(J):
    E = [(i, (i + 1) % N, J[i]) for i in range(N)]
    P = mp.zeros(len(S), len(S))
    for a, s in enumerate(S):
        for v in range(N):
            h = sum(j * (s[w] if u == v else s[u]) for u, w, j in E if v in (u, w))
            p = 1 / (1 + mp.exp(-2 * h))
            for spin, q in ((1, p), (-1, 1 - p)):
                t = list(s)
                t[v] = spin
                P[a, IX[tuple(t)]] += q / N
    return P


def dbar(M):
    n = M.rows
    return max(sum(abs(M[a, k] - M[b, k]) for k in range(n)) / 2 for a in range(n) for b in range(a))


if __name__ == "__main__":
    for x in (mp.mpf(1) / 2, mp.mpf(1)):
        P = kernel([x, 2, mp.mpf(1) / 2, 4, 2])
        print("x = %s  dbar(3) = %s" % (mp.nstr(x, 3), mp.nstr(dbar(P ** 3), 20)))
