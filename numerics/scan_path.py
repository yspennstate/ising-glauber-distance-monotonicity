"""Checks for the path and cycle examples.
(a) Kernel test against a theorem: Nacu (2003) proved the spectral gap of Glauber dynamics on the cycle is nonincreasing in
    every coupling, i.e. lambda_2 nondecreasing. Check lambda_2 (second-largest eigenvalue of the discrete kernel) over all
    C4 and C5 assignments in {1/2,1,2,4} for every one-step increase of one edge. Any violation means the kernel is wrong.
(b) Path 0-1-2-3, J = (2, x, 4): lambda_2 at x = 1, 2; the set of t <= 3000 where dbar(x=2) < dbar(x=1), and where the
    order flips back.
(c) Continuous-time value for the path example (each vertex at rate one), t on a grid.

    python check.py
"""
import itertools

import numpy as np
from scipy.linalg import expm

from tvdisc import dbar_disc
from tvmono import generator, states

VALS = [0.5, 1.0, 2.0, 4.0]


def lam2(J, S):
    n = S.shape[1]
    Q, pi = generator(J, S)
    P = np.eye(len(Q)) + Q / n
    r = np.sqrt(pi)
    A = (r[:, None] * P) / r[None, :]
    ev = np.sort(np.linalg.eigvalsh((A + A.T) / 2))
    return ev[-2], ev[0]


def cycleJ(n, w):
    J = np.zeros((n, n))
    for i in range(n):
        J[i, (i + 1) % n] = J[(i + 1) % n, i] = w[i]
    return J


if __name__ == "__main__":
    for n in (4, 5):
        S = states(n)
        L = {v: lam2(cycleJ(n, [VALS[k] for k in v]), S)[0] for v in itertools.product(range(4), repeat=n)}
        viol = 0
        worst = 0.0
        for v, l in L.items():
            for e in range(n):
                if v[e] < 3:
                    u = list(v)
                    u[e] += 1
                    d = l - L[tuple(u)]
                    worst = max(worst, d)
                    viol += d > 1e-12
        print("(a) C%d: %d assignments, lambda_2 decreases in a coupling %d times (largest %.2e)" % (n, len(L), viol, worst),
              flush=True)
    S = states(4)
    Jp = {x: np.array([[0, 2, 0, 0], [2, 0, x, 0], [0, x, 0, 4], [0, 0, 4, 0]], float) for x in (1.0, 2.0)}
    for x in (1.0, 2.0):
        l2, lmin = lam2(Jp[x], S)
        print("(b) path (2,%g,4): lambda_2 = %.12f  smallest eigenvalue %.6f" % (x, l2, lmin), flush=True)
    T = 3000
    c1, c2 = dbar_disc(Jp[1.0], S, T), dbar_disc(Jp[2.0], S, T)
    dec = np.nonzero(c2 < c1 - 1e-13)[0] + 1
    runs = np.split(dec, np.nonzero(np.diff(dec) > 1)[0] + 1) if len(dec) else []
    print("(b) dbar(x=2) < dbar(x=1) at %d of t=1..%d; runs %s" % (len(dec), T, [(int(r[0]), int(r[-1])) for r in runs]),
          flush=True)
    k = int(np.argmax(c1 - c2))
    print("(b) largest drop %.4e at t=%d; at t=%d: %.6e vs %.6e" % ((c1 - c2)[k], k + 1, T, c1[-1], c2[-1]), flush=True)
    for t in (5.0, 10.0, 20.0, 30.0, 40.0, 60.0):
        d = []
        for x in (1.0, 2.0):
            Q, _ = generator(Jp[x], S)
            M = expm(Q * t)
            d.append(0.5 * np.abs(M[:, None, :] - M[None, :, :]).sum(-1).max())
        print("(c) continuous path t=%g: dbar %.12f -> %.12f diff %.3e" % (t, d[0], d[1], d[1] - d[0]), flush=True)
