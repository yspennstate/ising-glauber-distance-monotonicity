"""Discrete-time setting of the question (Levin-Peres 2nd ed. sec. 26.1 Question 2, last paragraph): discrete-time Glauber
(heat-bath) dynamics, P = (1/n) sum_v K_v, couplings J_uv > 0 on the edges of a fixed graph. Is dbar(t) = max over pairs of
||P^t(s,.) - P^t(e,.)||_TV monotone increasing in each J_uv for each integer t? Exact powers of P (stable, no spectral
division). Sweeps one edge's J on a grid (edges stay edges: grid starts above 0).

    python tvdisc.py N1 N2 TRIALS TMAX
"""
import sys

import numpy as np

from tvmono import generator, random_J, states

TOL = 1e-10


def dbar_disc(J, S, tmax):
    n = S.shape[1]
    Q, pi = generator(J, S)
    P = np.eye(len(Q)) + Q / n          # rate-1-per-vertex generator / n = random-vertex heat bath minus identity
    out, M = [], np.eye(len(Q))
    for t in range(1, tmax + 1):
        M = M @ P
        out.append(0.5 * np.abs(M[:, None, :] - M[None, :, :]).sum(-1).max())
    return np.array(out)


if __name__ == "__main__":
    n1, n2, T, TMAX = (int(a) for a in sys.argv[1:5])
    rng = np.random.default_rng(43)
    for n in range(n1, n2 + 1):
        S = states(n)
        worst = (0.0, None)
        for trial in range(T):
            J = random_J(n, rng)
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if J[i, j] > 0]
            if not edges:
                continue
            i, j = edges[rng.integers(len(edges))]
            grid = np.linspace(0.05, rng.choice([0.5, 1.5, 4.0]), 13)
            C = []
            for g in grid:
                J2 = J.copy()
                J2[i, j] = J2[j, i] = g
                C.append(dbar_disc(J2, S, TMAX))
            C = np.array(C)
            D = C[:-1] - C[1:]
            k = np.unravel_index(np.argmax(D), D.shape)
            if D[k] > worst[0]:
                worst = (D[k], (J.round(3).tolist(), (i, j), grid[k[0]], grid[k[0] + 1], k[1] + 1))
            if D[k] > TOL:
                print("DECREASE n=%d edge=(%d,%d) J_e %.4f->%.4f t=%d drop=%.3e J=%s" % (
                    n, i, j, grid[k[0]], grid[k[0] + 1], k[1] + 1, D[k], J.round(3).tolist()), flush=True)
        print("n=%d trials=%d largest decrease=%.3e at %s" % (n, T, worst[0], worst[1]), flush=True)
