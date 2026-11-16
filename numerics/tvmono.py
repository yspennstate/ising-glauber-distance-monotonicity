"""Continuous-time setting (Levin-Peres, Markov Chains and Mixing Times 2nd ed., sec. 26.1, Question 2): zero-field ferromagnetic Ising
heat-bath chain, every vertex updating at rate one. Is dbar_J(t) = max_{s,e} ||P_t(s,.) - P_t(e,.)||_TV nondecreasing in
every coupling J_uv >= 0, for every fixed t?

Exact: 2^n states, generator Q, P_t = exp(tQ) through the symmetrised generator (reversible). For random coupling
matrices (some entries zero) pick an edge, sweep J_uv on a grid, and look for dbar(t) decreasing in J_uv at any t on a
grid. Then refine the worst case by local search. Prints every decrease larger than TOL.

    python tvmono.py N1 N2 TRIALS
"""
import itertools
import sys

import numpy as np

TOL = 1e-9
TS = np.concatenate([np.linspace(0.02, 1, 25), np.linspace(1.1, 6, 25), np.linspace(6.5, 30, 15)])


def states(n):
    return np.array(list(itertools.product([-1, 1], repeat=n)), dtype=float)


def generator(J, S):
    N, n = S.shape
    E = -0.5 * np.einsum("ai,ij,aj->a", S, J, S)          # energy H = -sum_{i<j} J_ij s_i s_j
    pi = np.exp(-(E - E.min()))
    pi /= pi.sum()
    Q = np.zeros((N, N))
    idx = {tuple(s): a for a, s in enumerate(S.astype(int))}
    for a in range(N):
        for v in range(n):
            s = S[a].copy()
            h = J[v] @ s - J[v, v] * s[v]
            s[v] = -s[v]
            b = idx[tuple(s.astype(int))]
            p_new = 1 / (1 + np.exp(-2 * h * s[v]))           # heat bath: P(s_v = new value | rest)
            Q[a, b] += p_new
            Q[a, a] -= p_new
    return Q, pi


def dbar_curve(J, S, ts):
    Q, pi = generator(J, S)
    r = np.sqrt(pi)
    A = (r[:, None] * Q) / r[None, :]
    A = (A + A.T) / 2
    lam, U = np.linalg.eigh(A)
    out = []
    for t in ts:
        P = (U * np.exp(lam * t)) @ U.T
        P = P / r[:, None] * r[None, :]
        d = 0.5 * np.abs(P[:, None, :] - P[None, :, :]).sum(-1).max()
        out.append(d)
    return np.array(out)


def random_J(n, rng):
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < rng.uniform(0.3, 1.0):
                J[i, j] = J[j, i] = rng.exponential(rng.choice([0.2, 0.6, 1.5]))
    return J


if __name__ == "__main__":
    n1, n2, T = (int(a) for a in sys.argv[1:4])
    rng = np.random.default_rng(41)
    for n in range(n1, n2 + 1):
        S = states(n)
        worst = (0.0, None)
        for trial in range(T):
            J = random_J(n, rng)
            i, j = sorted(rng.choice(n, 2, replace=False))
            grid = np.linspace(0, rng.choice([0.5, 1.5, 4.0]), 13)
            curves = []
            for g in grid:
                J2 = J.copy()
                J2[i, j] = J2[j, i] = g
                curves.append(dbar_curve(J2, S, TS))
            C = np.array(curves)                          # [grid, t]
            drop = (C[:-1] - C[1:]).max()                 # largest decrease in J at a fixed t
            if drop > worst[0]:
                k = np.unravel_index(np.argmax(C[:-1] - C[1:]), C[:-1].shape)
                worst = (drop, (J.round(3).tolist(), (i, j), grid[k[0]], grid[k[0] + 1], TS[k[1]]))
            if drop > TOL:
                k = np.unravel_index(np.argmax(C[:-1] - C[1:]), C[:-1].shape)
                print("DECREASE n=%d edge=(%d,%d) J_e %.3f->%.3f t=%.3f drop=%.3e J=%s" % (
                    n, i, j, grid[k[0]], grid[k[0] + 1], TS[k[1]], drop, J.round(3).tolist()), flush=True)
        print("n=%d trials=%d largest decrease=%.3e at %s" % (n, T, worst[0], worst[1]), flush=True)
