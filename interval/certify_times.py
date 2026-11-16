"""Interval certificates for the path example at several times, for the heat-bath chain P and the lazy chain (I+P)/2.
Uses the outward-rounded integer interval arithmetic of interval.py (units of 1e-100, no floating point).

    python certify_times.py
"""
import json

from interval import S, build, ceil_div, decimal, diameter, product, z_interval

TIMES = {"heat-bath": [7, 30, 144], "lazy": [30, 144]}


def lazy(p):
    n = len(p)
    return [[((p[i][j][0] + (S if i == j else 0)) // 2, ceil_div(p[i][j][1] + (S if i == j else 0), 2))
             for j in range(n)] for i in range(n)]


if __name__ == "__main__":
    z = z_interval()
    out = {}
    for kind, ts in TIMES.items():
        mats = {x: build(x, z) for x in (1, 2)}
        if kind == "lazy":
            mats = {x: lazy(m) for x, m in mats.items()}
        pw = {x: [[(S, S) if i == j else (0, 0) for j in range(16)] for i in range(16)] for x in (1, 2)}
        t = 0
        for target in ts:
            while t < target:
                pw = {x: product(pw[x], mats[x]) for x in (1, 2)}
                t += 1
            d1, d2 = diameter(pw[1]), diameter(pw[2])
            delta = (d2[0] - d1[1], d2[1] - d1[0])
            out["%s t=%d" % (kind, t)] = {"dbar_x1": [decimal(k)[:24] for k in d1], "dbar_x2": [decimal(k)[:24] for k in d2],
                                          "delta": [decimal(k)[:26] for k in delta], "certified_decrease": delta[1] < 0}
            print(kind, t, out["%s t=%d" % (kind, t)]["delta"], delta[1] < 0, flush=True)
    with open("certify_times.json", "w") as f:
        json.dump(out, f, indent=2)
