"""Interval certificate for every time 1 <= t <= 700 of the heat-bath chain on the path (2, x, 4), x = 1, 2.
Uses the outward-rounded integer interval arithmetic of interval.py (units of 1e-100, no floating point).
For each t it records whether dbar_2(t) - dbar_1(t) is certified negative, certified positive, or undecided, and
writes the enclosures at t = 7, 30, 144, 699, 700 and the smallest certified margin on 7..699.

    python certify_all.py
"""
import json

from interval import S, build, decimal, diameter, product, z_interval

T_MAX = 700

if __name__ == "__main__":
    z = z_interval()
    mats = {x: build(x, z) for x in (1, 2)}
    pw = {x: [[(S, S) if i == j else (0, 0) for j in range(16)] for i in range(16)] for x in (1, 2)}
    sign, keep, widest = {}, {}, 0
    for t in range(1, T_MAX + 1):
        pw = {x: product(pw[x], mats[x]) for x in (1, 2)}
        d1, d2 = diameter(pw[1]), diameter(pw[2])
        lo, hi = d2[0] - d1[1], d2[1] - d1[0]
        sign[t] = "neg" if hi < 0 else "pos" if lo > 0 else "undecided"
        widest = max(widest, d1[1] - d1[0], d2[1] - d2[0])
        if t in (6, 7, 30, 144, 699, 700):
            keep[t] = {"dbar_x1": [decimal(k)[:26] for k in d1], "dbar_x2": [decimal(k)[:26] for k in d2],
                       "delta": [decimal(k)[:28] for k in (lo, hi)]}
        if t % 50 == 0:
            print(t, sign[t], flush=True)
    neg = [t for t in sign if sign[t] == "neg"]
    out = {"T_MAX": T_MAX, "negative_times": [min(neg), max(neg)] if neg else None,
           "negative_count": len(neg), "all_negative_7_699": all(sign[t] == "neg" for t in range(7, 700)),
           "positive_times_1_6": [t for t in range(1, 7) if sign[t] == "pos"],
           "positive_times_700": sign[T_MAX] == "pos", "undecided": [t for t in sign if sign[t] == "undecided"],
           "widest_enclosure_units_1e-100": widest, "enclosures": keep}
    print(json.dumps({k: v for k, v in out.items() if k != "enclosures"}))
    with open("certify_all.json", "w") as f:
        json.dump(out, f, indent=2)
