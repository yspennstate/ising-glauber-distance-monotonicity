"""Interval certificate for the 5-cycle example: couplings (x, 2, 1/2, 4, 2) around the cycle,
discrete-time heat-bath dynamics, dbar(3) at x = 1/2 and x = 1.

Every coupling is a multiple of 1/2, so every local field h is a multiple of 1/2 and every
conditional probability is 1/(1 + exp(-1)**(2h)). exp(-1) is enclosed between two partial sums of
its alternating Taylor series, and all further arithmetic uses the outward-rounded integer
intervals of interval.py. No floating point is used.

    python certify_cycle.py
"""
from fractions import Fraction
import itertools
import json
from pathlib import Path

from interval import S, ceil_div, mul, product, diameter, decimal

N = 5


def e_interval():
    # Alternating series for exp(-1); terms decrease from k = 1 on. The partial sum through
    # k = 200 (last term positive) is an upper bound, adding the k = 201 term gives a lower bound.
    term = Fraction(1)
    partial = term
    for k in range(1, 201):
        term *= Fraction(-1, k)
        partial += term
    upper = partial
    lower = partial + term * Fraction(-1, 201)
    return (lower.numerator * S // lower.denominator, ceil_div(upper.numerator * S, upper.denominator))


def conditional(twice_field, z):
    power = (S, S)
    for _ in range(abs(twice_field)):
        power = mul(power, z)
    lo = S * S // (S + power[1])
    hi = ceil_div(S * S, S + power[0])
    return (lo, hi) if twice_field >= 0 else (S - hi, S - lo)


def build(x, z):
    states = list(itertools.product((-1, 1), repeat=N))
    ix = {s: i for i, s in enumerate(states)}
    J = [x, Fraction(2), Fraction(1, 2), Fraction(4), Fraction(2)]
    edges = [(i, (i + 1) % N, J[i]) for i in range(N)]
    p = [[(0, 0) for _ in states] for _ in states]
    for a, s in enumerate(states):
        for site in range(N):
            h = sum(j * (s[w] if u == site else s[u]) for u, w, j in edges if site in (u, w))
            assert (2 * h).denominator == 1
            lo, hi = conditional(int(2 * h), z)
            for spin, prob in ((1, (lo, hi)), (-1, (S - hi, S - lo))):
                target = list(s)
                target[site] = spin
                b = ix[tuple(target)]
                old = p[a][b]
                p[a][b] = (old[0] + prob[0] // N, old[1] + ceil_div(prob[1], N))
    return p


if __name__ == '__main__':
    z = e_interval()
    report = {'scale_digits': 100, 'exp_minus_1': [decimal(k) for k in z]}
    bounds = []
    for x in (Fraction(1, 2), Fraction(1)):
        p = build(x, z)
        a = p
        for _ in range(2):
            a = product(a, p)
        d = diameter(a)
        bounds.append(d)
        report[str(x)] = {'dbar3_lower': decimal(d[0]), 'dbar3_upper': decimal(d[1]),
                          'interval_width_units': d[1] - d[0]}
    delta = (bounds[1][0] - bounds[0][1], bounds[1][1] - bounds[0][0])
    report['delta'] = [decimal(k) for k in delta]
    assert delta[1] < 0, 'Certificate failed: intervals overlap.'
    report['strict_decrease_certified'] = True
    Path('certify_cycle.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: str(vv)[:40] for kk, vv in v.items()})
                      for k, v in report.items() if k != 'exp_minus_1'}, indent=2))
