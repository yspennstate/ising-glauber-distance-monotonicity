"""Rational interval arithmetic for the path example (heat-bath kernel, couplings (2, x, 4)).

No floating point or transcendental library is used. Intervals have integer
endpoints in units of 10**-100, and every rounding is outward.
"""
from fractions import Fraction
import itertools
import json
from pathlib import Path

DIGITS = 100
S = 10**DIGITS


def ceil_div(a,b):
    return -((-a)//b)


def mul(a,b):
    # All probability intervals here are nonnegative.
    assert a[0]>=0 and b[0]>=0
    return a[0]*b[0]//S,ceil_div(a[1]*b[1],S)


def z_interval():
    # Alternating Taylor series for exp(-2); terms decrease after k=2.
    # The even partial sum through k=200 is an upper bound and the next
    # odd partial sum is a lower bound, by the alternating remainder theorem.
    term = Fraction(1)
    partial = term
    for k in range(1,201):
        term *= Fraction(-2,k)
        partial += term
    upper = partial
    lower = partial + term*Fraction(-2,201)
    return (lower.numerator*S//lower.denominator,
            ceil_div(upper.numerator*S,upper.denominator))


def conditional(field,z):
    power = (S,S)
    for _ in range(abs(field)):
        power = mul(power,z)
    # p(+ | field >= 0) = 1/(1+exp(-2*field)).
    lo = S*S//(S+power[1])
    hi = ceil_div(S*S,S+power[0])
    return (lo,hi) if field>=0 else (S-hi,S-lo)


def build(x,z):
    states=list(itertools.product((-1,1),repeat=4))
    ix={s:i for i,s in enumerate(states)}
    p=[[(0,0) for _ in states] for _ in states]
    edges=[(0,1,2),(1,2,x),(2,3,4)]
    for a,s in enumerate(states):
        for site in range(4):
            h=sum(j*s[v] if u==site else j*s[u] if v==site else 0
                  for u,v,j in edges)
            lo,hi=conditional(h,z)
            for spin,prob in [(1,(lo,hi)),(-1,(S-hi,S-lo))]:
                target=list(s)
                target[site]=spin
                b=ix[tuple(target)]
                old=p[a][b]
                p[a][b]=(old[0]+prob[0]//4,old[1]+ceil_div(prob[1],4))
    return p


def product(a,b):
    n=len(a)
    result=[[(0,0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            # Sum exact integer products before one outward division.
            result[i][j]=(sum(a[i][k][0]*b[k][j][0] for k in range(n))//S,
                          ceil_div(sum(a[i][k][1]*b[k][j][1] for k in range(n)),S))
    return result


def diameter(a):
    lower=upper=0
    for i in range(len(a)):
        for j in range(i):
            lows=highs=0
            for ai,aj in zip(a[i],a[j]):
                lo,hi=ai[0]-aj[1],ai[1]-aj[0]
                lows += 0 if lo<=0<=hi else min(abs(lo),abs(hi))
                highs += max(abs(lo),abs(hi))
            lower=max(lower,lows//2)
            upper=max(upper,ceil_div(highs,2))
    return lower,upper


def decimal(k):
    sign='-' if k<0 else ''
    k=abs(k)
    return f'{sign}{k//S}.{k%S:0{DIGITS}d}'


if __name__=='__main__':
    z=z_interval()
    report={'scale_digits':DIGITS,'exp_minus_2':[decimal(k) for k in z]}
    bounds=[]
    for x in [1,2]:
        p=build(x,z)
        a=[[(S,S) if i==j else (0,0) for j in range(16)] for i in range(16)]
        for _ in range(7):
            a=product(a,p)
        d=diameter(a)
        bounds.append(d)
        report[str(x)]={'dbar_lower':decimal(d[0]),'dbar_upper':decimal(d[1]),
                        'interval_width_units':d[1]-d[0]}
    delta=(bounds[1][0]-bounds[0][1],bounds[1][1]-bounds[0][0])
    report['delta']=[decimal(k) for k in delta]
    assert delta[1]<0, 'Certificate failed: intervals overlap.'
    report['strict_decrease_certified']=True
    Path('interval_t7.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
