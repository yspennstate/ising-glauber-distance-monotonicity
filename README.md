# The pairwise distance of Ising Glauber dynamics is not monotone in the couplings

Paper and code. On the path 0-1-2-3 with couplings (2, x, 4), raising x from 1 to 2 lowers the pairwise total
variation distance of the discrete-time heat-bath chain at every time t from 7 to 699, and raises it at t = 700.
This answers the second part of Question 2 of Levin and Peres, Markov Chains and Mixing Times (2nd ed., Section 26.1), the one for the pairwise distance, in the negative.
A second example on the 5-cycle, couplings (x, 2, 1/2, 4, 2), is proved the same way at t = 3.

- `paper/` the LaTeX source and PDF.
- `interval/` the proof: exact interval arithmetic with integer endpoints in units of 1e-100, rounded outward.
  `certify_all.py` checks every t from 1 to 700 and writes `certify_all.json`; `certify_times.py` gives the
  table values and the lazy chain; `interval.py` holds the arithmetic and the certificate at t = 7;
  `certify_cycle.py` proves the 5-cycle example and writes `certify_cycle.json`.
- `numerics/` the numerical statements of the paper: eigenvalues, the constants C_x, the crossing time and the
  maximising pair (`constants.py`, `eig.py`), the 5-cycle
  example (`cycle.py`), and the double-precision scans in discrete, lazy and continuous time (`scan_all.py`,
  `scan_path.py`).

Requirements: Python 3.10 or later with numpy, scipy and mpmath. The interval code uses only the standard library.
Run each script from its own folder, for example `cd interval && python certify_all.py` (about two minutes).
