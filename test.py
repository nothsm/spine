import sys

import mlx as mx
import numpy as np

from main import *

def rnncellfwd_basic():
    np.random.seed(546)

    cell = rnncellnew(2, 3)
    (wx, wh) = cell
    assert np.allclose(wx, np.array([[-0.06553868,  0.40492537,  1.05272082],
                                     [-0.01006521,  2.69813824, -0.00936148]]))
    assert np.allclose(wh, np.array([[-0.03541319,  0.97702002, -0.14594773],
                                     [-1.17679928, -0.59678869, -0.12321351],
                                     [1.06107138,  2.07162012, -0.08470909]]))
    xt = np.array([1., 2.])
    hprev = np.array([3., 4., 5.])

    ht = rnncellfwd(cell, xt, hprev)
    assert np.allclose(ht, np.array([ 0.38528493,  1., -0.30972828]))

def rnnfwd_basic():
    assert False

TESTS = [
    ("rnncell_basic", rnncellfwd_basic),
    ("rnnfwd_basic", rnnfwd_basic)
]

def main():
    npass = 0
    for testname, test in TESTS:
        try:
            test()
            npass += 1
            print(f"{testname} - pass")
        except AssertionError as e:
            print(f"{testname} - error")
    print(f"{npass}/{len(TESTS)} tests pass")

    

if __name__ == '__main__':
    main()