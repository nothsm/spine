import sys

import mlx.core as mx
import numpy as np

from main import *

ANSI_COLOR_RED = "\x1b[31m"
ANSI_COLOR_GREEN = "\x1b[32m"
ANSI_COLOR_BLUE = "\x1b[36m"
ANSI_COLOR_RESET = "\x1b[0m"
ANSI_BOLD = "\x1b[1m"
ANSI_ITALIC = "\x1b[3m"
ANSI_FG = "\x1b[38;5;11m"

PASS = "pass"
FAIL = "fail!!!"

def sgdupdate_basic():
    sgd = sgdnew(lr=1e-1) # lr=0.1

    params = {
        'w0': mx.array([1., 2.]), # you can use either np or mx arrays here
        'w1': mx.array([3., 4.]),
        'b': mx.array([5.]) 
    }
    grads = {
        'w0': mx.array([5., 16.]) ,
        'w1': mx.array([28., 39.]),
        'b': mx.array([47.]) 
    }
    params = sgdupdate(sgd, params, grads)
    assert mx.allclose(params['w0'], mx.array([0.5, 0.4]))
    assert mx.allclose(params['w1'], mx.array([0.2, 0.1]))
    assert mx.allclose(params['b'], mx.array([0.3]))

def sgdupdate_nested():
    sgd = sgdnew(lr=1e-1) # lr=0.1

    params = {
        'w0': {
            'w00': np.array([1., 2.]),
            'w01': np.array([2., 1.]),
            'b0': {
                'b00': np.array([6.]),
                'b01': np.array([7.])
            }
        },
        'w1': np.array([3., 4.]),
        'b': np.array([5.]) 
    }
    grads = {
        'w0': {
            'w00': np.array([9., 19.5]),
            'w01': np.array([18., 9.]),
            'b0': {
                'b00': np.array([50.5]),
                'b01': np.array([55.])
            }
        },
        'w1': np.array([28., 39.]),
        'b': np.array([47.]) 
    }
    params = sgdupdate(sgd, params, grads)
    assert np.allclose(params['w0']['w00'], np.array([0.1, 0.05]))
    assert np.allclose(params['w0']['w01'], np.array([0.2, 0.1]))
    assert np.allclose(params['w0']['b0']['b00'], np.array([0.95]))
    assert np.allclose(params['w0']['b0']['b01'], np.array([1.5]))
    assert np.allclose(params['w1'], np.array([0.2, 0.1]))
    assert np.allclose(params['b'], np.array([0.3]))

def rcfwd_basic():
    np.random.seed(546)

    cell = rcnew(2, 3)

    wx = mx.array([[-0.06553868,  0.40492537,  1.05272082],
                   [-0.01006521,  2.69813824, -0.00936148]])
    wh = mx.array([[-0.03541319,  0.97702002, -0.14594773],
                   [-1.17679928, -0.59678869, -0.12321351],
                   [1.06107138,  2.07162012, -0.08470909]])
    rcinit(cell, wx, wh)
    assert mx.allclose(cell['wx'], mx.array([[-0.06553868,  0.40492537,  1.05272082],
                                             [-0.01006521,  2.69813824, -0.00936148]]))
    assert mx.allclose(cell['wh'], mx.array([[-0.03541319,  0.97702002, -0.14594773],
                                             [-1.17679928, -0.59678869, -0.12321351],
                                             [1.06107138,  2.07162012, -0.08470909]]))
    xt = mx.array([1., 2.])
    hprev = mx.array([3., 4., 5.])

    ht = rcfwd(cell, xt, hprev)
    assert mx.allclose(ht, mx.array([0.38528493,  1., -0.30972828]))

def rcclassify_basic():
    np.random.seed(546)

    cell = rcnew(2, 3)

    wx = mx.array([[-0.06553868,  0.40492537,  1.05272082],
                   [-0.01006521,  2.69813824, -0.00936148]])
    wh = mx.array([0.5])
    rcinit(cell, wx, wh)
    assert mx.allclose(cell['wx'], mx.array([[-0.06553868,  0.40492537,  1.05272082],
                                             [-0.01006521,  2.69813824, -0.00936148]]))
    assert mx.allclose(cell['wh'], mx.array([0.5]))
    xt = mx.array([1., 2.])
    hprev = mx.array([3.])

    preds = rcclassify(cell, xt, hprev)

    assert False
    # assert mx.allclose(ht, mx.array([0.38528493,  1., -0.30972828]))

# use seq of length 3 and h0 = [0, 0, 1]
def rnnfwd_basic():
    assert False

TESTS = [
    ("sgdupdate_basic", sgdupdate_basic),
    ("sgdupdate_nested", sgdupdate_nested),
    ("rcfwd_basic", rcfwd_basic),
    ("rcclassify_basic", rcclassify_basic),
    ("rnnfwd_basic", rnnfwd_basic)
]

# TODO: Add color to the tests
def main():
    npass = 0
    for testname, test in TESTS:
        try:
            test()
            npass += 1
            print(f"{testname} - {ANSI_COLOR_BLUE}{PASS}{ANSI_COLOR_RESET}")
        except AssertionError as e:
            print(f"{testname} - {ANSI_COLOR_RED}{FAIL}{ANSI_COLOR_RESET}")
    print(f"{npass}/{len(TESTS)} tests pass")
    

if __name__ == '__main__':
    main()