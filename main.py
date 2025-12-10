import math
import time
from functools import partial
from typing import Any, TypedDict

import mlx.core as mx
from mlx.utils import tree_map
import numpy as np
import numpy.typing as npt


# -------------------------------- type alias ----------------------------------

Array = mx.array
NPArray = npt.NDArray[Any] # TODO: tighten this
Dataset = tuple[NPArray, NPArray, NPArray, NPArray]

# ---------------------------------- datasets ----------------------------------

def load_bixor(n_samples: int = 32):
    ...

# TODO
def load_xor():
    ...

# TODO
def load_tsp5():
    ...

# TODO
def load_tsp10():
    ...

# TODO: Return a dict here?
def load_donut(n_train: int = 32, n_test: int = 8, r: float = 0.5) -> Dataset:
    X_train = np.random.uniform(-1, 1, size=(n_train, 2))
    y_train = np.array([1. if math.sqrt(x[0]**2 + x[1]**2) < r else -1. for x in X_train.tolist()])
    X_test = np.random.uniform(-1, 1, size=(n_test, 2))
    y_test = np.array([1. if math.sqrt(x[0]**2 + x[1]**2) < r else -1. for x in X_test.tolist()])
    return X_train, y_train, X_test, y_test

def load_regression(ntrain=32, ntest=8, input_dim=2, return_params=False):
    true_params = np.random.normal(size=(input_dim,))

    eps_train = 1e-2 * np.random.normal(size=(ntrain,))
    X_train = np.random.normal(size=(ntrain, input_dim))
    y_train = (X_train @ true_params) + eps_train

    eps_test = 1e-2 * np.random.normal(size=(ntest,))
    X_test = np.random.normal(size=(ntest, input_dim))
    y_test = (X_test @ true_params) + eps_test

    if return_params:
        return X_train, y_train, X_test, y_test, true_params
    else:
        return X_train, y_train, X_test, y_test

def load_tanh(ntrain=32, ntest=8, input_dim=2, return_params=False):
    tmp = load_regression(ntrain=ntrain, ntest=ntest, input_dim=input_dim, return_params=return_params)
    if return_params:
        X_train, y_train, X_test, y_test, true_params = tmp
        y_train = np.tanh(y_train)
        y_test = np.tanh(y_test)
        return X_train, y_train, X_test, y_test, true_params
    else:
        X_train, y_train, X_test, y_test = tmp
        y_train = np.tanh(y_train)
        y_test = np.tanh(y_test)
        return X_train, y_train, X_test, y_test

def load_sincos(nsamples=32):
    ...

# ----------------------------- optimizers/solvers -----------------------------

class MySGD(TypedDict):
    lr: float

def sgdnew(lr: float) -> MySGD:
    return MySGD(lr=lr)

# TODO: Should I make this private?
def sgdupdate1(sgd, param, grad):
    lr = sgd['lr']
    return param - lr * grad

def sgdupdate(sgd: MySGD, params, grads):
    return tree_map(lambda param, grad: sgdupdate1(sgd, param, grad), params, grads)

def sgdsolve(sgd, model, X, y, batch_size, n_epochs=10, print_every=None):
    (fwd, params) = model
    mx.eval(params)

    # pre: ypreds.shape == y.shape
    def mse(params, X, y):
        ypreds = fwd(params, X)
        lossval = (ypreds - y).square().mean()
        return lossval

    loss_and_grad_fn = mx.value_and_grad(mse)

    @mx.compile
    def step(params, X, y):
        loss, grads = loss_and_grad_fn(params, X, y)
        params = sgdupdate(sgd, params, grads)
        return params, loss

    def batch_iterate(batch_size, X, y):
        perm = mx.array(np.random.permutation(y.size))
        for i in range(0, y.size, batch_size):
            ixs = perm[i:i + batch_size]
            yield X[ixs], y[ixs]

    steps, metrics = 0, []
    for e in range(n_epochs): # TODO: increase this
        for X_, y_ in batch_iterate(batch_size, X, y): # TODO: Doesn't this have overhead?
            # train step
            tic = time.perf_counter_ns()

            params, loss = step(params, X_, y_)
            mx.eval(loss, params)

            toc = time.perf_counter_ns()

            steps += 1

            # save metrics
            step_loss, step_dt = loss.item(), toc - tic
            metrics.append({'loss': step_loss, 'dt': step_dt})
            if print_every and (steps % print_every) == 0:
                print(f"step: {steps} | epoch: {e} | loss: {step_loss:.5f} | dt: {step_dt}ns")
    return params, metrics


# TODO
class MyAdam(TypedDict):
    ...

# TODO
def adamnew():
    ...

# TODO
def adamupdate():
    ...

# ---------------------------------- models ------------------------------------

# TODO: Support bias
class RNNCell(TypedDict):
    wx: Array
    wh: Array

def rnncellnew(input_dim, hidden_dim) -> RNNCell:
    assert hidden_dim == 1

    wx = 1e-2 * mx.random.normal(shape=(input_dim,)) # TODO: Use better init
    wh = 1e-2 * mx.random.normal(shape=(hidden_dim, hidden_dim))
    return RNNCell(wx=wx, wh=wh)

def rnncellinit(cell, wx, wh):
    cell['wx'], cell['wh'] = mx.array(wx), mx.array(wh)

# TODO: Play with this activation
# Note: If any of the entries is large, tanh makes it go to 1
def rnncellfwd(cell, xt, hprev=None):
    wx, wh = cell['wx'], cell['wh']
    hprev = hprev or mx.zeros(wh.shape[0]) # TODO: how does this interact with compilation?
    ht = mx.tanh(mx.matmul(xt, wx) + mx.matmul(hprev, wh)) # TODO: Why are the matmuls transposed?
    return ht

class RNN(TypedDict):
    cell: RNNCell

def rnnnew(input_dim: int, hidden_dim: int) -> RNN:
    cell = rnncellnew(input_dim=input_dim,
                      hidden_dim=hidden_dim)
    return RNN(cell=cell)

# TODO
def rnnfwd(rnn: RNN, xs, h0):
    y, h = ...
    return y, h

# # TODO
# class MyDataset(TypedDict):
#     ...

# # TODO
# class MyDataLoader(TypedDict):
#     ...

# # TODO
# class MyLRScheduler(TypedDict):
#     ...

# TODO
class RNNEncoder(TypedDict):
    ...

# TODO
class RNNDecoder(TypedDict):
    ...

# TODO
class RNNPointerNet(TypedDict):
    ...

# TODO
class MyNCO(TypedDict):
    embedding: npt.NDArray[np.float32]
    actor_net: RNNPointerNet

# TODO: I don't think I need Attention, PointerNet for just using RNN's?

# TODO
class Encoder(TypedDict):
    ...

# TODO
class Attention(TypedDict):
    ...

# TODO
class Decoder(TypedDict):
    ...

# TODO
class PointerNet(TypedDict):
    ...

# TODO
class NCO(TypedDict):
    ...

# -----------

# TODO
def reward(solution):
    ...

# TODO
def load_tsp5_data():
    raise NotImplementedError()

def main():
    print("Hello from spine!")

if __name__ == "__main__":
    main()

# TODO:
# - [ ] Add shebang to use current uv environment
# - [ ] Implement data/tsp/prepare.py
# - [ ] Implement data/ARC-AGI/prepare.py
# - [ ] Implement data/ARC-AGI-2/prepare.py
# - [ ] Implement or find supervised learning baseline
# - [ ] Implement or find Christofides baseline
# - [ ] Implement or find OR-Tools baseline
# - [ ] Implement or find optimal baseline (use Concorde)
# - [ ] How does the Python object system work?
# - [ ] Try generating data from a different distribution than U([0, 1]^2)
# - [ ] Fun: pure C implementation
# - [ ] only do stochastic decoding