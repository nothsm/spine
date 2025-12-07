import math
from functools import partial
from typing import TypedDict 

import mlx.core as mx
from mlx.utils import tree_map
import numpy as np
import numpy.typing as npt


# -------------------------------- type alias ----------------------------------

Array = npt.NDArray[np.float32]

# ---------------------------------- datasets ----------------------------------

def load_bixor(nsamples=32):
    ...

# TODO: Return a dict here?
def load_donut(ntrain=32, ntest=8, r=0.5):
    X_train = np.random.uniform(-1, 1, size=(ntrain, 2))
    y_train = np.array([1. if math.sqrt(x[0]**2 + x[1]**2) < r else -1. for x in X_train.tolist()])
    X_test = np.random.uniform(-1, 1, size=(ntest, 2))
    y_test = np.array([1. if math.sqrt(x[0]**2 + x[1]**2) < r else -1. for x in X_test.tolist()])
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

def sgdupdate(sgd, params, grads):
    return tree_map(lambda param, grad: sgdupdate1(sgd, param, grad), params, grads) 

def sgdsolve(sgd, model, X, y):
    (fwd, params) = model # params <=> cell

    def mse(params, X, y):
        ypreds = fwd(params, X) # TODO: parameterize this
        lossval = (ypreds - y).square().mean()
        return lossval

    loss_and_grad_fn = mx.value_and_grad(mse)
    for e in range(10): # TODO: increase this
        loss, grads = loss_and_grad_fn(params, X, y)
        params = sgdupdate(sgd, params, grads) # TODO: I don't like that this allocates
        mx.eval(params) # TODO: Why do I need this?
        print(loss)
    return params


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
    wx = mx.random.normal(shape=(input_dim, hidden_dim)) # TODO: Use better init
    wh = mx.random.normal(shape=(hidden_dim, hidden_dim))
    return RNNCell(wx=wx, wh=wh)

def rnncellinit(cell, wx, wh):
    cell['wx'] = mx.array(wx)
    cell['wh'] = mx.array(wh)

# TODO: Play with this activation
# Note: If any of the entries is large, tanh makes it go to 1
def rnncellfwd(cell, xt, hprev=None):
    wx, wh = cell['wx'], cell['wh']

    # TODO: Write this in a more concise way
    if hprev is None:
        hprev = mx.zeros(wh.shape[0])

    ht = mx.tanh(mx.matmul(xt, wx) + mx.matmul(hprev, wh)) # TODO: Why are the matmuls transposed?
    return ht

# TODO
def rnncellbwd(cell, dht, cache):
    wx, wh = cell['wx'], cell['wh']
    (xt, prevh, ht) = cache
    ...

class RNN(TypedDict):
    cell: RNNCell

def rnnnew(input_dim, hidden_dim):
    cell = rnncellnew(input_dim, hidden_dim)
    return RNN(cell)

# TODO
def rnnfwd(rnn: RNN, xs, h0):
    y, h = ...
    return y, h

# TODO
class MyDataset(TypedDict):
    ...

# TODO
class MyDataLoader(TypedDict):
    ...

# TODO
class MyLRScheduler(TypedDict):
    ...

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