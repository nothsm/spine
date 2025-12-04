from typing import NamedTuple

import mlx as mx
import numpy as np
import numpy.typing as npt

# ---------------------------------- models ------------------------------------

class RNNCell(NamedTuple):
    wx: npt.NDArray[np.float32]
    wh: npt.NDArray[np.float32]

def rnncellnew(input_dim, hidden_dim):
    wx = np.random.normal(size=(input_dim, hidden_dim)) # TODO: Use better init
    wh = np.random.normal(size=(hidden_dim, hidden_dim))
    return RNNCell(wx, wh)

# TODO: Play with this activation
# Note: If any of the entries is large, tanh makes it go to 1
def rnncellfwd(cell, xt, prevh):
    (wx, wh) = cell
    ht = np.tanh(np.matmul(xt, wx) + np.matmul(prevh, wh)) # TODO: Why backwards?
    return ht

class RNN(NamedTuple):
    Wx: npt.NDArray[np.float32]
    Wh: npt.NDArray[np.float32]
    b:  npt.NDArray[np.float32]

def rnnfwd(rnn: RNN):
    ...

# TODO
class MyDataset(NamedTuple):
    ...

# TODO
class MyDataLoader(NamedTuple):
    ...

# TODO
class MyAdam(NamedTuple):
    ...

# TODO
class MyLRScheduler(NamedTuple):
    ...

# TODO
class RNNEncoder(NamedTuple):
    ...

# TODO
class RNNDecoder(NamedTuple):
    ...

# TODO
class RNNPointerNet(NamedTuple):
    ...

# TODO
class MyNCO(NamedTuple):
    embedding: npt.NDArray[np.float32]
    actor_net: RNNPointerNet

# TODO: I don't think I need Attention, PointerNet for just using RNN's?

# TODO
class Encoder(NamedTuple):
    ...

# TODO
class Attention(NamedTuple):
    ...

# TODO
class Decoder(NamedTuple):
    ...

# TODO
class PointerNet(NamedTuple):
    ...

# TODO
class NCO(NamedTuple):
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
# - [ ] How does NamedTuple work?
# - [ ] Try generating data from a different distribution than U([0, 1]^2)
# - [ ] Fun: pure C implementation