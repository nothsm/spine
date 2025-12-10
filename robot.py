import math
import time
import os
from functools import partial
from typing import Any, TypedDict

import mlx.core as mx
from mlx.utils import tree_map
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

# -------------------------------- constants ----------------------------------

TSP5_TRAIN_FILE = 'data/tsp/tsp_5_train/tsp5.txt'
TSP5_TEST_FILE  = 'data/tsp/tsp_5_train/tsp5_test.txt'

TSP10_TRAIN_FILE = 'data/tsp/tsp_10_train/tsp10.txt'
TSP10_TEST_FILE = 'data/tsp/tsp_10_train/tsp10_test.txt'

TSP50_TRAIN_FILE = 'data/tsp/tsp50.txt'

# -------------------------------- type alias ----------------------------------

Array = mx.array 
NPArray = npt.NDArray[Any]

# ----------------------------------- utils ------------------------------------

def gradclip(grads, max_norm: float):
    leaves = []
    
    def _extract_arrays(tree):
        if isinstance(tree, mx.array):
            leaves.append(tree)
        elif isinstance(tree, dict):
            for v in tree.values():
                _extract_arrays(v)
        elif isinstance(tree, (list, tuple)):
            for v in tree:
                _extract_arrays(v)
        # Ignore None, scalars, etc.
    
    _extract_arrays(grads)
    
    if not leaves:
        return grads
        
    total_norm = mx.sqrt(sum(mx.sum(x**2) for x in leaves))
    
    clip_coef = max_norm / (total_norm + 1e-6)
    scale = mx.minimum(1.0, clip_coef)
    
    return tree_map(lambda g: g * scale, grads)

# ---------------------------------- datasets ----------------------------------

def load_tsp(file: str) -> tuple[NPArray, NPArray]:
    xs, ys = [], []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                x_str, y_str = line.split(' output ')
                x_words = x_str.split()
                y_words = y_str.split()
                points = [[float(x_words[i]), float(x_words[i+1])] for i in range(0, len(x_words), 2)]
                tour = [int(w) - 1 for w in y_words]
                xs.append(points)
                ys.append(tour)
            except ValueError:
                continue
                
    X = np.array(xs, dtype=np.float32).transpose(0, 2, 1) # (Batch, 2, N_Cities)
    Y = np.array(ys, dtype=np.int32)
    return X, Y

def load_tsp5() -> tuple[NPArray, NPArray]:
    return load_tsp(TSP5_TRAIN_FILE)

def load_tsp10() -> tuple[NPArray, NPArray]:
    return load_tsp(TSP10_TRAIN_FILE)

def load_tsp50() -> tuple[NPArray, NPArray]:
    return load_tsp(TSP50_TRAIN_FILE)

# ----------------------------- optimizers -----------------------------

class MySGD(TypedDict):
    lr: float

def sgdnew(lr: float) -> MySGD:
    return MySGD(lr=lr)

def sgdupdate(sgd: MySGD, params, grads):
    return tree_map(lambda param, grad: param - sgd['lr'] * grad, params, grads) 

# ---------------------------------- models ------------------------------------

class AttentionConfig(TypedDict):
    dim: int
    use_tanh: bool
    C: float

class AttentionParams(TypedDict):
    Wq: Array
    bq: Array
    We: Array
    be: Array
    v: Array

def attnnew(dim: int, use_tanh: bool = False, C: float = 10.0) -> tuple[AttentionParams, AttentionConfig]:
    bound = 1.0 / math.sqrt(dim) # he init
    params = AttentionParams(
        Wq=mx.random.uniform(-bound, bound, shape=(dim, dim)),
        bq=mx.random.uniform(-bound, bound, shape=(dim,)),
        We=mx.random.uniform(-bound, bound, shape=(dim, dim)),
        be=mx.random.uniform(-bound, bound, shape=(dim,)),
        v=mx.random.uniform(-bound, bound, shape=(dim, 1))
    )
    config = AttentionConfig(dim=dim, use_tanh=use_tanh, C=C)
    return params, config

def attnfwd(params: AttentionParams, config: AttentionConfig, query: Array, ref: Array):
    q_proj = query @ params['Wq'] + params['bq']
    ref_perm = mx.transpose(ref, (1, 0, 2))
    e = ref_perm @ params['We'] + params['be']
    
    q_expanded = mx.expand_dims(q_proj, 1)
    u_activation = mx.tanh(q_expanded + e)
    
    u = u_activation @ params['v']
    logits = mx.squeeze(u, -1)
    
    if config['use_tanh']:
        logits = config['C'] * mx.tanh(logits)
    return e, logits


class LSTMParams(TypedDict):
    Wih: Array
    bih: Array
    Whh: Array
    bhh: Array

class LSTMConfig(TypedDict):
    hidden_dim: int

def lstmnew(input_dim: int, hidden_dim: int) -> tuple[LSTMParams, LSTMConfig]:
    scale = 1.0 / math.sqrt(hidden_dim)
    params = LSTMParams(
        Wih=mx.random.uniform(-scale, scale, shape=(input_dim, 4 * hidden_dim)),
        bih=mx.random.uniform(-scale, scale, shape=(4 * hidden_dim,)),
        Whh=mx.random.uniform(-scale, scale, shape=(hidden_dim, 4 * hidden_dim)),
        bhh=mx.random.uniform(-scale, scale, shape=(4 * hidden_dim,))
    )
    return params, LSTMConfig(hidden_dim=hidden_dim)

def lstmcellfwd(params: LSTMParams, x: Array, state: tuple[Array, Array]) -> tuple[Array, Array]:
    h, c = state
    gates = (x @ params['Wih'] + params['bih']) + (h @ params['Whh'] + params['bhh'])
    chunks = mx.split(gates, 4, axis=-1)
    i, f, g, o = mx.sigmoid(chunks[0]), mx.sigmoid(chunks[1]), mx.tanh(chunks[2]), mx.sigmoid(chunks[3])
    c_next = (f * c) + (i * g)
    h_next = o * mx.tanh(c_next)
    return h_next, c_next

def lstmfwd(params: LSTMParams, config: LSTMConfig, x: Array, state0: tuple[Array, Array] | None = None):
    if x.ndim == 2: x = mx.expand_dims(x, 0)
    seq_len, batch_size, _ = x.shape
    
    if state0 is None:
        h0 = mx.zeros((batch_size, config['hidden_dim']))
        c0 = mx.zeros((batch_size, config['hidden_dim']))
        state0 = (h0, c0)
        
    h, c = state0
    hs = []
    for t in range(seq_len):
        h, c = lstmcellfwd(params, x[t], (h, c))
        hs.append(h)
    return mx.stack(hs), (h, c)


class EncoderParams(TypedDict):
    lstm: LSTMParams
    h0: Array
    c0: Array

class EncoderConfig(TypedDict):
    lstm: LSTMConfig

def encnew(embedding_dim: int, hidden_dim: int) -> tuple[EncoderParams, EncoderConfig]:
    lstm_p, lstm_c = lstmnew(embedding_dim, hidden_dim)
    scale = 1.0 / math.sqrt(hidden_dim)
    params = EncoderParams(
        lstm=lstm_p,
        h0=mx.random.uniform(-scale, scale, shape=(1, hidden_dim)),
        c0=mx.random.uniform(-scale, scale, shape=(1, hidden_dim))
    )
    return params, EncoderConfig(lstm=lstm_c)

def encfwd(params: EncoderParams, config: EncoderConfig, inputs: Array):
    batch_size = inputs.shape[1]
    h0_batch = mx.repeat(params['h0'], batch_size, axis=0)
    c0_batch = mx.repeat(params['c0'], batch_size, axis=0)
    return lstmfwd(params['lstm'], config['lstm'], inputs, (h0_batch, c0_batch))


class DecoderParams(TypedDict):
    Wi: Array
    bi: Array
    Wh: Array
    bh: Array
    pointer: AttentionParams
    glimpse: AttentionParams

class DecoderConfig(TypedDict):
    pointer: AttentionConfig
    glimpse: AttentionConfig
    max_length: int
    n_glimpses: int

def decnew(embedding_dim: int, hidden_dim: int, max_length: int, 
               tanh_exploration: float = 10.0, use_tanh: bool = True, n_glimpses: int = 1):
    scale = 1.0 / math.sqrt(hidden_dim)
    
    pt_p, pt_c = attnnew(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
    gl_p, gl_c = attnnew(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
    
    params = DecoderParams(
        Wi=mx.random.uniform(-scale, scale, shape=(embedding_dim, 4 * hidden_dim)),
        bi=mx.random.uniform(-scale, scale, shape=(4 * hidden_dim,)),
        Wh=mx.random.uniform(-scale, scale, shape=(hidden_dim, 4 * hidden_dim)),
        bh=mx.random.uniform(-scale, scale, shape=(4 * hidden_dim,)),
        pointer=pt_p, glimpse=gl_p
    )
    config = DecoderConfig(pointer=pt_c, glimpse=gl_c, max_length=max_length, n_glimpses=n_glimpses)
    return params, config

def _lstm_step_decoder(params: DecoderParams, x: Array, h: Array, c: Array):
    gates = (x @ params['Wi'] + params['bi']) + (h @ params['Wh'] + params['bh'])
    chunks = mx.split(gates, 4, axis=1)
    i, f, g, o = mx.sigmoid(chunks[0]), mx.sigmoid(chunks[1]), mx.tanh(chunks[2]), mx.sigmoid(chunks[3])
    c_next = (f * c) + (i * g)
    h_next = o * mx.tanh(c_next)
    return h_next, c_next

def decfwd(params: DecoderParams, config: DecoderConfig, decoder_input: Array, embedded_inputs: Array, 
               hidden: tuple[Array, Array], context: Array):
    h, c = hidden
    batch_size = context.shape[1]
    sourceL = context.shape[0]
    mask = mx.zeros((batch_size, sourceL), dtype=mx.bool_)
    
    outputs, selections = [], []
    curr_input = decoder_input
    
    for _ in range(config['max_length']):
        h, c = _lstm_step_decoder(params, curr_input, h, c)
        g_l = h 
        for _ in range(config['n_glimpses']):
            _, logits = attnfwd(params['glimpse'], config['glimpse'], g_l, context)
            logits = mx.where(mask, -1e9, logits)
            probs_glimpse = mx.softmax(logits, axis=1)
            ctx_perm = mx.transpose(context, (1, 0, 2))
            g_l = mx.squeeze(mx.expand_dims(probs_glimpse, 1) @ ctx_perm, axis=1)

        _, logits = attnfwd(params['pointer'], config['pointer'], g_l, context)
        logits = mx.where(mask, -1e9, logits)
        probs = mx.softmax(logits, axis=1)
        
        selected_idxs = mx.random.categorical(logits)
        
        indices = mx.arange(sourceL)
        selected_expanded = mx.expand_dims(selected_idxs, -1)
        new_mask = (indices == selected_expanded)
        mask = mx.logical_or(mask, new_mask)
        
        emb_perm = mx.transpose(embedded_inputs, (1, 0, 2))
        selected_idxs_expanded = mx.expand_dims(mx.expand_dims(selected_idxs, -1), -1)
        next_input = mx.squeeze(mx.take_along_axis(emb_perm, selected_idxs_expanded, axis=1), axis=1)
        
        curr_input = next_input
        outputs.append(probs)
        selections.append(selected_idxs)
    return outputs, selections


class PtrNetParams(TypedDict):
    encoder: EncoderParams
    decoder: DecoderParams
    decoder_in_0: Array

class PtrNetConfig(TypedDict):
    encoder: EncoderConfig
    decoder: DecoderConfig

def ptrnetnew(embedding_dim: int, hidden_dim: int, max_decoding_len: int,
                      n_glimpses: int = 1, tanh_exploration: float = 10.0, use_tanh: bool = True):
    enc_p, enc_c = encnew(embedding_dim, hidden_dim)
    dec_p, dec_c = decnew(embedding_dim, hidden_dim, max_decoding_len, tanh_exploration, use_tanh, n_glimpses)
    scale = 1.0 / math.sqrt(embedding_dim)
    params = PtrNetParams(encoder=enc_p, decoder=dec_p, decoder_in_0=mx.random.uniform(-scale, scale, shape=(embedding_dim,)))
    config = PtrNetConfig(encoder=enc_c, decoder=dec_c)
    return params, config

def ptrnetfwd(params: PtrNetParams, config: PtrNetConfig, inputs: Array):
    batch_size = inputs.shape[1]
    enc_h, (hn, cn) = encfwd(params['encoder'], config['encoder'], inputs)
    decoder_input = mx.repeat(mx.expand_dims(params['decoder_in_0'], 0), batch_size, axis=0)
    probs, idxs = decfwd(params['decoder'], config['decoder'], decoder_input, inputs, (hn, cn), enc_h)
    return probs, idxs


class NCOParams(TypedDict):
    actor: PtrNetParams
    W_embed: Array
    baseline: Array
    alpha: Array 

class NCOConfig(TypedDict):
    actor: PtrNetConfig

def nconew(input_dim: int, embedding_dim: int, hidden_dim: int, max_decoding_len: int,
           n_glimpses: int = 1, tanh_exploration: float = 10.0, use_tanh: bool = True, alpha: float = 0.9):
    actor_p, actor_c = ptrnetnew(embedding_dim, hidden_dim, max_decoding_len, n_glimpses, tanh_exploration, use_tanh)
    scale = 1.0 / math.sqrt(embedding_dim)
    params = NCOParams(
        actor=actor_p,
        W_embed=mx.random.uniform(-scale, scale, shape=(input_dim, embedding_dim)),
        baseline=mx.array(0.0),
        alpha=mx.array(alpha) 
    )
    config = NCOConfig(actor=actor_c)
    return params, config

def ncofwd(params: NCOParams, config: NCOConfig, inputs: Array, objective_fn: callable, is_train: bool = True):
    inputs_transposed = mx.transpose(inputs, (0, 2, 1))
    embedded_inputs_batch = inputs_transposed @ params['W_embed']
    embedded_inputs_pnet = mx.transpose(embedded_inputs_batch, (1, 0, 2))
    
    probs_list, idxs_list = ptrnetfwd(params['actor'], config['actor'], embedded_inputs_pnet)
    
    actions = []
    for idxs in idxs_list:
        gather_idxs = mx.expand_dims(mx.expand_dims(idxs, -1), -1)
        selected = mx.squeeze(mx.take_along_axis(inputs_transposed, gather_idxs, axis=1), axis=1)
        actions.append(selected)

    reward = objective_fn(actions)
    
    final_probs = []
    if is_train:
        for prob_dist, idx in zip(probs_list, idxs_list):
            idx_exp = mx.expand_dims(idx, -1)
            selected_prob = mx.squeeze(mx.take_along_axis(prob_dist, idx_exp, axis=1), axis=1)
            final_probs.append(selected_prob)
    else:
        final_probs = probs_list

    return reward, final_probs, actions, idxs_list

# ---------------------------------- Training ----------------------------------

def train(train_data, n_epochs: int = 2):
    task = 'tsp'
    n_train_samples = train_data.shape[0]
    n_cities = train_data.shape[2]
    print(f"Using provided data: {n_train_samples} samples, {n_cities} cities.")

    batch_size = 32
    learning_rate = 1e-4
    embed_dim = 128
    hidden_dim = 128
    n_glimpses = 1
    
    print(f"Starting training for {task}_{n_cities} using SGD...")
    
    params, config = nconew(
        input_dim=2,
        embedding_dim=embed_dim,
        hidden_dim=hidden_dim,
        max_decoding_len=n_cities,
        n_glimpses=n_glimpses,
        tanh_exploration=10.0,
        use_tanh=True,
        alpha=0.99
    )
    
    optimizer = sgdnew(lr=learning_rate)
    
    def loss_fn(params, inputs):
        def tour_length(actions):
            dist = mx.zeros((inputs.shape[0],))
            for i in range(len(actions) - 1):
                d = mx.linalg.norm(actions[i] - actions[i+1], axis=1)
                dist = dist + d
            d_last = mx.linalg.norm(actions[-1] - actions[0], axis=1)
            dist = dist + d_last
            return dist

        tour_lens, probs, _, _ = ncofwd(params, config, inputs, tour_length, is_train=True)
        
        baseline = params['baseline']
        advantage = mx.stop_gradient(tour_lens - baseline)
        
        log_prob_sum = mx.zeros((inputs.shape[0],))
        for p in probs:
            p = mx.maximum(p, 1e-8) # prevent NaN
            log_prob_sum = log_prob_sum + mx.log(p)
            
        loss = mx.mean(advantage * log_prob_sum)
        return loss, tour_lens

    loss_and_grad_fn = mx.value_and_grad(loss_fn)

    @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
    def step(params, optimizer, inputs):
        (loss, tour_lens), grads = loss_and_grad_fn(params, inputs)
        
        grads = gradclip(grads, max_norm=1.0)
        
        new_params = sgdupdate(optimizer, params, grads)
        
        # baseline update 
        mean_reward = mx.mean(tour_lens)
        alpha = new_params['alpha']
        current_baseline = new_params['baseline']
        new_baseline = (current_baseline * alpha) + (mean_reward * (1.0 - alpha))
        new_params['baseline'] = new_baseline
        
        return new_params, loss, mean_reward

    X_train = train_data.astype(np.float32)

    tic = time.time() 
    for epoch in range(n_epochs):
        print(f"\n--- Epoch {epoch + 1}/{n_epochs} ---")
        indices = np.random.permutation(n_train_samples)
        X_shuffled = X_train[indices]
        n_batches = n_train_samples // batch_size
        pbar = tqdm(range(n_batches))
        
        epoch_loss = 0.0
        epoch_len = 0.0
        
        for i in pbar:
            batch_data = X_shuffled[i*batch_size : (i+1)*batch_size]
            batch_mx = mx.array(batch_data)
            
            params, step_loss, step_len = step(params, optimizer, batch_mx)
            
            mx.eval(step_loss, step_len)
            l_val = step_loss.item()
            r_val = step_len.item()
            epoch_loss += l_val
            epoch_len += r_val
            pbar.set_description(f"Loss: {l_val:.4f} | Avg Tour: {r_val:.4f}")
            
        print(f"Epoch {epoch + 1} Stats -> Loss: {epoch_loss/n_batches:.4f} | Avg Tour Length: {epoch_len/n_batches:.4f}")
        print(f"Baseline: {params['baseline'].item():.4f}")

    toc = time.time()
    print("Training Finished.")
    print(f"Training time (s): {toc - tic}")
    return params

if __name__ == "__main__":
    np.random.seed(123)
    mx.random.seed(123)

    n_cities = input('Please enter how many cities to train on (5 or 10): ').strip()
    if n_cities == '5':
        file = TSP5_TRAIN_FILE
    elif n_cities == '10':
        file = TSP10_TRAIN_FILE
    else:
        raise NotImplementedError(f"Training on {n_cities} is not supported")

    X, Y = load_tsp(file)
    train(train_data=X)