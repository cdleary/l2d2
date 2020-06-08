import datetime

import jax
from jax import random as rng
from jax.experimental import optimizers
from jax import numpy as jnp

import preprocess


INPUT_BYTES = 15  # Number of input bytes (upper limit).
_FC = 4096  # Fully connected neurons to use on last hidden output.
CLASSES = INPUT_BYTES+1 # Length categories; 0 for "not a valid instruction".

# We turn the input bytes into some number of input floats.
VALUE_OPTS = preprocess.ValueOpts(byte=True, nibbles=True, crumbs=True, bits=True)
INPUT_FLOATS_PER_BYTE = len(preprocess.value_byte_to_floats(0, VALUE_OPTS))
BYTES = 15              # Input byte count (with instruction data).
INPUT_FLOATS = (        # Bytes are turned into a number of floats.
    BYTES * INPUT_FLOATS_PER_BYTE)


@jax.partial(jax.jit, static_argnums=1)
def init_params(key, carry_len: int):
    p = [(rng.normal(key, (carry_len, carry_len), dtype='float32'),
          rng.normal(key, (carry_len,), dtype='float32'))
         for _ in range(INPUT_BYTES)]
    p.append((
        rng.normal(key, (carry_len, _FC), dtype='float32'),
        rng.normal(key, (_FC,), dtype='float32'),
    ))
    p.append(rng.normal(key, (_FC, CLASSES), dtype='float32'))
    return p


@jax.partial(jax.jit, static_argnums=2)
def step(xs, p, carry_len: int):
    #hdot = lambda lhs, rhs: jax.lax.dot(lhs.astype('float16'), rhs.astype('float16')).astype('float32')
    hdot = jax.lax.dot

    assert xs.shape[1] == INPUT_FLOATS, xs.shape
    assert xs.dtype == 'float32'
    batch_size = xs.shape[0]
    h = jnp.zeros((batch_size, carry_len), dtype='float32')
    for byteno, i in enumerate(range(0, INPUT_BYTES*INPUT_FLOATS_PER_BYTE, INPUT_FLOATS_PER_BYTE)):
        w, b = p[byteno]
        x = xs[:,i:i+INPUT_FLOATS_PER_BYTE]
        assert x.dtype == 'float32'
        assert x.shape == (batch_size, INPUT_FLOATS_PER_BYTE), x.shape
        x = jnp.concatenate((x, h[:,INPUT_FLOATS_PER_BYTE:]), axis=-1)
        assert x.dtype == 'float32'
        assert x.shape == (batch_size, carry_len), x.shape
        y = hdot(x, w) + b
        z = jnp.tanh(y)
        h = z
    assert h.shape == (batch_size, carry_len)
    z = jnp.tanh(hdot(h, p[-2][0]) + p[-2][1])
    assert z.shape == (batch_size, _FC)
    return jax.nn.softmax(hdot(z, p[-1]))


@jax.partial(jax.jit, static_argnums=3)
def loss(p, sample, target, carry_len: int):
    batch_size = sample.shape[0]
    assert sample.shape == (batch_size, INPUT_FLOATS)
    assert target.shape == (batch_size,), target.shape
    predictions = step(sample, p, carry_len)
    labels = jax.nn.one_hot(target, CLASSES, dtype='float32')
    return jnp.sum((labels - predictions)**2)


@jax.partial(jax.jit, static_argnums=(4, 5, 6))
def train_step(i, opt_state, sample, target, carry_len: int, get_params, opt_update):
    batch_size = sample.shape[0]
    assert sample.shape == (batch_size, INPUT_FLOATS)
    assert target.shape == (batch_size,), target
    params = get_params(opt_state)
    g = jax.grad(loss)(params, sample, target, carry_len)
    return opt_update(i, g, opt_state)


def time_train_step(batch_size: int, carry_len: int, opt_state, get_params, opt_update) -> datetime.timedelta:
    """Times a single training step after warmup."""
    fake_sample = jnp.zeros((batch_size, INPUT_FLOATS), dtype='float32')
    fake_target = jnp.zeros((batch_size,), dtype='int32')

    # Forward warmup.
    loss(get_params(opt_state), fake_sample, fake_target, carry_len).block_until_ready()

    # Forward timed.
    start = datetime.datetime.now()
    loss(get_params(opt_state), fake_sample, fake_target, carry_len).block_until_ready()
    end = datetime.datetime.now()
    fwd_time = end - start

    # Train warmup.
    opt_state = train_step(0, opt_state, fake_sample, fake_target, carry_len, get_params, opt_update)
    get_params(opt_state)[-1].block_until_ready()

    # Train timed.
    start = datetime.datetime.now()
    opt_state = train_step(0, opt_state, fake_sample, fake_target, carry_len, get_params, opt_update)
    get_params(opt_state)[-1].block_until_ready()
    end = datetime.datetime.now()
    step_time = end - start

    start = datetime.datetime.now()
    TO_AMORTIZE = 4096
    for i in range(TO_AMORTIZE):
        opt_state = train_step(0, opt_state, fake_sample, fake_target, carry_len, get_params, opt_update)
    get_params(opt_state)[-1].block_until_ready()
    end = datetime.datetime.now()
    step_per_sec = float(TO_AMORTIZE)/(end-start).total_seconds()
    samples_per_sec = step_per_sec * batch_size
    print(f'amortized: {step_per_sec:.2f} steps/s => {samples_per_sec:,.2f} samples/s')

    return fwd_time, step_time


def main():
    STEP_SIZE = 1e-3
    CARRY_LEN = 1024
    BATCH_SIZE = 128
    opt_init, opt_update, get_params = optimizers.adagrad(step_size=STEP_SIZE)
    #opt_init, opt_update, get_params = optimizers.sgd(step_size=STEP_SIZE)

    key = rng.PRNGKey(0)
    p = init_params(key, CARRY_LEN)
    opt_state = opt_init(p)
    fwd_time, step_time = time_train_step(BATCH_SIZE, CARRY_LEN, opt_state, get_params, opt_update)
    steps_per_sec = 1.0/step_time.total_seconds()
    samples_per_sec = steps_per_sec * BATCH_SIZE
    fwd_percent = fwd_time.total_seconds() / step_time.total_seconds() * 100.0
    fwd_us = int(fwd_time.total_seconds() * 1e6)
    step_us = int(step_time.total_seconds() * 1e6)
    print(f'bs{BATCH_SIZE:<8} fwd time approximately:  {fwd_us:6,} us ({fwd_percent:.2f}%)')
    print(f'bs{BATCH_SIZE:<8} step time approximately: {step_us:6,} us; {steps_per_sec:.1f} '
          f'steps/s; {samples_per_sec:.1f} samples/s')



if __name__ == '__main__':
    main()
