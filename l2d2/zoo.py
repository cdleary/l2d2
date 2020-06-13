import datetime
import optparse

import jax
from jax import random as rng
from jax.experimental import optimizers
from jax import numpy as jnp
from jax.nn.initializers import glorot_normal, normal

import options
import preprocess


INPUT_BYTES = 8         # Number of input bytes (upper limit).
_FC = 128               # Fully connected neurons to use on last hidden output.
CLASSES = INPUT_BYTES+1 # Length categories; 0 for "not a valid instruction".

# We turn the input bytes into some number of input floats.
INPUT_FLOATS_PER_BYTE = 8
INPUT_FLOATS = (        # Bytes are turned into a number of floats.
    INPUT_BYTES * INPUT_FLOATS_PER_BYTE)
DROPOUT_RATE = 0.5


@jax.partial(jax.jit, static_argnums=1)
def init_params(key, carry_len: int):
    w_init = glorot_normal()
    b_init = normal()
    p = []
    key, w_subkey = rng.split(key)
    key, b_subkey = rng.split(key)
    t = (w_init(w_subkey, (INPUT_FLOATS, _FC), dtype='float32'),
         b_init(b_subkey, (_FC,), dtype='float32'))
    p.append(t)

    key, w_subkey = rng.split(key)
    p.append(w_init(w_subkey, (_FC, CLASSES), dtype='float32'))
    return p


@jax.partial(jax.jit, static_argnums=(3, 4))
def step(xs, p, rng_key, carry_len: int, train: bool):
    #hdot = lambda lhs, rhs: jax.lax.dot(lhs.astype('float16'), rhs.astype('float16')).astype('float32')
    hdot = jax.lax.dot

    z = jax.nn.relu(xs @ p[0][0] + p[0][1])
    sm = jax.nn.softmax(z @ p[1])
    return sm


@jax.partial(jax.jit, static_argnums=(4, 5))
def loss_and_preds(p, sample, target, rng_key, carry_len: int, train: bool):
    batch_size = sample.shape[0]
    assert sample.shape == (batch_size, INPUT_FLOATS), sample.shape
    assert target.shape == (batch_size,), target.shape
    predictions = step(sample, p, rng_key, carry_len, train)
    labels = jax.nn.one_hot(target, CLASSES, dtype='float32')
    l = jnp.mean((predictions-labels)**2)
    return l, predictions


@jax.partial(jax.jit, static_argnums=(4, 5))
def loss(p, sample, target, rng, carry_len: int, train: bool):
    return loss_and_preds(p, sample, target, rng, carry_len, train)[0]


@jax.partial(jax.jit, static_argnums=(5, 6, 7))
def train_step(i, opt_state, sample, target, rng_key, carry_len: int,
               get_params, opt_update):
    batch_size = sample.shape[0]
    assert sample.shape == (batch_size, INPUT_FLOATS), (sample.shape, INPUT_FLOATS)
    assert target.shape == (batch_size,), target.shape
    params = get_params(opt_state)
    l, g = jax.value_and_grad(loss)(params, sample, target, rng_key, carry_len,
                                    True)
    return opt_update(i, g, opt_state), l


def time_train_step(batch_size: int, carry_len: int,
                    opt_state, get_params, opt_update) -> datetime.timedelta:
    """Times a single training step after warmup."""
    fake_sample = jnp.zeros((batch_size, INPUT_FLOATS), dtype='float32')
    fake_target = jnp.zeros((batch_size,), dtype='int32')
    rng_key = rng.PRNGKey(0)

    # Forward warmup.
    loss(get_params(opt_state), fake_sample, fake_target, rng_key,
         carry_len, False).block_until_ready()

    # Forward timed.
    start = datetime.datetime.now()
    loss(get_params(opt_state), fake_sample, fake_target, rng_key,
         carry_len, False).block_until_ready()
    end = datetime.datetime.now()
    fwd_time = end - start

    # Train warmup.
    opt_state, _loss = train_step(
        0, opt_state, fake_sample, fake_target, rng_key, carry_len, get_params,
        opt_update)
    get_params(opt_state)[-1].block_until_ready()

    # Train timed.
    start = datetime.datetime.now()
    opt_state, _loss = train_step(0, opt_state, fake_sample, fake_target, rng_key,
                           carry_len, get_params, opt_update)
    get_params(opt_state)[-1].block_until_ready()
    end = datetime.datetime.now()
    step_time = end - start

    start = datetime.datetime.now()
    TO_AMORTIZE = 4096
    for i in range(TO_AMORTIZE):
        opt_state, _loss = train_step(0, opt_state, fake_sample, fake_target,
                                      rng_key, carry_len, get_params,
                                      opt_update)
    get_params(opt_state)[-1].block_until_ready()
    end = datetime.datetime.now()
    step_per_sec = float(TO_AMORTIZE)/(end-start).total_seconds()
    samples_per_sec = step_per_sec * batch_size
    print(f'amortized: {step_per_sec:.2f} steps/s => {samples_per_sec:,.2f} samples/s')

    return fwd_time, step_time


def main():
    parser = optparse.OptionParser()
    options.add_model_hparams(parser)
    opts, args = parser.parse_args()

    #opt_init, opt_update, get_params = optimizers.adagrad(step_size=opts.step_size)
    opt_init, opt_update, get_params = optimizers.sgd(step_size=opts.step_size)

    key = rng.PRNGKey(0)
    p = init_params(key, opts.carry_len)
    opt_state = opt_init(p)
    fwd_time, step_time = time_train_step(opts.batch_size, opts.carry_len,
                                          opt_state, get_params, opt_update)
    steps_per_sec = 1.0/step_time.total_seconds()
    samples_per_sec = steps_per_sec * opts.batch_size
    fwd_percent = fwd_time.total_seconds() / step_time.total_seconds() * 100.0
    fwd_us = int(fwd_time.total_seconds() * 1e6)
    step_us = int(step_time.total_seconds() * 1e6)
    print(f'bs{opts.batch_size:<8} fwd time approximately:  {fwd_us:6,} us ({fwd_percent:.2f}%)')
    print(f'bs{opts.batch_size:<8} step time approximately: {step_us:6,} us; {steps_per_sec:.1f} '
          f'steps/s; {samples_per_sec:.1f} samples/s')



if __name__ == '__main__':
    main()
