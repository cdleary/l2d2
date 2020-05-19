import collections
import datetime
import os
import optparse
import sys
from typing import List, Tuple

import numpy as np

import jax
from jax.experimental import optimizers
from jax import numpy as jnp
from jax import random as rng

import float_helpers
import ingest
import ingest_sampler

BYTES = 7               # Input byte count (with instruction data).
INPUT_FLOATS_PER_BYTE = 15
INPUT_FLOATS = (        # Bytes are turned into a number of floats.
    BYTES * INPUT_FLOATS_PER_BYTE)
CLASSES = BYTES+1       # Length categories; 0 for "not a valid instruction".
INPUT_LEN = 512         # Vector dimension input to recurrent structure.
CARRY_LEN = INPUT_LEN   # Vector dimension carried to next recurrent structure.
EVAL_MINIBATCHES = 1024 # Number of minibatches for during-training eval.
STEP_SIZE = 1e-5        # Learning rate.
FC = 1024               # Fully connected neurons to use on last hidden output.
MINIBATCH_SIZE = 128    # Number of samples per minibatch.


@jax.jit
def step(xs, p):
    assert xs.shape == (MINIBATCH_SIZE, INPUT_FLOATS), xs.shape
    h = jnp.zeros((MINIBATCH_SIZE, CARRY_LEN))
    for byteno, i in enumerate(range(0, BYTES*INPUT_FLOATS_PER_BYTE, INPUT_FLOATS_PER_BYTE)):
        w, b = p[byteno]
        x = xs[:,i:i+INPUT_FLOATS_PER_BYTE]
        assert x.shape == (MINIBATCH_SIZE, INPUT_FLOATS_PER_BYTE), x.shape
        x = jnp.concatenate((x, h[:,INPUT_FLOATS_PER_BYTE:]), axis=-1)
        assert x.shape == (MINIBATCH_SIZE, INPUT_LEN), x.shape
        y = x @ w + b
        z = jnp.tanh(y)
        h = z
    assert h.shape == (MINIBATCH_SIZE, CARRY_LEN)
    z = jnp.tanh(h @ p[-2])
    assert z.shape == (MINIBATCH_SIZE, FC)
    return jax.nn.softmax(z @ p[-1])


def byte_to_float(x):
    assert x & 0xff == x, x
    return float_helpers.parts2float((0, float_helpers.BIAS, x << (23-8)))


def crumb(b: int, index: int) -> int:
    return (b >> (2 * index)) & 3


def bit(b: int, index: int) -> int:
    return (b >> index) & 1


def value_byte_to_floats(b: int) -> np.array:
    """Constructs a vector that characterizes a byte's content in various ways.

    We create:

        [byte, hi_nibble, lo_nibble, crumb0, crumb1, crumb2, crumb3] +
        [bit0, bit1, ..., bit8]

    Where the bits contents are placed in a float via byte_to_float.
    """
    pieces = [
        byte_to_float(b),               # value for whole byte
        byte_to_float((b >> 4) & 0xf),  # upper nibble
        byte_to_float(b & 0x0f),        # lower nibble
        byte_to_float(crumb(b, 0)),
        byte_to_float(crumb(b, 1)),
        byte_to_float(crumb(b, 2)),
        byte_to_float(crumb(b, 3)),
    ]
    for i in range(8):
        pieces.append(byte_to_float(bit(b, i)))
    return np.array(pieces)


def value_to_sample(bs: List[int]) -> np.array:
    """Creates an input vector sample from the sequence of bytes."""
    assert len(bs) == BYTES
    return np.concatenate(tuple(value_byte_to_floats(b) for b in bs))


@jax.jit
def loss(p, sample, target):
    assert sample.shape == (MINIBATCH_SIZE, INPUT_FLOATS)
    assert target.shape == (MINIBATCH_SIZE,), target.shape
    predictions = step(sample, p)
    labels = jax.nn.one_hot(target, CLASSES, dtype='float32')
    return jnp.sum((labels - predictions)**2)


#opt_init, opt_update, get_params = optimizers.adagrad(step_size=STEP_SIZE)
opt_init, opt_update, get_params = optimizers.sgd(step_size=STEP_SIZE)


@jax.jit
def train_step(i, opt_state, sample, target):
    assert sample.shape == (MINIBATCH_SIZE, INPUT_FLOATS)
    assert target.shape == (MINIBATCH_SIZE,), target
    params = get_params(opt_state)
    g = jax.grad(loss)(params, sample, target)
    return opt_update(i, g, opt_state)


def run_eval(p, eval_minibatches: Tuple):
    """Runs evaluation step on the given eval_minibatches with parameters p."""
    confusion = collections.defaultdict(lambda: 0)
    for samples, wants in eval_minibatches:
        probs = step(samples_to_input(samples), p)
        gots = probs.argmax(axis=-1)
        for i in range(MINIBATCH_SIZE):
            confusion[(wants[i], gots[i].item())] += 1

    # Print out the confusion matrix.
    for want in range(CLASSES):
        print('want', want, ': ', end='')
        for got in range(CLASSES):
            print('{:3d}'.format(confusion[(want, got)]), end=' ')
        print()

    # Print out summary accuracy statistic(s).
    accuracy = sum(
            confusion[(i, i)]
            for i in range(CLASSES)) / float(sum(confusion.values())) * 100.0
    print(f'accuracy: {accuracy:.2f}%')


def time_train_step(opt_state) -> datetime.timedelta:
    """Times a single training step after warmup."""
    fake_sample = np.zeros((MINIBATCH_SIZE, INPUT_FLOATS), dtype='float32')
    fake_target = np.zeros((MINIBATCH_SIZE,), dtype='int32')

    # Forward warmup.
    loss(get_params(opt_state), fake_sample, fake_target).block_until_ready()

    # Forward timed.
    start = datetime.datetime.now()
    loss(get_params(opt_state), fake_sample, fake_target).block_until_ready()
    end = datetime.datetime.now()
    fwd_time = end - start

    # Train warmup.
    opt_state = train_step(0, opt_state, fake_sample, fake_target)
    get_params(opt_state)[-1].block_until_ready()

    # Train timed.
    start = datetime.datetime.now()
    opt_state = train_step(0, opt_state, fake_sample, fake_target)
    get_params(opt_state)[-1].block_until_ready()
    end = datetime.datetime.now()
    step_time = end - start
    return fwd_time, step_time


def samples_to_input(samples: List[List[int]]) -> np.array:
    """Converts a sequence of bytes-samples into a minibatch array."""
    return np.array([value_to_sample(sample) for sample in samples])


@jax.jit
def init_params(key):
    p = [(rng.normal(key, (INPUT_LEN, CARRY_LEN)),
          rng.normal(key, (CARRY_LEN,)))
         for _ in range(BYTES)]
    p.append(rng.normal(key, (CARRY_LEN, FC)))
    p.append(rng.normal(key, (FC, CLASSES)))
    return p


def run_train(time_step_only) -> None:
    """Runs a training routine."""
    # Initialize parameters.
    key = rng.PRNGKey(0)
    p = init_params(key)

    # Package up in optimizer state.
    opt_state = opt_init(p)

    fwd_time, step_time = time_train_step(opt_state)
    steps_per_sec = 1.0/step_time.total_seconds()
    samples_per_sec = steps_per_sec * MINIBATCH_SIZE
    fwd_percent = fwd_time.total_seconds() / step_time.total_seconds() * 100.0
    fwd_us = int(fwd_time.total_seconds() * 1e6)
    step_us = int(step_time.total_seconds() * 1e6)
    print(f'fwd time approximately:  {fwd_us:6,} us ({fwd_percent:.2f}%)')
    print(f'step time approximately: {step_us:6,} us; {steps_per_sec:.1f} '
          f'steps/s; {samples_per_sec:.1f} samples/s')

    if time_step_only:
        return

    data = ingest.load_state('/tmp/x86.pickle')

    eval_sampler = ingest_sampler.sample_minibatches(
            data, MINIBATCH_SIZE, BYTES, zeros_ok=False, replacement=True)
    eval_sample_start = datetime.datetime.now()
    eval_data = tuple(next(eval_sampler) for _ in range(EVAL_MINIBATCHES))
    eval_sample_end = datetime.datetime.now()
    print('sampling time per minibatch: {:.3f} ms'.format(
          (eval_sample_end-eval_sample_start).total_seconds()
            / EVAL_MINIBATCHES * 1e3))


    sampler = ingest_sampler.sample_minibatches(
            data, MINIBATCH_SIZE, BYTES, zeros_ok=False, replacement=False)

    train_start = datetime.datetime.now()

    for epoch in range(1):
        print('... epoch start', epoch)
        for i, minibatch in enumerate(sampler):
            if i % 64 == 63:
                now = datetime.datetime.now()
                print('... epoch', epoch, 'step', i, '@', now,
                      '@ {:.2f} step/s'.format(
                          i / (now-train_start).total_seconds()))
                run_eval(get_params(opt_state), eval_data)
            samples, targets = minibatch
            assert len(targets) == MINIBATCH_SIZE, targets
            assert len(samples) == MINIBATCH_SIZE
            samples = samples_to_input(samples)
            opt_state = train_step(i, opt_state, samples,
                                   np.array(targets, dtype='uint8'))
        p = get_params(opt_state)
        run_eval(p, eval_data)

    train_end = datetime.datetime.now()

    print('train time:', train_end - train_start)


def main():
    parser = optparse.OptionParser()
    parser.add_option('--time-step-only', action='store_true', default=False,
                      help='Just time a training step, do not train.')
    opts, args = parser.parse_args()
    run_train(opts.time_step_only)

if __name__ == '__main__':
    main()
