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

import ingest
import preprocess
import xtrie


INPUT_FLOATS_PER_BYTE = len(preprocess.value_byte_to_floats(0))
BYTES = 15              # Input byte count (with instruction data).
INPUT_FLOATS = (        # Bytes are turned into a number of floats.
    BYTES * INPUT_FLOATS_PER_BYTE)
CLASSES = BYTES+1       # Length categories; 0 for "not a valid instruction".
INPUT_LEN = 512         # Vector dimension input to recurrent structure.
CARRY_LEN = INPUT_LEN   # Vector dimension carried to next recurrent structure.
EVAL_MINIBATCHES = 1024 # Number of minibatches for during-training eval.
STEP_SIZE = 1e-6        # Learning rate.
FC = 4096               # Fully connected neurons to use on last hidden output.


@jax.jit
def step(xs, p):
    assert xs.shape[1] == INPUT_FLOATS, xs.shape
    batch_size = xs.shape[0]
    h = jnp.zeros((batch_size, CARRY_LEN))
    for byteno, i in enumerate(range(0, BYTES*INPUT_FLOATS_PER_BYTE, INPUT_FLOATS_PER_BYTE)):
        w, b = p[byteno]
        x = xs[:,i:i+INPUT_FLOATS_PER_BYTE]
        assert x.shape == (batch_size, INPUT_FLOATS_PER_BYTE), x.shape
        x = jnp.concatenate((x, h[:,INPUT_FLOATS_PER_BYTE:]), axis=-1)
        assert x.shape == (batch_size, INPUT_LEN), x.shape
        y = x @ w + b
        z = jnp.tanh(y)
        h = z
    assert h.shape == (batch_size, CARRY_LEN)
    z = jnp.tanh(h @ p[-2])
    assert z.shape == (batch_size, FC)
    return jax.nn.softmax(z @ p[-1])


@jax.jit
def loss(p, sample, target):
    batch_size = sample.shape[0]
    assert sample.shape == (batch_size, INPUT_FLOATS)
    assert target.shape == (batch_size,), target.shape
    predictions = step(sample, p)
    labels = jax.nn.one_hot(target, CLASSES, dtype='float32')
    return jnp.sum((labels - predictions)**2)


#opt_init, opt_update, get_params = optimizers.adagrad(step_size=STEP_SIZE)
opt_init, opt_update, get_params = optimizers.sgd(step_size=STEP_SIZE)


@jax.jit
def train_step(i, opt_state, sample, target):
    batch_size = sample.shape[0]
    assert sample.shape == (batch_size, INPUT_FLOATS)
    assert target.shape == (batch_size,), target
    params = get_params(opt_state)
    g = jax.grad(loss)(params, sample, target)
    return opt_update(i, g, opt_state)


def run_eval(p, eval_minibatches: Tuple):
    """Runs evaluation step on the given eval_minibatches with parameters p."""
    confusion = collections.defaultdict(lambda: 0)
    for samples, wants in eval_minibatches:
        probs = step(preprocess.samples_to_input(samples), p)
        gots = probs.argmax(axis=-1)
        assert isinstance(samples, list)
        for i in range(len(samples)):
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


def time_train_step(batch_size: int, opt_state) -> datetime.timedelta:
    """Times a single training step after warmup."""
    fake_sample = np.zeros((batch_size, INPUT_FLOATS), dtype='float32')
    fake_target = np.zeros((batch_size,), dtype='int32')

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


@jax.jit
def init_params(key):
    p = [(rng.normal(key, (INPUT_LEN, CARRY_LEN)),
          rng.normal(key, (CARRY_LEN,)))
         for _ in range(BYTES)]
    p.append(rng.normal(key, (CARRY_LEN, FC)))
    p.append(rng.normal(key, (FC, CLASSES)))
    return p


def _get_eval_data(data: xtrie.XTrie, batch_size: int) -> Tuple:
    data = data.clone()
    eval_sample_start = datetime.datetime.now()
    eval_data = tuple(data.sample_nr_mb(batch_size, BYTES) for _ in range(EVAL_MINIBATCHES))
    eval_sample_end = datetime.datetime.now()
    print('sampling time per minibatch: {:.3f} us'.format(
          (eval_sample_end-eval_sample_start).total_seconds()
            / EVAL_MINIBATCHES * 1e6))
    return eval_data


def run_train(epochs: int, batch_size: int, time_step_only: bool) -> None:
    """Runs a training routine."""
    # Initialize parameters.
    key = rng.PRNGKey(0)
    p = init_params(key)

    # Package up in optimizer state.
    opt_state = opt_init(p)

    fwd_time, step_time = time_train_step(batch_size, opt_state)
    steps_per_sec = 1.0/step_time.total_seconds()
    samples_per_sec = steps_per_sec * batch_size
    fwd_percent = fwd_time.total_seconds() / step_time.total_seconds() * 100.0
    fwd_us = int(fwd_time.total_seconds() * 1e6)
    step_us = int(step_time.total_seconds() * 1e6)
    print(f'bs{batch_size:<8} fwd time approximately:  {fwd_us:6,} us ({fwd_percent:.2f}%)')
    print(f'bs{batch_size:<8} step time approximately: {step_us:6,} us; {steps_per_sec:.1f} '
          f'steps/s; {samples_per_sec:.1f} samples/s')

    if time_step_only:
        return

    data = ingest.load_state('/tmp/x86.state')

    eval_data = _get_eval_data(data, batch_size)

    train_start = datetime.datetime.now()

    for epoch in range(epochs):
        print('... epoch start', epoch)
        stepno = 0
        while not data.empty():
            if stepno % (16 * batch_size) == (16 * batch_size - 1):
                now = datetime.datetime.now()
                print('... epoch', epoch, 'step', stepno, '@', now,
                      '@ {:.2f} step/s'.format(
                          stepno / (now-train_start).total_seconds()))
                run_eval(get_params(opt_state), eval_data)
            samples, targets = data.sample_nr_mb(batch_size, BYTES)
            assert len(targets) == batch_size, targets
            assert len(samples) == batch_size
            samples = preprocess.samples_to_input(samples)
            opt_state = train_step(stepno, opt_state, samples,
                                   np.array(targets, dtype='uint8'))
            stepno += 1

        # End of epoch!
        p = get_params(opt_state)
        run_eval(p, eval_data)

    train_end = datetime.datetime.now()

    print('train time:', train_end - train_start)


def main():
    parser = optparse.OptionParser()
    parser.add_option('--time-step-only', action='store_true', default=False,
                      help='Just time a training step, do not train.')
    parser.add_option('--batch-size', type=int, default=8,
                      help='minibatch size')
    parser.add_option('--epochs', type=int, default=8,
                      help='number of iterations through the training data before completing')
    opts, args = parser.parse_args()
    run_train(opts.epochs, opts.batch_size, opts.time_step_only)


if __name__ == '__main__':
    main()
