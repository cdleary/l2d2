import collections
import math
import csv
import os
import random
import struct
import sys
from datetime import datetime
from typing import List, Tuple

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
INPUT_LEN = 256         # Vector dimension input to recurrent structure.
CARRY_LEN = (           # Vector dimension carried to next recurrent structure.
    INPUT_LEN-INPUT_FLOATS_PER_BYTE)
EVAL_MINIBATCHES = 1024 # Number of minibatches for during-training eval.
STEP_SIZE=1e-3          # Learning rate.
FC = 1024               # Fully connected neurons to use on last hidden output.
MINIBATCH_SIZE = 32     # Number of samples per minibatch.


@jax.jit
def step(xs, p):
    assert xs.shape == (MINIBATCH_SIZE, INPUT_FLOATS), xs.shape
    h = jnp.zeros((MINIBATCH_SIZE, CARRY_LEN))
    for byteno, i in enumerate(range(0, BYTES*INPUT_FLOATS_PER_BYTE, INPUT_FLOATS_PER_BYTE)):
        w, b = p[byteno]
        x = xs[:,i:i+INPUT_FLOATS_PER_BYTE]
        assert x.shape == (MINIBATCH_SIZE, INPUT_FLOATS_PER_BYTE), x.shape
        x = jnp.concatenate((x, h), axis=-1)
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


def value_byte_to_floats(b):
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
    return jnp.array(pieces)


def value_to_sample(bs):
    return jnp.concatenate(tuple(value_byte_to_floats(b) for b in bs))


def loss(p, sample, target):
    assert sample.shape == (MINIBATCH_SIZE, INPUT_FLOATS)
    assert target.shape == (MINIBATCH_SIZE,), target.shape
    predictions = step(sample, p)
    labels = jax.nn.one_hot(target, CLASSES, dtype='float32')
    return jnp.sum((labels - predictions)**2)


opt_init, opt_update, get_params = optimizers.adagrad(step_size=STEP_SIZE)
#opt_init, opt_update, get_params = optimizers.sgd(step_size=STEP_SIZE)


@jax.jit
def train_step(i, opt_state, sample, target):
    assert sample.shape == (MINIBATCH_SIZE, INPUT_FLOATS)
    assert target.shape == (MINIBATCH_SIZE,), target
    params = get_params(opt_state)
    g = jax.grad(loss)(params, sample, target)
    return opt_update(i, g, opt_state)


def read_records(limit=False):
    d = []
    with open(os.path.expanduser('~/x86_3B.csv')) as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            d.append((int(row[0].replace(' ', ''), 16), int(row[1])))
            if limit and len(d) == MINIBATCH_SIZE:
                break
    print(f'records read: {len(d):,}')
    return d


def run_eval(p, eval_minibatches: Tuple):
    confusion = collections.defaultdict(lambda: 0)
    for samples, wants in eval_minibatches:
        #print('samples:', samples)
        #print('  wants:', wants)
        probs = step(samples_to_input(samples), p)
        #print('probs:', probs)
        gots = probs.argmax(axis=-1)
        for i in range(MINIBATCH_SIZE):
            confusion[(wants[i], gots[i].item())] += 1


    for want in range(CLASSES):
        print('want', want, ': ', end='')
        for got in range(CLASSES):
            print('{:3d}'.format(confusion[(want, got)]), end=' ')
        print()

    accuracy = sum(confusion[(i, i)] for i in range(CLASSES)) / float(sum(confusion.values())) * 100.0
    print(f'accuracy: {accuracy:.2f}%')


def random_minibatch_iter(d, count):
    for _ in range(count):
        sampled = random.sample(d, MINIBATCH_SIZE)
        samples = jnp.array([value_to_sample(s) for s, _ in sampled])
        targets = jnp.array([t for _, t in sampled])
        yield samples, targets


def minibatch_iter(d):
    samples = []
    targets = []
    for in_bytes, target in d:
        samples.append(value_to_sample(in_bytes))
        targets.append(target)
        if len(samples) == MINIBATCH_SIZE:
            yield jnp.array(samples), jnp.array(targets)
            samples.clear()
            targets.clear()
    if samples:
        yield jnp.array(samples), jnp.array(targets)


def time_train_step(opt_state):
    fake_sample = jnp.zeros((MINIBATCH_SIZE, INPUT_FLOATS), dtype='float32')
    fake_target = jnp.zeros((MINIBATCH_SIZE,), dtype='int8')

    # Warmup.
    train_step(0, opt_state, fake_sample, fake_target)

    # Timed.
    start = datetime.now()
    train_step(0, opt_state, fake_sample, fake_target)
    end = datetime.now()
    return end - start


def samples_to_input(samples: List[List[int]]):
    return jnp.array([value_to_sample(sample) for sample in samples])


def run_train():
    train_start = datetime.now()

    # Initialize parameters.
    key = rng.PRNGKey(0)
    p = [(rng.normal(key, (INPUT_LEN, CARRY_LEN)),
          rng.normal(key, (CARRY_LEN,)))
         for _ in range(BYTES)]
    p.append(rng.normal(key, (CARRY_LEN, FC)))
    p.append(rng.normal(key, (FC, CLASSES)))

    # Package up in optimizer state.
    opt_state = opt_init(p)

    #step_time = time_train_step(opt_state)
    #steps_per_sec = 1.0/step_time.total_seconds()
    #samples_per_sec = steps_per_sec * MINIBATCH_SIZE
    #print(f'step time approximately: {step_time}; {steps_per_sec:.1f} steps/s; {samples_per_sec:.1f} samples/s')

    data = ingest.load_state('/tmp/x86.pickle')

    eval_sampler = ingest_sampler.sample_minibatches(data, MINIBATCH_SIZE, BYTES, zeros_ok=False, replacement=True)
    eval_data = tuple(next(eval_sampler) for _ in range(EVAL_MINIBATCHES))

    sampler = ingest_sampler.sample_minibatches(data, MINIBATCH_SIZE, BYTES, zeros_ok=False, replacement=False)

    for epoch in range(1):
        print('... epoch start', epoch)
        for i, minibatch in enumerate(sampler):
            if i % 64 == 63:
                now = datetime.now()
                print('... epoch', epoch, 'step', i, '@', now, '@ {:.2f} step/s'.format(i / (now-train_start).total_seconds()))
                run_eval(get_params(opt_state), eval_data)
            samples, targets = minibatch
            assert len(targets) == MINIBATCH_SIZE, targets
            assert len(samples) == MINIBATCH_SIZE
            samples = samples_to_input(samples)
            opt_state = train_step(i, opt_state, samples, jnp.array(targets, dtype='uint8'))
        p = get_params(opt_state)
        run_eval(p, eval_data)

    train_end = datetime.now()

    print('train time:', train_end - train_start)


def main():
    run_train()

if __name__ == '__main__':
    main()
