import collections
import math
import csv
import os
import random
import struct
import sys
from datetime import datetime

import jax
from jax.experimental import optimizers
from jax import numpy as jnp
from jax import random as rng

import float_helpers

BYTES = 3
CLASSES = 4
INPUT_LEN = 1024
CARRY_LEN = INPUT_LEN-1
EVAL_SAMPLES = 1024
STEP_SIZE=1e-6
FC = 1024


H0 = jnp.zeros((CARRY_LEN,))


@jax.jit
def step(xs, p):
    h = H0
    for i in range(BYTES):
        w, b = p[i]
        x = xs[i:i+1]
        x = jnp.concatenate((h, x))
        y = x @ w + b
        z = jnp.tanh(y)
        h = z
    z = jnp.tanh(h @ p[-2])
    return jax.nn.softmax(z @ p[-1])


def byte_to_float(x):
    assert x & 0xff == x, x
    return float_helpers.parts2float((0, float_helpers.BIAS, x << (23-8)))

def value_to_sample(x):
    result = jnp.array([
        byte_to_float((x >> 16) & 0xff),
        byte_to_float((x >> 8) & 0xff),
        byte_to_float((x >> 0) & 0xff),
    ], dtype='float32')
    return result


def loss(p, sample, target):
    predictions = step(sample, p)
    labels = jax.nn.one_hot(target, CLASSES, dtype='float32')
    return jnp.sum((labels - predictions)**2)


#opt_init, opt_update, get_params = optimizers.adagrad(step_size=1e-3)
opt_init, opt_update, get_params = optimizers.sgd(step_size=STEP_SIZE)


@jax.jit
def train_step(i, opt_state, sample, target):
  params = get_params(opt_state)
  g = jax.grad(loss)(params, sample, target)
  return opt_update(i, g, opt_state)


def read_records():
    d = []
    with open(os.path.expanduser('~/x86_3B.csv')) as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            d.append((int(row[0].replace(' ', ''), 16), int(row[1])))
    print(f'records read: {len(d):,}')
    return d


def run_eval(p, d):
    confusion = collections.defaultdict(lambda: 0)
    for in_bytes, want in random.sample(d, EVAL_SAMPLES):
        sample = value_to_sample(in_bytes)
        probs = step(sample, p)
        #print('probs:', probs)
        got = probs.argmax(axis=0)
        confusion[(want, got.item())] += 1


    for want in range(CLASSES):
        print('want', want, ': ', end='')
        for got in range(CLASSES):
            print('{:3d}'.format(confusion[(want, got)]), end=' ')
        print()

    accuracy = sum(confusion[(i, i)] for i in range(CLASSES)) / float(EVAL_SAMPLES) * 100.0
    print(f'accuracy: {accuracy:.2f}%')


def minibatch_iter(d):
    samples = []
    targets = []
    for in_bytes, target in d:
        minibatch.append(value_to_sample(in_bytes))
        targets.append(target)
        if len(samples) % 128 == 0:
            yield jnp.array(samples), jnp.array(targets)
    if samples:
        yield jnp.array(samples), jnp.array(targets)


def run_train():
    train_start = datetime.now()
    d = read_records()
    key = rng.PRNGKey(0)
    p = [(rng.normal(key, (INPUT_LEN, CARRY_LEN)), rng.normal(key, (CARRY_LEN,))) for _ in range(BYTES)]
    p.append(rng.normal(key, (CARRY_LEN, FC)))
    p.append(rng.normal(key, (FC, CLASSES)))
    opt_state = opt_init(p)

    for epoch in range(16):
        print('... epoch shuffle', epoch)
        random.shuffle(d)
        print('... epoch start', epoch)
        for i, minibatch in enumerate(minibatch_iter(d)):
            if i % 256 == 255:
                now = datetime.now()
                print('... step', i, '@', now, '@ {:.2f} step/s'.format(i / (now-train_start).total_seconds()))
                run_eval(get_params(opt_state), d)
            samples, targets = minibatch
            opt_state = train_step(i, opt_state, samples, targets)
        p = get_params(opt_state)
        run_eval(p)

    train_end = datetime.now()

    print('train time:', train_end - train_start)


def main():
    run_train()

if __name__ == '__main__':
    main()
