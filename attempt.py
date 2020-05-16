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


key = rng.PRNGKey(0)
H0 = jnp.zeros((CARRY_LEN,))
p = [(rng.normal(key, (INPUT_LEN, CARRY_LEN)), rng.normal(key, (CARRY_LEN,)))
     for _ in range(BYTES)]
p.append(rng.normal(key, (CARRY_LEN, CLASSES)))


@jax.jit
def step(xs, p):
    h = H0
    for i, (w, b) in enumerate(p[:-1]):
        x = jnp.concatenate((xs[i:i+1], h))
        y = x @ w + b
        z = jnp.tanh(y)
        h = z
    w = p[-1]
    return jax.nn.softmax(h @ w)


def byte_to_float(x):
    assert x & 0xff == x, x
    return float_helpers.parts2float((0, float_helpers.BIAS, x << (23-8)))

d = {}
with open(os.path.expanduser('~/x86_3B.csv')) as f:
    reader = csv.reader(f, delimiter=';')
    for row in reader:
        d[int(row[0].replace(' ', ''), 16)] = int(row[1])
print(f'records read: {len(d):,}')


def value_to_sample(x):
    result = jnp.array([
        byte_to_float((x >> 16) & 0xff),
        byte_to_float((x >> 8) & 0xff),
        byte_to_float((x >> 0) & 0xff),
    ], dtype='float32')


def loss(p, sample, target):
    predictions = step(sample, p)
    labels = jax.nn.one_hot(target, CLASSES, dtype='float32')
    return jnp.sum((labels - predictions)**2)


#opt_init, opt_update, get_params = optimizers.adagrad(step_size=1e-3)
opt_init, opt_update, get_params = optimizers.sgd(step_size=1e-5)


@jax.jit
def train_step(i, opt_state, sample, target):
  params = get_params(opt_state)
  g = jax.grad(loss)(params, sample, target)
  return opt_update(i, g, opt_state)


train_start = datetime.now()
opt_state = opt_init(p)

for i in range(8):
    for i in range(8192):
        if i % 1024 == 1023:
            now = datetime.now()
            print('... step', i, '@', now, '@ {:.2f} step/s'.format(i / (now-train_start).total_seconds()))
        in_bytes = random.randrange(0xffffff+1)
        target = d.get(in_bytes, 0)
        sample = value_to_sample(in_bytes)
        #print('sample:', sample, 'target:', target)
        opt_state = train_step(i, opt_state, sample, target)
    p = get_params(opt_state)

    confusion = collections.defaultdict(lambda: 0)
    for i in range(EVAL_SAMPLES):
        in_bytes = random.randrange(0xffffff+1)
        want = d.get(in_bytes, 0)
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

train_end = datetime.now()

print('train time:', train_end - train_start)


