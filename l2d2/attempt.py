import collections
import contextlib
import datetime
import functools
import os
import optparse
import queue
import sys
import threading
from typing import List, Tuple

import numpy as np

import jax
from jax.experimental import optimizers
from jax import numpy as jnp
from jax import random as rng

import ingest
import preprocess
import xtrie


VALUE_OPTS = preprocess.ValueOpts(byte=True, nibbles=True, crumbs=True, bits=True)
INPUT_FLOATS_PER_BYTE = len(preprocess.value_byte_to_floats(0, VALUE_OPTS))
BYTES = 15              # Input byte count (with instruction data).
INPUT_FLOATS = (        # Bytes are turned into a number of floats.
    BYTES * INPUT_FLOATS_PER_BYTE)
CLASSES = BYTES+1       # Length categories; 0 for "not a valid instruction".
EVAL_MINIBATCHES = 32   # Number of minibatches for during-training eval.
STEP_SIZE = 1e-3        # Learning rate.
FC = 4096               # Fully connected neurons to use on last hidden output.


@jax.partial(jax.jit, static_argnums=2)
def step(xs, p, carry_len: int):
    assert xs.shape[1] == INPUT_FLOATS, xs.shape
    batch_size = xs.shape[0]
    h = jnp.zeros((batch_size, carry_len))
    for byteno, i in enumerate(range(0, BYTES*INPUT_FLOATS_PER_BYTE, INPUT_FLOATS_PER_BYTE)):
        w, b = p[byteno]
        x = xs[:,i:i+INPUT_FLOATS_PER_BYTE]
        assert x.shape == (batch_size, INPUT_FLOATS_PER_BYTE), x.shape
        x = jnp.concatenate((x, h[:,INPUT_FLOATS_PER_BYTE:]), axis=-1)
        assert x.shape == (batch_size, carry_len), x.shape
        y = x @ w + b
        z = jnp.tanh(y)
        h = z
    assert h.shape == (batch_size, carry_len)
    z = jnp.tanh(h @ p[-2][0] + p[-2][1])
    assert z.shape == (batch_size, FC)
    return jax.nn.softmax(z @ p[-1])


@jax.partial(jax.jit, static_argnums=3)
def loss(p, sample, target, carry_len: int):
    batch_size = sample.shape[0]
    assert sample.shape == (batch_size, INPUT_FLOATS)
    assert target.shape == (batch_size,), target.shape
    predictions = step(sample, p, carry_len)
    labels = jax.nn.one_hot(target, CLASSES, dtype='float32')
    return jnp.sum((labels - predictions)**2)


opt_init, opt_update, get_params = optimizers.adagrad(step_size=STEP_SIZE)
#opt_init, opt_update, get_params = optimizers.sgd(step_size=STEP_SIZE)


@jax.partial(jax.jit, static_argnums=4)
def train_step(i, opt_state, sample, target, carry_len: int):
    batch_size = sample.shape[0]
    assert sample.shape == (batch_size, INPUT_FLOATS)
    assert target.shape == (batch_size,), target
    params = get_params(opt_state)
    g = jax.grad(loss)(params, sample, target, carry_len)
    return opt_update(i, g, opt_state)


def run_eval(p, eval_data: Tuple[np.array, np.array], carry_len: int):
    """Runs evaluation step on the given eval_minibatches with parameters p."""
    print('... running eval')
    sys.stdout.flush()
    confusion = collections.defaultdict(lambda: 0)
    samples, wants = eval_data
    probs = step(preprocess.samples_to_input(samples, VALUE_OPTS), p, carry_len)
    gots = probs.argmax(axis=-1)
    assert isinstance(samples, list)
    for i in range(len(samples)):
        confusion[(wants[i], gots[i].item())] += 1

    # Print out the confusion matrix.
    for want in range(CLASSES):
        print(f'want {want:2d}: ', end='')
        for got in range(CLASSES):
            print('{:5d}'.format(confusion[(want, got)]), end=' ')
        print()

    # Print out summary accuracy statistic(s).
    accuracy = sum(
            confusion[(i, i)]
            for i in range(CLASSES)) / float(sum(confusion.values())) * 100.0
    print(f'accuracy: {accuracy:.2f}%')
    sys.stdout.flush()


def time_train_step(batch_size: int, carry_len: int, opt_state) -> datetime.timedelta:
    """Times a single training step after warmup."""
    fake_sample = np.zeros((batch_size, INPUT_FLOATS), dtype='float32')
    fake_target = np.zeros((batch_size,), dtype='int32')

    # Forward warmup.
    loss(get_params(opt_state), fake_sample, fake_target, carry_len).block_until_ready()

    # Forward timed.
    start = datetime.datetime.now()
    loss(get_params(opt_state), fake_sample, fake_target, carry_len).block_until_ready()
    end = datetime.datetime.now()
    fwd_time = end - start

    # Train warmup.
    opt_state = train_step(0, opt_state, fake_sample, fake_target, carry_len)
    get_params(opt_state)[-1].block_until_ready()

    # Train timed.
    start = datetime.datetime.now()
    opt_state = train_step(0, opt_state, fake_sample, fake_target, carry_len)
    get_params(opt_state)[-1].block_until_ready()
    end = datetime.datetime.now()
    step_time = end - start
    return fwd_time, step_time


@jax.partial(jax.jit, static_argnums=1)
def init_params(key, carry_len: int):
    p = [(rng.normal(key, (carry_len, carry_len), dtype='float32'),
          rng.normal(key, (carry_len,), dtype='float32'))
         for _ in range(BYTES)]
    p.append((
        rng.normal(key, (carry_len, FC), dtype='float32'),
        rng.normal(key, (FC,), dtype='float32'),
    ))
    p.append(rng.normal(key, (FC, CLASSES), dtype='float32'))
    return p


def _get_eval_data(data: xtrie.XTrie, batch_size: int) -> Tuple[List[bytes], List[int]]:
    data = data.clone()
    eval_sample_start = datetime.datetime.now()
    inputs, targets, opcodes = [], [], []
    for _ in range(EVAL_MINIBATCHES):
        bs, ts, opcs = data.sample_nr_mb(batch_size, BYTES)
        inputs += bs
        targets += ts
        opcodes += opcs
    eval_sample_end = datetime.datetime.now()
    print('sampling time per minibatch: {:.3f} us'.format(
          (eval_sample_end-eval_sample_start).total_seconds()
            / EVAL_MINIBATCHES * 1e6))
    return inputs, targets


class SamplerThread(threading.Thread):
    def __init__(self, queue: queue.Queue, data: xtrie.XTrie, batch_size: int):
        super().__init__()
        self._queue = queue
        self._data = data
        self._batch_size = batch_size
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def _run_with_cancel(self, data_to_put) -> bool:
        try:
            self._queue.put(data_to_put, timeout=.1)
        except queue.Full:
            pass
        return self._cancel.is_set()

    def run(self):
        while not self._data.empty():
            mb = self._data.sample_nr_mb(self._batch_size, BYTES)
            while True:
                did_cancel = self._run_with_cancel(mb)
                if did_cancel:
                    return
        self._queue.put(None)


@contextlib.contextmanager
def scoped_time(annotation):
    print('...', annotation)
    start = datetime.datetime.now()
    yield
    end = datetime.datetime.now()
    print('...', annotation, 'done in', end-start)


def _do_train(opt_state, eval_data, epochs: int, batch_size: int, carry_len: int,
              train_steps_per_eval: int, q: queue.Queue):
    print('... starting training')
    train_start = datetime.datetime.now()

    for epoch in range(epochs):
        print('... epoch start', epoch)
        stepno = 0
        while True:
            if stepno % train_steps_per_eval == train_steps_per_eval - 1:
                now = datetime.datetime.now()
                print()
                print('... epoch', epoch, 'step', stepno, '@', now,
                      '@ {:.2f} step/s'.format(
                          stepno / (now-train_start).total_seconds()))
                sys.stdout.flush()
                run_eval(get_params(opt_state), eval_data, carry_len)

            # Grab item from the queue, see if it's a "terminate" sentinel.
            item = q.get()
            if item is None:
                break  # Done with epoch.

            samples, targets, opcodes = item
            assert len(targets) == batch_size, targets
            assert len(samples) == batch_size
            samples = preprocess.samples_to_input(samples, VALUE_OPTS)
            opt_state = train_step(stepno, opt_state, samples,
                                   np.array(targets, dtype='uint8'), carry_len)
            stepno += 1
            sys.stdout.write('.')
            if stepno % 64 == 0:
                sys.stdout.write('\n')
            sys.stdout.flush()

        # End of epoch!
        print()
        print(f'... end of epoch {epoch}; {stepno} steps => {stepno * batch_size} samples')
        p = get_params(opt_state)
        run_eval(p, eval_data, carry_len)

    train_end = datetime.datetime.now()

    print('... train time:', train_end - train_start)


def run_train(opts) -> None:
    """Runs a training routine."""
    with scoped_time('initializing params'):
        # Initialize parameters.
        key = rng.PRNGKey(0)
        p = init_params(key, opts.carry_len)

        # Package up in optimizer state.
        opt_state = opt_init(p)

    fwd_time, step_time = time_train_step(opts.batch_size, opts.carry_len, opt_state)
    steps_per_sec = 1.0/step_time.total_seconds()
    samples_per_sec = steps_per_sec * opts.batch_size
    fwd_percent = fwd_time.total_seconds() / step_time.total_seconds() * 100.0
    fwd_us = int(fwd_time.total_seconds() * 1e6)
    step_us = int(step_time.total_seconds() * 1e6)
    print(f'bs{opts.batch_size:<8} fwd time approximately:  {fwd_us:6,} us ({fwd_percent:.2f}%)')
    print(f'bs{opts.batch_size:<8} step time approximately: {step_us:6,} us; {steps_per_sec:.1f} '
          f'steps/s; {samples_per_sec:.1f} samples/s')

    if opts.time_step_only:
        return

    with scoped_time('loading x86 data'):
        data = ingest.load_state('/tmp/x86.state')

    with scoped_time('getting eval data'):
        eval_data = _get_eval_data(data, opts.batch_size)

    print('... starting sampler thread')
    q = queue.Queue(maxsize=32)
    sampler_thread = SamplerThread(q, data, opts.batch_size)
    sampler_thread.start()

    try:
        _do_train(opt_state, eval_data, opts.epochs, opts.batch_size, opts.carry_len, opts.steps_per_eval, q)
    except Exception as e:
        sampler_thread.cancel()
        sampler_thread.join()
        raise


def main():
    parser = optparse.OptionParser()
    parser.add_option('--time-step-only', action='store_true', default=False,
                      help='Just time a training step, do not train.')
    parser.add_option('--batch-size', type=int, default=32,
                      help='minibatch size')
    parser.add_option('--carry-len', type=int, default=1024,
                      help='vector length for recurrent state')
    parser.add_option('--epochs', type=int, default=8,
                      help='number of iterations through the training data before completing')
    parser.add_option('--steps-per-eval', type=int, default=4096,
                      help='Number of training steps to perform before doing an accuracy evaluation')
    opts, args = parser.parse_args()
    run_train(opts)


if __name__ == '__main__':
    main()
