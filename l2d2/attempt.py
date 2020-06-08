import collections
import contextlib
import datetime
import functools
import os
import optparse
import queue
import sys
import time
from typing import List, Tuple

import termcolor

import jax
from jax import random as rng
from jax import numpy as jnp
from jax.experimental import optimizers

import ingest
import options
import preprocess
import sampler
import xtrie
import zoo


def run_eval(p, eval_data: Tuple[jnp.array, jnp.array], carry_len: int) -> Tuple[float, float]:
    """Runs evaluation step on the given eval_minibatches with parameters p."""
    print('... running eval')
    sys.stdout.flush()
    confusion = collections.defaultdict(lambda: 0)
    floats, wants = eval_data
    loss, probs = zoo.loss_and_preds(p, floats, wants, carry_len)
    gots = probs.argmax(axis=-1)
    for i in range(floats.shape[0]):
        confusion[(wants[i], gots[i].item())] += 1

    # Print out the confusion matrix.
    for want in range(zoo.CLASSES):
        print(f'want {want:2d}: ', end='')
        for got in range(zoo.CLASSES):
            value = confusion[(want, got)]
            color = 'green' if want == got else ('red' if value != 0 else None)
            print(termcolor.colored(f'{value:5d}', color=color), end=' ')
        print()

    # Print out summary accuracy statistic(s).
    correct = sum(
            confusion[(i, i)]
            for i in range(zoo.CLASSES))
    total = sum(confusion.values())
    accuracy = correct / float(total) * 100.0
    print(f'loss: {loss:.3f} accuracy: {accuracy:.2f}% ({correct} / {total})')
    sys.stdout.flush()
    return accuracy, loss


@contextlib.contextmanager
def scoped_time(annotation):
    print('...', annotation)
    start = datetime.datetime.now()
    yield
    end = datetime.datetime.now()
    print('...', annotation, 'done in', end-start)


class StatRecorder:

    def __init__(self):
        self.f = open('/tmp/l2d2_{}.txt'.format(time.time()), 'w')

    def note_eval_accuracy(self, epochno: int, stepno: int, accuracy: float, loss: float):
        now = datetime.datetime.now()
        print(f't: {now} epochno: {epochno} stepno: {stepno} accuracy: {accuracy} loss: {loss}', file=self.f)
        self.f.flush()



def _do_train(opt_state, eval_data, epochs: int, batch_size: int,
              carry_len: int, train_steps_per_eval: int, q: queue.Queue,
              stat_recorder: StatRecorder, get_params, opt_update):
    print('... starting training')

    train_start = datetime.datetime.now()

    for epoch in range(epochs):
        print('... epoch start', epoch)
        epoch_start = datetime.datetime.now()
        stepno = 0

        def compute_rate(now=None):
            now = now or datetime.datetime.now()
            steps_per_sec = stepno / (now-train_start).total_seconds()
            samples_per_sec = steps_per_sec * batch_size
            return '{:6.2f} step/s => {:8,.2f} samples/s'.format(
                              steps_per_sec, samples_per_sec)

        while True:
            if stepno % train_steps_per_eval == train_steps_per_eval - 1:
                now = datetime.datetime.now()
                rate = compute_rate(now)
                print()
                print('... epoch', epoch, 'step', stepno, '@', now, '@', rate)
                accuracy, loss = run_eval(get_params(opt_state), eval_data, carry_len)
                stat_recorder.note_eval_accuracy(epoch, stepno, accuracy, loss)

            # Grab item from the queue, see if it's a "terminate" sentinel.
            item = q.get()
            if item is None:
                break  # Done with epoch.

            floats, lengths = item 
            opt_state = zoo.train_step(stepno, opt_state, floats, lengths,
                                       carry_len, get_params, opt_update)
            stepno += 1
            sys.stdout.write('.')
            if stepno % 64 == 0:
                print(' ' + compute_rate())

        # End of epoch!
        print()
        print(f'... end of epoch {epoch}; {stepno} steps => {stepno * batch_size} samples')
        p = get_params(opt_state)
        accuracy, loss = run_eval(p, eval_data, carry_len)
        stat_recorder.note_eval_accuracy(epoch, stepno, accuracy, loss)

    train_end = datetime.datetime.now()

    print('... train time:', train_end - train_start)


def run_train(opts) -> None:
    """Runs a training routine."""
    opt_init, opt_update, get_params = optimizers.adagrad(
        step_size=opts.step_size)

    with scoped_time('initializing params'):
        # Initialize parameters.
        key = rng.PRNGKey(0)
        p = zoo.init_params(key, opts.carry_len)

        # Package up in optimizer state.
        opt_state = opt_init(p)

    with scoped_time('loading x86 data'):
        xtopts = xtrie.mk_opts()
        data = ingest.load_state('/tmp/x86.state', xtopts)

    with scoped_time('getting eval data'):
        eval_data = sampler.get_eval_data(data, opts.batch_size,
                                          opts.eval_minibatches)

    print('... starting sampler thread')
    q = queue.Queue(maxsize=32)
    sampler_thread = sampler.SamplerThread(q, data, opts.batch_size)
    sampler_thread.start()

    stat_recorder = StatRecorder()

    try:
        _do_train(opt_state, eval_data, opts.epochs, opts.batch_size,
                  opts.carry_len, opts.steps_per_eval, q, stat_recorder,
                  get_params, opt_update)
    except Exception as e:
        sampler_thread.cancel()
        sampler_thread.join()
        raise


def main():
    parser = optparse.OptionParser()
    parser.add_option('--epochs', type=int, default=8,
                      help='Number of iterations through the training data '
                           'before completing')
    parser.add_option('--steps-per-eval', type=int, default=4096,
                      help='Number of training steps to perform before doing '
                           'an accuracy evaluation')
    parser.add_option('--eval-minibatches', type=int, default=16,
                      help='number of minibatches to use for eval (test) data')
    options.add_model_hparams(parser)
    opts, args = parser.parse_args()
    run_train(opts)


if __name__ == '__main__':
    main()
