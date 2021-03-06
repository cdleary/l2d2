import collections
import datetime
import functools
import os
import optparse
import queue
import sys
import time
from typing import List, Tuple

import termcolor
import numpy as np

import jax
from jax import random as rng
from jax import numpy as jnp
from jax.experimental import optimizers

from .common import scoped_time
from . import ingest
from . import options
from . import preprocess
from . import sampler
from . import zoo

import xtrie


def run_eval(p, eval_data: Tuple[jnp.array, jnp.array], rng_key, carry_len: int) -> Tuple[float, float]:
    """Runs evaluation step on the given eval_minibatches with parameters p."""
    print('... running eval')
    sys.stdout.flush()
    confusion = collections.defaultdict(lambda: 0)
    floats, wants = eval_data
    print(f'... {len(floats)} samples')
    loss, probs = zoo.loss_and_preds(p, floats, wants, rng_key, carry_len, False)
    gots = probs.argmax(axis=-1)
    for i in range(floats.shape[0]):
        confusion[(wants[i].item(), gots[i].item())] += 1

    print(f'loss: {loss:.3f}')
    accuracy = common.print_confusion(confusion, zoo.CLASSES)
    return accuracy, loss


class StatRecorder:

    def __init__(self):
        self.f = open('/tmp/l2d2_{}.txt'.format(time.time()), 'w')

    def note_eval_accuracy(self, epochno: int, stepno: int, accuracy: float, loss: float):
        now = datetime.datetime.now()
        print(f't: {now} epochno: {epochno:3d} stepno: {stepno:9,d} accuracy: {accuracy:5.2f} loss: {loss:9.6f}', file=self.f)
        self.f.flush()



def _do_train(opt_state, eval_data, epochs: int, batch_size: int, rng_key,
              carry_len: int, train_steps_per_eval: int, sampler: sampler.SamplerThread,
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
                accuracy, loss = run_eval(get_params(opt_state), eval_data, rng_key, carry_len)
                stat_recorder.note_eval_accuracy(epoch, stepno, accuracy, loss)

            # Grab minibatch from the queue, see if it's a "terminate"
            # sentinel.
            mb = sampler.q.get()
            if mb is None:
                sampler.restart()
                break  # Done with epoch.

            #print('i:', mb.floats[0])
            #print('t:', mb.lengths[0])
            opt_state, loss = zoo.train_step(stepno, opt_state, mb.floats,
                                       mb.lengths, rng_key, carry_len, get_params,
                                       opt_update)
            stepno += 1
            if stepno & 7 == 7:
                print(f'{loss.item():7.3f} ', end='')
            if stepno & 63 == 63:
                print('|', compute_rate())

        # End of epoch!
        print()
        print(f'... end of epoch {epoch}; {stepno} steps => {stepno * batch_size} samples')
        p = get_params(opt_state)
        accuracy, loss = run_eval(p, eval_data, rng_key, carry_len)
        stat_recorder.note_eval_accuracy(epoch, stepno, accuracy, loss)

    train_end = datetime.datetime.now()

    print('... train time:', train_end - train_start)


def run_train(opts) -> None:
    """Runs a training routine."""
    if opts.optimizer == 'momentum':
        print('... using momentum optimizer')
        opt_init, opt_update, get_params = optimizers.momentum(mass=0.9,
            step_size=opts.step_size)
    elif opts.optimizer == 'sgd':
        print('... using sgd optimizer')
        opt_init, opt_update, get_params = optimizers.sgd(step_size=opts.step_size)
    else:
        assert opts.optimizer == 'adagrad'
        print('... using adagrad optimizer')
        opt_init, opt_update, get_params = optimizers.adagrad(step_size=opts.step_size)

    with scoped_time('initializing params'):
        # Initialize parameters.
        rng_key = rng.PRNGKey(0)
        p = zoo.init_params(rng_key, opts.carry_len)

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
                  rng_key, opts.carry_len, opts.steps_per_eval, sampler_thread,
                   stat_recorder,
                  get_params, opt_update)
    except Exception as e:
        print('... exception', e, ': terminating sampler thread.')
        sampler_thread.cancel()
        sampler_thread.join()
        raise


def main():
    np.set_printoptions(linewidth=float('inf'))

    parser = optparse.OptionParser()
    parser.add_option('--epochs', type=int, default=1024,
                      help='Number of iterations through the training data '
                           'before completing')
    parser.add_option('--steps-per-eval', type=int, default=4096,
                      help='Number of training steps to perform before doing '
                           'an accuracy evaluation')
    options.add_model_hparams(parser)
    opts, args = parser.parse_args()
    run_train(opts)


if __name__ == '__main__':
    main()
