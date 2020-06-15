import collections
import datetime
import pickle
import subprocess as subp

import flax
import jax
from jax import numpy as jnp

from l2d2 import common
from l2d2 import sampler
import xtrie


class CNN(flax.nn.Module):
  def apply(self, x):
    x = flax.nn.Conv(x, features=16, kernel_size=(5,))
    x = flax.nn.relu(x)
    x = flax.nn.Conv(x, features=32, kernel_size=(3,))
    x = flax.nn.relu(x)
    x = x.reshape(x.shape[0], -1)
    x = flax.nn.Dense(x, features=9)
    x = flax.nn.log_softmax(x)
    return x


def cross_entropy_loss(logits, label):
    return -(logits*label)


@jax.partial(jax.jit, static_argnums=2)
def train_step(optimizer, minibatch, num_classes: int):
    floats, lengths = minibatch
    lengths = jax.nn.one_hot(lengths, num_classes)

    def loss_fn(model):
        logits = model(floats)
        #print('floats shape: ', floats.shape)
        #print('lengths shape:', lengths.shape)
        #print('logits shape: ', logits.shape)
        loss = jnp.mean(cross_entropy_loss(
            logits, lengths))
        return loss

    grad = jax.grad(loss_fn)(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer


def train(s: sampler.SamplerThread, opts):
    if opts.train_continue:
        model = load_model(opts)
    else:
        _, initial_params = CNN.init_by_shape(
            jax.random.PRNGKey(0),
            [((opts.batch_size, 8, 15), jnp.float32)])
        model = flax.nn.Model(CNN, initial_params)

    optimizer = flax.optim.Adam(
      learning_rate=opts.step_size, weight_decay=1e-5).create(model)

    print('... training for', opts.epochs, 'epochs')
    start = datetime.datetime.now()

    for epoch in range(opts.epochs):
        epoch_start = datetime.datetime.now()
        print('... epoch', epoch, epoch_start-start)
        for minibatch in s.get_epoch_iterator():
            assert minibatch.floats.shape == (opts.batch_size, 8, 15)
            assert minibatch.lengths.shape == (opts.batch_size,)
            optimizer = train_step(optimizer, (minibatch.floats, minibatch.lengths), opts.len_limit)
        steps_per_sec = s.i/(datetime.datetime.now()-epoch_start).total_seconds()
        print('... finished epoch', epoch, f'{steps_per_sec:.2f} steps/s')

    print('... dumping model to disk:', opts.model)
    with open(opts.model, 'wb') as f:
        state = flax.serialization.to_state_dict(optimizer.target)
        pickle.dump(state, f)

    return optimizer


def disasm_bytes(bs) -> str:
    return subp.check_output(['/usr/bin/ndisasm', '-b', '64', '-'], input=bytes(bs)).decode('utf-8').strip()


def do_eval(model, eval_data, opts, xtopts):
    floats, want_lengths = eval_data
    print('eval floats: ', floats.shape)
    print('eval lengths:', want_lengths.shape)

    @jax.jit
    def f(floats):
        logits = model(floats)
        return logits.argmax(axis=-1)

    got_lengths = f(floats)
    assert got_lengths.shape == want_lengths.shape, (got_lengths.shape, want_lengths.shape)
    confusion = collections.defaultdict(lambda: 0)
    for i in range(floats.shape[0]):
        wl = want_lengths[i].item()
        gl = got_lengths[i].item()
        confusion[(wl, gl)] += 1
        if wl != gl:
            bs = xtrie.floats_to_bytes(floats[i], xtopts)[:want_lengths[i]]
            print(' '.join(f'{b:02x}' for b in bs), '\n::', disasm_bytes(bs))

    common.print_confusion(confusion, classes=opts.len_limit)


def do_train(s: sampler.SamplerThread, opts):
    try:
        optimizer = train(s, opts)
    except Exception as e:
        print('... stopping on exception:', e)
        s.cancel()
        s.join()
        raise

    s.cancel()
    s.join()
    print('... done training')
    return optimizer


def load_model(opts):
    print('... loading model from:', opts.model)
    with open(opts.model, 'rb') as f:
        state_dict = pickle.load(f)
    model = flax.nn.Model(CNN, state_dict['params'])
    return model


def main():
    import optparse
    import queue

    from l2d2 import ingest
    from l2d2 import options

    parser = optparse.OptionParser()
    parser.add_option('--epochs', type=int, default=32,
                      help='Number of iterations through the training data '
                           'before completing')
    parser.add_option('--len-limit', type=int, default=9,
                      help='Limit on the length of an instruction (number '
                           'of classes / input data spatial)')
    parser.add_option('--model', default='/tmp/l2d2_model.pickle', type='str',
                      help='Path to saved model')
    parser.add_option('--train', default=False, action='store_true',
                      help='Train a model and save it before eval')
    parser.add_option('--train-continue', default=False, action='store_true',
                      help='Load saved model before training')
    parser.add_option('--data', default='/tmp/x86.state',
                      help='Path to ingested x86 data')
    options.add_model_hparams(parser)
    opts, args = parser.parse_args()

    if args:
        parser.error('No args are permitted')

    xtopts = xtrie.mk_opts()
    with common.scoped_time('loading x86 data'):
        data = ingest.load_state(opts.data, xtopts)

    # Spin up the data sampler thread.
    q = queue.Queue(maxsize=64)
    s = sampler.SamplerThread(q, data, opts.batch_size)
    s.start()

    if opts.train:
        model = do_train(s, opts).target
    else:
        model = load_model(opts)

    # Grab eval data.
    with common.scoped_time('getting eval data'):
        eval_data = sampler.get_eval_data(data, opts.batch_size,
                                          opts.eval_minibatches)
    do_eval(model, eval_data, opts, xtopts)

    s.cancel()
    s.join()



if __name__ == '__main__':
    from ipdb import launch_ipdb_on_exception
    with launch_ipdb_on_exception():
        main()
