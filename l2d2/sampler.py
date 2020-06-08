import datetime
import threading
import queue
from typing import Tuple, List

import numpy as np
from jax import numpy as jnp

import xtrie
from zoo import BYTES


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
            floats = jnp.array(mb.floats, dtype='float32')
            assert floats.shape[0] == self._batch_size
            lengths = jnp.array(mb.lengths, dtype='uint8')
            assert lengths.shape == (self._batch_size,)
            mb = (floats, lengths)
            while True:
                did_cancel = self._run_with_cancel(mb)
                if did_cancel:
                    return
        self._queue.put(None)


def get_eval_data(data: xtrie.XTrie, batch_size: int,
                  eval_minibatches: int) -> Tuple:
    data = data.clone()
    eval_sample_start = datetime.datetime.now()
    inputs, lengths, opcodes = [], [], []
    for _ in range(eval_minibatches):
        mb = data.sample_nr_mb(batch_size, BYTES)
        inputs += mb.floats
        lengths += mb.lengths
        opcodes += mb.opcodes

    inputs = np.array(inputs, dtype='float32')
    lengths = np.array(lengths, dtype='uint8')
    eval_sample_end = datetime.datetime.now()
    print('sampling time per minibatch: {:.3f} us'.format(
          (eval_sample_end-eval_sample_start).total_seconds()
            / eval_minibatches * 1e6))
    return inputs, lengths
