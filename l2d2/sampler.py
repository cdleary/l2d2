import datetime
import threading
import time
import queue
from typing import Tuple, List

import numpy as np

from l2d2.common import scoped_time
from l2d2.zoo import INPUT_BYTES
import xtrie


class SamplerThread(threading.Thread):
    def __init__(self, queue: queue.Queue, data: xtrie.XTrie, batch_size: int):
        super().__init__()
        self.q = queue
        self._orig_data = data
        self._data = data.clone()
        self._batch_size = batch_size
        self._cancel = threading.Event()
        self._start = threading.Event()
        self.i = 0

    def restart(self):
        self._data = self._orig_data.clone()
        self._start.set()
        self.i = 0

    def cancel(self) -> None:
        self._cancel.set()

    def start(self):
        self._start.set()
        super().start()

    def get_epoch_iterator(self):
        self.restart()
        while True:
            try:
                mb = self.q.get(timeout=.1)
            except queue.Empty:
                if self._cancel.is_set():
                    return
                else:
                    continue
            if mb is None:
                return
            yield mb

    def _run(self):
        while True:
            if self._cancel.is_set():
                return
            if not self._start.is_set():
                time.sleep(0.1)
                continue

            while not self._data.empty():
                self.i += 1
                mb = self._data.sample_nr_mb(self._batch_size, INPUT_BYTES)
                while True:
                    try:
                        self.q.put(mb, timeout=.1)
                        break
                    except queue.Full:
                        if self._cancel.is_set():
                            return
            print(f'... done after {self.i} enqueues')
            self.q.put(None)
            self._start = threading.Event()

    def run(self):
        try:
            self._run()
        except Exception as e:
            self._cancel.set()
            raise


def get_eval_data(data: xtrie.XTrie, batch_size: int,
                  eval_minibatches: int) -> Tuple:
    data = data.clone()  # Clone the data because we sample w/o replacement.
    inputs, lengths = [], []
    for _ in range(eval_minibatches):
        if data.empty():
            # If we don't have enough minibatches, break early.
            break
        mb = data.sample_nr_mb(batch_size, INPUT_BYTES)
        inputs.append(mb.floats)
        lengths.append(mb.lengths)

    inputs = np.concatenate(inputs)
    lengths = np.concatenate(lengths)
    return inputs, lengths



def _benchmark_sample_nr_mb(data, batch_size: int):
    print('... using batch size:', batch_size)

    COUNT = 256
    byte_count = BYTES
    start = datetime.datetime.now()
    for i in range(COUNT):
        data.sample_nr_mb(batch_size, byte_count)
    end = datetime.datetime.now()
    direct_sample_usec = (end-start).total_seconds()/COUNT*1e6
    print(f'... avg {direct_sample_usec:,.1f} usec/mb')
    print(f'... avg {direct_sample_usec/batch_size:,.1f} usec/sample')


def main():
    import optparse
    import timeit

    import options
    import ingest

    parser = optparse.OptionParser()
    options.add_model_hparams(parser)
    opts, args = parser.parse_args()

    xtopts = xtrie.mk_opts()
    with scoped_time('loading x86 data'):
        orig_data = ingest.load_state('/tmp/x86.state', xtopts)

    data = orig_data.clone()

    print('... timing nop call')
    NOP_COUNT = 4096
    start = datetime.datetime.now()
    for i in range(NOP_COUNT):
        data.nop()
    end = datetime.datetime.now()
    nop_call_nsec = (end-start).total_seconds() * 1e9 / NOP_COUNT
    print(f'... nop call avg {nop_call_nsec:,.3f} nsec')

    for bs in (128,):
        _benchmark_sample_nr_mb(data, bs)

    data = orig_data.clone()

    q = queue.Queue()
    sampler = SamplerThread(q, data, opts.batch_size)
    sampler.start()

    with scoped_time('warmup0'):
        mb = q.get()  # Warmup.
        assert mb.floats.shape[0] == opts.batch_size
        assert mb.lengths.shape == (opts.batch_size,)

    with scoped_time('warmup1'):
        q.get()

    COUNT = 256
    start = datetime.datetime.now()
    for _ in range(COUNT):
        q.get()
    end = datetime.datetime.now()

    sampler.cancel()
    sampler.join()

    seconds = (end-start).total_seconds()
    print('throughput:            {:,.2f} mb/s'.format(COUNT/seconds))
    print('amortized / minibatch: {:,.2f} usec'.format(seconds/COUNT*1e6))



if __name__ == '__main__':
    main()
