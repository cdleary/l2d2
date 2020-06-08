import datetime
import threading
import queue
from typing import Tuple, List

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
            while True:
                did_cancel = self._run_with_cancel(mb)
                if did_cancel:
                    return
        self._queue.put(None)


def get_eval_data(data: xtrie.XTrie, batch_size: int,
                  eval_minibatches: int) -> Tuple[List[bytes], List[int]]:
    data = data.clone()
    eval_sample_start = datetime.datetime.now()
    inputs, targets, opcodes = [], [], []
    for _ in range(eval_minibatches):
        mb = data.sample_nr_mb(batch_size, BYTES)
        inputs += mb.bytes
        targets += mb.lengths
        opcodes += mb.opcodes
    eval_sample_end = datetime.datetime.now()
    print('sampling time per minibatch: {:.3f} us'.format(
          (eval_sample_end-eval_sample_start).total_seconds()
            / eval_minibatches * 1e6))
    return inputs, targets
