import random
from typing import List, Optional, Tuple

from ingest import State, Terminal


_SAMPLED_SENTINEL = object()


def _sample_instruction(node: List, replacement: bool) -> Optional[Tuple[List[int], int]]:
    if all(e is _SAMPLED_SENTINEL for e in node):
        return None  # Node is exhausted.

    while True:
        index = random.randrange(len(node))
        sub_node = node[index]

        if sub_node is _SAMPLED_SENTINEL:
            # Keep sampling until we find something we haven't sampled yet, as
            # a quick hack. Better would be to condense the valid indices and
            # select among those, then map them back.
            continue
        elif isinstance(sub_node, Terminal):
            if not replacement:
                node[index] = _SAMPLED_SENTINEL
            return [index], sub_node.length
        elif isinstance(sub_node, list):
            result = _sample_instruction(sub_node, replacement)
            if result is None:
                if not replacement:
                    node[index] = _SAMPLED_SENTINEL
                return [index], 0
            bs, target = result
            return [index] + bs, target
        else:
            assert sub_node is None
            if not replacement:
                node[index] = _SAMPLED_SENTINEL
            return [index], 0


def sample_instruction(state: State, zeros_ok: bool,
                       replacement: bool) -> Optional[Tuple[List[int], int]]:
    while True:
        result = _sample_instruction(state.data, replacement=replacement)
        if not zeros_ok and result is not None and result[1] == 0:
            continue
        return result


def sample_minibatches(state: State, minibatch_size: int, length: int,
                       zeros_ok: bool, replacement: bool):
    sampled = []
    targets = []
    while True:
        for i in range(minibatch_size):
            got = sample_instruction(state, zeros_ok=zeros_ok,
                                     replacement=replacement)
            if got is None:
                break
            sample, target = got
            # Fill the rest of the bytes with random noise.
            while len(sample) < length:
                sample.append(random.randrange(0x100))
            sampled.append(sample)
            targets.append(target)
        yield sampled, targets
        sampled, targets = [], []

    while len(sampled) < length:
        sampled.append([0] * length)
        targets.append(0)
    yield sampled, targets
