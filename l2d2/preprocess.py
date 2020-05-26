from dataclasses import dataclass
from typing import List

import numpy as np

import float_helpers


def byte_to_float(x):
    assert x & 0xff == x, x
    return float_helpers.parts2float((0, float_helpers.BIAS, x << (23-8)))


def crumb(b: int, index: int) -> int:
    return (b >> (2 * index)) & 3


def bit(b: int, index: int) -> int:
    return (b >> index) & 1


@dataclass
class ValueOpts:
    byte: bool
    nibbles: bool
    crumbs: bool
    bits: bool



def value_byte_to_floats(b: int, opts: ValueOpts) -> np.array:
    """Constructs a vector that characterizes a byte's content in various ways.

    We create:

        [byte, hi_nibble, lo_nibble, crumb0, crumb1, crumb2, crumb3] +
        [bit0, bit1, ..., bit8]

    Where the bits contents are placed in a float via byte_to_float.
    """
    pieces = []
    if opts.byte:
        pieces.append(byte_to_float(b)) # value for whole byte
    if opts.nibbles:
        byte_to_float((b >> 4) & 0xf),  # upper nibble
        byte_to_float(b & 0x0f),        # lower nibble
    if opts.crumbs:
        byte_to_float(crumb(b, 0)),
        byte_to_float(crumb(b, 1)),
        byte_to_float(crumb(b, 2)),
        byte_to_float(crumb(b, 3)),
    if opts.bits:
        for i in range(8):
            pieces.append(byte_to_float(bit(b, i)))
    return np.array(pieces)


def value_to_sample(bs: List[int], opts: ValueOpts) -> np.array:
    """Creates an input vector sample from the sequence of bytes."""
    return np.concatenate(tuple(value_byte_to_floats(b, opts) for b in bs))


def samples_to_input(samples: List[List[int]], opts: ValueOpts) -> np.array:
    """Converts a sequence of bytes-samples into a minibatch array."""
    return np.array([value_to_sample(sample, opts) for sample in samples])
