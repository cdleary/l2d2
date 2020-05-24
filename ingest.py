from dataclasses import dataclass
from datetime import datetime
import os
import pickle
import pprint
import subprocess as subp
from typing import List, Tuple
import re


@dataclass
class Terminal:
    length: int
    mnemonic: str


class State:
    def __init__(self):
        self.data = [None] * 256
        self.total = 0
        self.binaries = set()


def try_add_binary(state: State, path: str) -> bool:
    assert isinstance(path, str), path
    sha = subp.check_output(['/usr/bin/sha256sum', path]).strip()
    if sha in state.binaries:
        return False
    state.binaries.add(sha)
    return True


def get_object_intervals(binpath: str) -> List[Tuple[int, int]]:
    output = subp.check_output(['readelf', '--symbols', binpath]).decode('utf-8')
    results = []
    for line in output.splitlines():
        if 'OBJECT' not in line:
            continue
        m = re.match(r'\s*\d+: (?P<addr>[a-f\d]+)\s+(?P<size>[a-f\dx]+)\s+OBJECT\s+GLOBAL\s+DEFAULT\s+(?P<ndx>\d+)\s+(?P<symbol>\w+)', line)
        if not m:
            continue
        addr = int(m.group('addr'), 16)
        size = int(m.group('size'), 16)
        print(f'{addr:x}-{addr+size:x}: {m.group("symbol")}')
        results.append((addr, addr+size))
    return results


def ingest_binary(state: State, path: str) -> None:
    if not try_add_binary(state, path):
        print(f'... already seen {path:<60}')
        return
    before_total = state.total
    print(f'... ingesting {path:<60} (before: {before_total:12,})')
    output = subp.check_output(['objdump', '-d', path, '-j', '.text']).decode('utf-8')
    lines = output.splitlines()

    for line in lines:
        m = re.match(
            r'\s*(?P<addr>\d+):\s+(?P<bytes>([a-f\d]{2} )+)(?P<mnemonic>.*)$',
            line)
        if not m:
            continue
        bytes_group = m.group('bytes')
        bs = bytes_group.split()
        if len(bs) == 16:  # Data disassembly.
            continue
        addr = int(m.group('addr'), 16)
        mnemonic = m.group('mnemonic').strip() 
        if not mnemonic:
            continue
        node = state.data
        for byteno, b in enumerate(int(b, 16) for b in bs):
            if len(bs) == byteno+1:  # Note the length of the terminal.
                assert node[b] is None or node[b].length == len(bs), \
                    (byteno, line)
                state.total += node[b] != len(bs)
                node[b] = Terminal(len(bs), mnemonic)
            elif node[b] is None:
                node[b] = [None] * 256
                node = node[b]
            else:
                node = node[b]
                assert isinstance(node, list), (line)

    after_total = state.total
    delta = after_total - before_total
    print(f'... ingested  {path:<60} (after:  {after_total:12,}; delta: {delta:12,})')


def print_sparsely(d, depth=0, do_print=True):
    indent_str = ' ' * (2 * depth)
    maxdepth = depth
    seen_terminals = 0
    histo = [0] * 16
    for i, item in enumerate(d):
        if item is None:
            continue
        if isinstance(item, Terminal):
            if do_print:
                print(f'{indent_str}{i:02x}: {item.length}')
            seen_terminals += 1
            histo[item.length] += 1
        else:
            if do_print:
                print(f'{indent_str}{i:02x}:')
            sub_maxdepth, sub_seen_terminals, sub_histo = print_sparsely(
                item, depth+1, do_print=do_print)
            maxdepth = max(sub_maxdepth, maxdepth)
            seen_terminals += sub_seen_terminals
            histo = [h+sh for h, sh in zip(histo, sub_histo)]
    return maxdepth, seen_terminals, histo


def load_state(path: str) -> State:
    load_start = datetime.now()
    with open(path, 'rb') as f:
        state = pickle.load(f)
    load_end = datetime.now()
    print('load time:', load_end-load_start)
    return state
