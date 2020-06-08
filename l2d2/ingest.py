"""Library with functions for ingestion of x86 binaries.

Wraps the xtrie native module. TODO(cdleary) This could all be moved to native
Rust (particularly for fun, as it may not be compute bound).
"""

from dataclasses import dataclass
from datetime import datetime
import os
import pickle
import pprint
import re
import subprocess as subp
import sys
from typing import List, Tuple

import xtrie


_OBJDUMP_LINE_RE = re.compile(r'\s*\d+: (?P<addr>[a-f\d]+)\s+(?P<size>[a-f\dx]+)\s+OBJECT\s+GLOBAL\s+DEFAULT\s+(?P<ndx>\d+)\s+(?P<symbol>\w+)')
_OBJDUMP_ASM_RE = re.compile(
            r'\s*(?P<addr>\d+):\s+(?P<bytes>([a-f\d]{2} )+)(?P<mnemonic>.*)$')
_RELADDR_COMMENT_RE = re.compile(r'# [\da-fx]+')
_RELADDR_RE = re.compile(r'(jmpq?|j[abgln]?e|j[abglops]|jn[eops]|callq)\s+[\da-fx]+')
_SYMBOL_RE = re.compile(r'\<[^>]+\>')


def ingest_binary(state: xtrie.XTrie, path: str) -> None:
    before_total = state.total
    print(f'... ingesting {path:<60} (before: {before_total:12,})', file=sys.stderr)
    output = subp.check_output(['objdump', '-d', path, '-w', '-j', '.text']).decode('utf-8')
    lines = output.splitlines()

    for line in lines:
        m = _OBJDUMP_ASM_RE.match(line)
        if not m:
            continue
        bytes_group = m.group('bytes')
        bs = bytes_group.split()
        assert len(bs) <= 16, 'Overlong instruction'
        if len(bs) == 16:  # Data disassembly.
            continue
        addr = int(m.group('addr'), 16)
        mnemonic = m.group('mnemonic').strip() 
        if not mnemonic:
            continue
        # Sanitize symbols.
        mnemonic = _SYMBOL_RE.sub('', mnemonic)
        # Sanitize PC-relative notes.
        mnemonic = _RELADDR_COMMENT_RE.sub('', mnemonic)
        # Sanitize relative jumps.
        mnemonic = _RELADDR_RE.sub(r'\1', mnemonic)
        if not mnemonic or mnemonic.startswith('rex') or mnemonic.startswith('(bad)'):
            continue
        xtrie.parse_asm(mnemonic)
        bs = bytes(int(b, 16) for b in bs)
        state.insert(bs, mnemonic.strip())


def load_state(path: str, opts: xtrie.XTrieOpts) -> xtrie.XTrie:
    return xtrie.load_from_path(path, opts)


def dump_state(state: xtrie.XTrie, path: str) -> None:
    state.dump_to_path(path)


def mk_state(keep_asm: bool) -> xtrie.XTrie:
    return xtrie.mk_trie(keep_asm)
