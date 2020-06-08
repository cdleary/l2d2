import pickle
import itertools
import subprocess as subp
import optparse
import os
import sys
from datetime import datetime

import ingest
import xtrie


def find_binaries(dirpath):
    if not (os.path.exists(dirpath) and os.path.isdir(dirpath)):
        return []
    output = subp.check_output([
        'find', dirpath, '-type', 'f', '-executable']).decode('utf-8')
    candidates = output.splitlines()
    results = []
    for candidate in candidates:
        if b'ELF 64-bit' in subp.check_output(['file', candidate]):
            results.append(candidate)
    return results


def _do_ingest(opts, state: xtrie.XTrie) -> None:
    print('ingesting...', file=sys.stderr)
    ingest_start = datetime.now()

    bin_path = opts.bin_path or os.getenv('PATH')
    for binpath in itertools.chain.from_iterable(
            find_binaries(path) for path in bin_path.split(':')):
        if opts.ingest_limit and state.binary_count >= opts.ingest_limit:
            print(f'... stopping at ingest limit {opts.ingest_limit}; {state.binary_count} binaries have been ingested', file=sys.stderr)
            break
        if not state.try_add_binary(binpath):
            print(f'... already seen {binpath:<60}', file=sys.stderr)
            continue
        ingest.ingest_binary(state, binpath)

    ingest_end = datetime.now()
    print('... ingested for', ingest_end - ingest_start, file=sys.stderr)

    print('... dumping state', file=sys.stderr)
    dump_start = datetime.now()
    ingest.dump_state(state, opts.trie_path)
    dump_end = datetime.now()
    print('... done dumping state in', dump_end-dump_start, file=sys.stderr)


def _do_load_state(opts) -> xtrie.XTrie:
    print('loading state...', file=sys.stderr)
    load_start = datetime.now()
    state = ingest.load_state(opts.trie_path)
    load_end = datetime.now()
    print('load time:', load_end-load_start, file=sys.stderr)
    return state


def main():
    parser = optparse.OptionParser()
    parser.add_option('--bin-path', default=None,
                      help='Override for $PATH environment variable')
    parser.add_option('--trie-path', default='/tmp/x86.state',
                      help='Path to load/store results')
    parser.add_option('--no-load', dest='load', default=True,
                      action='store_false',
                      help='Avoid loading existing path from disk')
    parser.add_option('--avoid-dups', action='store_true', default=False,
                      help='Avoid processing already-seen binaries (via hash).')
    parser.add_option(
        '--no-ingest', dest='ingest', default=True, action='store_false',
        help='Whether to ingest binaries on this system')
    parser.add_option(
        '--ingest-limit', default=None, type='int',
        help='Limit on number of binaries to ingest; e.g. for testing')
    opts, parse = parser.parse_args()

    known_opcodes = xtrie.get_opcode_count()
    print(f'{known_opcodes} known opcodes')

    if os.path.exists(opts.trie_path) and opts.load:
        state = _do_load_state(opts)
    else:
        state = ingest.mk_state(keep_asm=False)

    if opts.ingest:
        _do_ingest(opts, state)

    print('... histogramming', file=sys.stderr)
    histo = state.histo()
    maxdepth = max(k for k, v in enumerate(histo) if v != 0) if histo else 0
    seen_terminals = sum(histo)
    print('maxdepth:      ', maxdepth, file=sys.stderr)
    print('seen terminals:', seen_terminals, file=sys.stderr)
    print('histo:         ', list(zip(range(len(histo)), histo)), file=sys.stderr)
    print('binaries:      ', state.binary_count, file=sys.stderr)


if __name__ == '__main__':
    main()
