import pickle
import itertools
import subprocess as subp
import optparse
import os
from datetime import datetime

import ingest


def find_binaries(dirpath):
    output = subp.check_output(['find', dirpath, '-type', 'f', '-executable']).decode('utf-8')
    candidates = output.splitlines()
    results = []
    for candidate in candidates:
        if b'ELF 64-bit' in subp.check_output(['file', candidate]):
            results.append(candidate)
    return results


def main():
    parser = optparse.OptionParser()
    parser.add_option('--path', default='/tmp/x86.pickle', help='Path to load/store results')
    parser.add_option('--no-load', dest='load', default=True, action='store_false', help='Avoid loading existing path from disk')
    parser.add_option('--no-ingest', dest='ingest', default=True, action='store_false', help='Whether to ingest binaries on this system')
    parser.add_option('--ingest-limit', default=None, type='int', help='Limit on number of binaries to ingest; e.g. for testing')
    opts, parse = parser.parse_args()

    if os.path.exists(opts.path) and opts.load:
        state = ingest.load_state(opts.path)
    else:
        state = ingest.State()

    if opts.ingest:
        ingest_start = datetime.now()

        for binpath in itertools.chain.from_iterable(find_binaries(path) for path in os.getenv('PATH').split(':')):
            ingest.ingest_binary(state, binpath)
            if opts.ingest_limit and len(state.binaries) >= opts.ingest_limit:
                print('... stopping at ingest limit')
                break

        ingest_end = datetime.now()
        print('ingest time:', ingest_end - ingest_start)
        
        with open(opts.path, 'wb') as f:
            pickle.dump(state, f)

    maxdepth, seen_terminals, histo = ingest.print_sparsely(state.data, do_print=False)
    print('maxdepth:      ', maxdepth)
    print('seen terminals:', seen_terminals)
    print('total:         ', state.total)
    print('histo:         ', list(zip(range(len(histo)), histo)))
    print('binaries:      ', len(state.binaries))
    print('1B terminals:  ', sum(1 for item in state.data if isinstance(item, int)))
    print('1B productions:', sum(1 for item in state.data if isinstance(item, list)))


if __name__ == '__main__':
    main()
