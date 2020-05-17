import os
import pickle
import pprint
import subprocess as subp
import re


PATH = '/tmp/x86.pickle'


state = dict(
    data = [None] * 256,
    total = 0,
    binaries = set(),
)


if os.path.exists(PATH):
    with open(PATH, 'rb') as f:
        state = pickle.load(f)


def try_add_binary(path: str) -> bool:
    sha = subp.check_output(['sha256sum', path]).strip()
    if sha in state['binaries']:
        return False
    state['binaries'].add(sha)
    return True


def ingest_binary(path: str) -> None:
    if not try_add_binary(path):
        print(f'... already seen {path:<60}')
        return
    print(f'... ingesting {path:<60} (before: {state["total"]:12,})')
    output = subp.check_output(['objdump', '-d', path, '-j', '.text']).decode('utf-8')
    lines = output.splitlines()

    for line in lines:
        m = re.match('\s*\d+:\s+(?P<bytes>([a-f\d]{2} )+)(?P<mnemonic>.*)$', line)
        if not m:
            continue
        if not m.group('mnemonic').strip():
            continue
        bs = m.group('bytes')
        node = state['data']
        bs = bs.split()
        for byteno, b in enumerate(int(b, 16) for b in bs):
            if len(bs) == byteno+1:  # Note the length of the terminal.
                assert node[b] == len(bs) or node[b] is None, (byteno, line)
                state['total'] += node[b] != len(bs)
                node[b] = len(bs)
            elif node[b] is None:
                node[b] = [None] * 256
                node = node[b]
            else:
                node = node[b]
                assert isinstance(node, list), (line)


def print_sparsely(d, depth=0, do_print=True):
    indent_str = ' ' * (2 * depth)
    maxdepth = depth
    seen_terminals = 0
    histo = [0] * 16
    for i, item in enumerate(d):
        if item is None:
            continue
        if isinstance(item, int):
            if do_print:
                print(f'{indent_str}{i:02x}: {item}')
            seen_terminals += 1
            histo[item] += 1
        else:
            if do_print:
                print(f'{indent_str}{i:02x}:')
            sub_maxdepth, sub_seen_terminals, sub_histo = print_sparsely(item, depth+1, do_print=do_print)
            maxdepth = max(sub_maxdepth, maxdepth)
            seen_terminals += sub_seen_terminals
            histo = [h+sh for h, sh in zip(histo, sub_histo)]
    return maxdepth, seen_terminals, histo


def find_binaries(dirpath):
    output = subp.check_output(['find', dirpath, '-type', 'f', '-executable']).decode('utf-8')
    candidates = output.splitlines()
    results = []
    for candidate in candidates:
        if b'ELF 64-bit' in subp.check_output(['file', candidate]):
            results.append(candidate)
    return results


#for path in ('/bin', '/sbin', '/usr/bin', '/usr/local/bin'):
#    for binpath in find_binaries(path):
#        ingest_binary(binpath)
#
#with open(PATH, 'wb') as f:
#    pickle.dump(state, f)

maxdepth, seen_terminals, histo = print_sparsely(state['data'], do_print=False)
print('maxdepth:      ', maxdepth)
print('seen terminals:', seen_terminals)
print('total:         ', state['total'])
print('histo:         ', list(zip(range(len(histo)), histo)))
print('binaries:      ', len(state['binaries']))
print('1B terminals:  ', sum(1 for item in state['data'] if isinstance(item, int)))
print('1B productions:', sum(1 for item in state['data'] if isinstance(item, list)))
