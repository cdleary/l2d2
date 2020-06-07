import xtrie


mk_trie = lambda: xtrie.mk_trie(keep_asm=True)


def test_simple_insert_and_histo():
    t = mk_trie()
    t.insert(b'1', 'add')
    assert t.histo() == [0, 1]
    #assert t.lookup(b'1') == 'add'


def test_multi_insert_and_histo():
    t = mk_trie()
    t.insert(b'1', 'add')
    t.insert(b'22', 'sub')
    assert t.histo() == [0, 1, 1]
    t.insert(b'234', 'test')
    assert t.histo() == [0, 1, 1, 1]

    t.insert(b'24', 'cmp')
    assert t.histo() == [0, 1, 2, 1]

    minibatch = t.sample_nr_mb(4, 3)
    assert sorted(minibatch.sizes) == [1, 2, 2, 3]


def test_empty_sample():
    t = mk_trie()
    assert t.sample_nr(None) is None
    t.insert(b'1', 'add')
    assert t.sample_nr(None) == xtrie.PyRecord(b'1', 1, 'add')
    assert t.sample_nr(None) is None


def test_simple_sample():
    t = mk_trie()
    t.insert(b'1', 'add')
    t.insert(b'22', 'sub')
    avail = [xtrie.PyRecord(b'1', 1, 'add'), xtrie.PyRecord(b'22', 2, 'sub')]
    got0 = t.sample_nr(None)
    assert got0 in avail
    avail.remove(got0)
    assert t.sample_nr(None) in avail
    assert t.sample_nr(None) is None


def test_sample_length():
    t = mk_trie()
    t.insert(b'1', 'add')
    t.insert(b'22', 'sub')
    length = 3
    got0 = t.sample_nr(length)
    print(got0)
    assert (got0.bytes[:1] == b'1' and got0.length == 1) or (got0.bytes[:2] == b'22' and got0.length == 2)
    got1 = t.sample_nr(length)
    assert (got1.bytes[:1] == b'1' and got1.length == 1) or (got1.bytes[:2] == b'22' and got1.length == 2)
    assert t.sample_nr(length) is None
