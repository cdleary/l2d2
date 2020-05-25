import xtrie


def test_simple_insert_and_histo():
    t = xtrie.mk_trie()
    t.insert(b'1', 'foo')
    assert t.histo() == [0, 1]
    assert t.lookup(b'1') == 'foo'

def test_multi_insert_and_histo():
    t = xtrie.mk_trie()
    t.insert(b'1', 'foo')
    t.insert(b'22', 'bar')
    assert t.histo() == [0, 1, 1]
    t.insert(b'234', 'baz')
    assert t.histo() == [0, 1, 1, 1]

    t.insert(b'24', 'bat')
    assert t.histo() == [0, 1, 2, 1]

def test_empty_sample():
    t = xtrie.mk_trie()
    assert t.sample_nr(None) is None
    t.insert(b'1', 'foo')
    assert t.sample_nr(None) == (b'1', 1)
    assert t.sample_nr(None) is None

def test_simple_sample():
    t = xtrie.mk_trie()
    t.insert(b'1', 'foo')
    t.insert(b'22', 'bar')
    avail = [(b'1', 1), (b'22', 2)]
    got0 = t.sample_nr(None)
    assert got0 in avail
    avail.remove(got0)
    assert t.sample_nr(None) in avail
    assert t.sample_nr(None) is None

def test_sample_length():
    t = xtrie.mk_trie()
    t.insert(b'1', 'foo')
    t.insert(b'22', 'bar')
    length = 3
    got0, got0_len = t.sample_nr(length)
    assert (got0[:1] == b'1' and got0_len == 1) or (got0[:2] == b'22' and got0_len == 2)
    got1, got1_len = t.sample_nr(length)
    assert (got1[:1] == b'1' and got1_len == 1) or (got1[:2] == b'22' and got1_len == 2)
    assert t.sample_nr(length) is None
