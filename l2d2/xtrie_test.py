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
