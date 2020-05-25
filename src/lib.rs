use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::{Python, PyErr};
use pyo3::exceptions;

//enum TrieElem {
//    Node(Box<[256; TrieElem]>),
//    Terminal({length: u8, mnemonic: String}),
//    Dne,
//}

struct Terminal {
    length: u8,
    asm: String,
}

enum TrieElem {
    Interior(Vec<TrieElem>),
    Terminal(Terminal),
    Nothing,
}

#[pyclass]
struct Trie {
    root: Vec<TrieElem>
}

fn mk_empty_interior() -> Vec<TrieElem> {
    let mut result = vec![];
    for _i in 0..256 {
        result.push(TrieElem::Nothing);
    }
    return result;
}

fn inserter(node: &mut Vec<TrieElem>, bytes: &[u8], asm: String, length: u8) -> () {
    assert!(bytes.len() > 0);
    let index = bytes[0];
    if bytes.len() == 1 {
        match &node[index as usize] {
            TrieElem::Nothing => (),
            TrieElem::Terminal(t) => {
                assert!(t.length == length);
                assert!(t.asm == asm);
                return;
            }
            TrieElem::Interior(_) => panic!("Interior present where terminal is now provided at length: {}", length),
        }
        node[index as usize] = TrieElem::Terminal(Terminal{length: length, asm: asm});
    } else {
        if let TrieElem::Nothing = &node[index as usize] {
            node[index as usize] = TrieElem::Interior(mk_empty_interior())
        }
        match &mut node[index as usize] {
            TrieElem::Nothing => panic!(),
            TrieElem::Terminal(_) => panic!(),
            TrieElem::Interior(sub_node) => inserter(sub_node, &bytes[1..], asm, length+1)
        }
    }
}

fn traverse<F>(node: &Vec<TrieElem>, f: &mut F) -> ()
    where F : FnMut(&Terminal) -> () {
    for t in node {
        match t {
            TrieElem::Interior(node) => traverse(node, f),
            TrieElem::Terminal(t) => f(t),
            TrieElem::Nothing => (),
        }
    }
}

fn resolver<'a>(node: &'a Vec<TrieElem>, bytes: &[u8]) -> Option<&'a Terminal> {
    if bytes.len() == 0 {
        return None
    }
    match &node[bytes[0] as usize] {
        TrieElem::Nothing => None,
        TrieElem::Terminal(t) => Some(t),
        TrieElem::Interior(n) => resolver(&n, &bytes[1..]),
    }
}

#[pymethods]
impl Trie {
    pub fn insert(&mut self, bytes: &[u8], asm: String) -> PyResult<()> {
        inserter(&mut self.root, bytes, asm, 1);
        Ok(())
    }

    pub fn lookup(&self, bytes: &[u8]) -> PyResult<String> {
        let root = &self.root;
        match resolver(root, bytes) {
            Some(t) => Ok(t.asm.clone()),
            None => Err(PyErr::new::<exceptions::KeyError, _>("Could not find bytes in trie"))
        }
    }

    pub fn histo(&self) -> PyResult<Vec<i32>> {
        let mut histo = vec![];
        let mut add_to_histo = |t: &Terminal| {
            if histo.len() <= (t.length as usize) {
                histo.resize((t.length+1) as usize, 0);
            }
            histo[t.length as usize] += 1;
        };
        traverse(&self.root, &mut add_to_histo);
        Ok(histo)
    }
}

#[pyfunction]
fn mk_trie() -> PyResult<Trie> {
    Ok(Trie{root: mk_empty_interior()})
}

/// This module is a python module implemented in Rust.
#[pymodule]
fn xtrie(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(mk_trie))?;

    Ok(())
}
