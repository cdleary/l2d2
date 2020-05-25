use std::collections::HashSet;
use std::fs::File;

use serde::{Serialize, Deserialize};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::{Python, PyErr};
use pyo3::exceptions;

// Represents a leaf in the x86 assembly trie; having a cached length (in
// bytes) and the assembly text associated with its byte sequence.
#[derive(Serialize, Deserialize)]
struct Terminal {
    length: u8,
    asm: String,
}

// Member of the trie; for any given byte we can have a node that leads to more
// terminals, a terminal, or not have any entry yet.
#[derive(Serialize, Deserialize)]
enum TrieElem {
    Interior(Vec<TrieElem>),
    Terminal(Terminal),
    Nothing,
}

#[derive(Serialize, Deserialize)]
struct Trie {
    root: Vec<TrieElem>,
    total: u64,

    // Hashes of binaries already processed.
    binaries: HashSet<String>,
}

impl Trie {
    fn new() -> Trie {
        Trie{root: mk_empty_interior(), total: 0, binaries: HashSet::new()}
    }
}

#[pyclass]
struct XTrie {
    trie: Trie
}

fn mk_empty_interior() -> Vec<TrieElem> {
    let mut result = vec![];
    for _i in 0..256 {
        result.push(TrieElem::Nothing);
    }
    return result;
}

fn hexbytes(bytes: &[u8]) -> String {
    bytes.iter().enumerate()
         .map(|(i, x)| format!("{:02x}", x) + if i+1 == bytes.len() { "" } else { " " })
         .collect::<Vec<_>>()
         .concat()
}

// Returns whether the terminal was newly added (false if it was already
// present).
fn inserter(node: &mut Vec<TrieElem>, allbytes: &[u8], bytes: &[u8], asm: String, length: u8) -> bool {
    assert!(bytes.len() > 0);
    let index = bytes[0];
    if bytes.len() == 1 {
        match &node[index as usize] {
            TrieElem::Nothing => (),
            TrieElem::Terminal(t) => {
                assert!(t.length == length);
                assert!(t.asm == asm, "{} vs {}; bytes: {}", t.asm, asm, hexbytes(allbytes));
                return false;
            }
            TrieElem::Interior(_) => panic!("Interior present where terminal is now provided at length: {}; bytes: {}; asm: {}", length, hexbytes(allbytes), asm),
        }
        node[index as usize] = TrieElem::Terminal(Terminal{length: length, asm: asm});
        return true;
    }
    if let TrieElem::Nothing = &node[index as usize] {
        node[index as usize] = TrieElem::Interior(mk_empty_interior())
    }
    match &mut node[index as usize] {
        TrieElem::Nothing => panic!(),
        TrieElem::Terminal(_) => panic!(),
        TrieElem::Interior(sub_node) => inserter(sub_node, allbytes, &bytes[1..], asm, length+1)
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
impl XTrie {
    #[getter]
    fn get_total(&self) -> PyResult<u64> {
        Ok(self.trie.total)
    }

    #[getter]
    fn get_binary_count(&self) -> PyResult<usize> {
        Ok(self.trie.binaries.len())
    }

    // Returns true iff the key was freshly inserted.
    pub fn try_add_binary(&mut self, key: String) -> PyResult<bool> {
        Ok(self.trie.binaries.insert(key))
    }

    pub fn has_binary(&self, key: &str) -> PyResult<bool> {
        Ok(self.trie.binaries.contains(key))
    }

    pub fn dump_to_path(&self, path: &str) -> PyResult<()> {
        let file = File::create(path)?;
        bincode::serialize_into(file, &self.trie).unwrap();
        Ok(())
    }

    pub fn insert(&mut self, bytes: &[u8], asm: String) -> PyResult<()> {
        self.trie.total += inserter(&mut self.trie.root, bytes, bytes, asm, 1) as u64;
        Ok(())
    }

    pub fn lookup(&self, bytes: &[u8]) -> PyResult<String> {
        let root = &self.trie.root;
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
        traverse(&self.trie.root, &mut add_to_histo);
        Ok(histo)
    }
}

#[pyfunction]
fn mk_trie() -> PyResult<XTrie> {
    Ok(XTrie{trie: Trie::new()})
}

#[pyfunction]
fn load_from_path(path: &str) -> PyResult<XTrie> {
    let file = File::open(path)?;
    let trie = bincode::deserialize_from(file).unwrap();
    Ok(XTrie{trie})
}

/// This module is a python module implemented in Rust.
#[pymodule]
fn xtrie(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(mk_trie))?;
    m.add_wrapped(wrap_pyfunction!(load_from_path))?;
    m.add_class::<XTrie>()?;

    Ok(())
}
