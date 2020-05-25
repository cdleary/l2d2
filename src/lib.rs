use std::collections::HashSet;
use std::fs::File;

use rand::Rng;
use rand::thread_rng;
use rand::seq::SliceRandom;
use serde::{Serialize, Deserialize};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::{Python, PyErr};
use pyo3::exceptions;
use pyo3::types::PyBytes;

// Represents a leaf in the x86 assembly trie; having a cached length (in
// bytes) and the assembly text associated with its byte sequence.
#[derive(Serialize, Deserialize, Clone)]
struct Terminal {
    length: u8,
    asm: String,
}

// Member of the trie; for any given byte we can have a node that leads to more
// terminals, a terminal, or not have any entry yet.
#[derive(Serialize, Deserialize, Clone)]
enum TrieElem {
    Interior(Vec<TrieElem>),
    Terminal(Terminal),
    Nothing,
}

#[derive(Serialize, Deserialize, Clone)]
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

fn all_nothing(v: &Vec<TrieElem>) -> bool {
    v.iter().all(|t| if let TrieElem::Nothing = t { true } else { false })
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

// Randomly selects a terminal from within "node".
//
// Returns the terminal, if present, and whether this node is now empty, since
// we sampled without replacement. If the node is empty, the parent node should
// make a note of it to avoid recursing into it in future samplings.
fn sampler(node: &mut Vec<TrieElem>, mut current: Vec<u8>) -> (Option<Vec<u8>>, bool) {
    let mut present: Vec<usize> = Vec::with_capacity(node.len());
    for (i, elem) in node.iter().enumerate() {
        match elem {
            TrieElem::Nothing => (),
            _ => present.push(i),
        }
    }
    assert!(!present.is_empty());

    let mut rng = thread_rng();
    let index: usize = *present.choose(&mut rng).unwrap();
    assert!(index <= 255);
    current.push(index as u8);
    let (result, sub_empty) = match &mut node[index] {
        TrieElem::Nothing => panic!("Should have filtered out Nothing indices."),
        TrieElem::Terminal(_) => {
            (Some(current), true)
        }
        TrieElem::Interior(n) => sampler(n, current),
    };
    if sub_empty {
        node[index] = TrieElem::Nothing;
    }
    (result, present.len() == 1 && sub_empty)
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

    pub fn clone(&self) -> PyResult<XTrie> {
        Ok(XTrie{trie: self.trie.clone()})
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

    pub fn empty(&self) -> PyResult<bool> {
        Ok(all_nothing(&self.trie.root))
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

    // "length" indicates the length to which we should pad the sampled bytes
    // with random garbage. If we padded it with zeros then the length of the
    // sample would be easy to determine by inspection of where the zeros
    // start!
    pub fn sample_nr<'p>(&mut self, py: Python<'p>, length: Option<u8>) -> PyResult<Option<(&'p PyBytes, u8)>> {
        if all_nothing(&self.trie.root) {
            return Ok(None)
        }
        let (result, _) = sampler(&mut self.trie.root, vec![]);
        match result {
            Some(mut v) => {
                let orig_len = v.len();
                if let Some(len) = length {
                    let mut rng = thread_rng();
                    while v.len() < len as usize {
                        let b: u8 = rng.gen();
                        v.push(b)
                    }
                }
                Ok(Some((PyBytes::new(py, &v), orig_len as u8)))
            }
            None => Ok(None),
        }
    }

    pub fn sample_nr_mb<'p>(&mut self, py: Python<'p>, minibatch_size: u8, length: u8) -> PyResult<(Vec<&'p PyBytes>, Vec<u8>)> {
        let mut bytes = vec![];
        let mut sizes = vec![];
        for _ in 0..minibatch_size {
            match self.sample_nr(py, Some(length)).unwrap() {
                Some((bs, len)) => { bytes.push(bs); sizes.push(len) },
                None => {
                    bytes.push(PyBytes::new(py, &vec![0u8; length as usize]));
                    sizes.push(0);
                }
            }
        }
        Ok((bytes, sizes))
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
