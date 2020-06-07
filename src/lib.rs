#![recursion_limit="4096"]

use std::collections::HashSet;
use std::fs::File;

use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use serde::{Deserialize, Serialize};

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyBool;
use pyo3::wrap_pyfunction;
use pyo3::class::basic::CompareOp;
use pyo3::{PyErr, Python};

mod asm_parser;

// Represents a leaf in the x86 assembly trie; having a cached length (in
// bytes) and the assembly text associated with its byte sequence.
#[derive(Serialize, Deserialize, Clone)]
struct Terminal {
    length: u8,
    opcode: asm_parser::Opcode,
    asm: Option<String>,
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

    // Whether to store provided assembly in the trie.
    keep_asm: bool,
}

impl Trie {
    fn new(keep_asm: bool) -> Trie {
        Trie {
            root: mk_empty_interior(),
            total: 0,
            binaries: HashSet::new(),
            keep_asm: keep_asm,
        }
    }
}

#[pyclass]
struct MiniBatch {
    bytes: Vec<Vec<u8>>,
    sizes: Vec<u8>,
    opcodes: Vec<u16>,
}

#[pyclass]
struct XTrie {
    trie: Trie,
}

fn mk_empty_interior() -> Vec<TrieElem> {
    let mut result = vec![];
    for _i in 0..256 {
        result.push(TrieElem::Nothing);
    }
    return result;
}

fn all_nothing(v: &Vec<TrieElem>) -> bool {
    v.iter().all(|t| {
        if let TrieElem::Nothing = t {
            true
        } else {
            false
        }
    })
}

fn hexbytes(bytes: &[u8]) -> String {
    bytes
        .iter()
        .enumerate()
        .map(|(i, x)| format!("{:02x}", x) + if i + 1 == bytes.len() { "" } else { " " })
        .collect::<Vec<_>>()
        .concat()
}

// Returns whether the terminal was newly added (false if it was already
// present).
fn inserter(
    node: &mut Vec<TrieElem>,
    allbytes: &[u8],
    bytes: &[u8],
    asm: String,
    length: u8,
    keep_asm: bool,
) -> bool {
    assert!(bytes.len() > 0);
    let index = bytes[0];
    if bytes.len() == 1 {
        match &node[index as usize] {
            TrieElem::Nothing => (),
            TrieElem::Terminal(t) => {
                assert!(t.length == length);
                //assert!(
                //    t.asm == asm,
                //    "{} vs {}; bytes: {}",
                //    t.asm,
                //    asm,
                //    hexbytes(allbytes)
                //);
                return false;
            }
            TrieElem::Interior(_) => panic!(
                "Interior present where terminal is now provided at length: {}; bytes: {}; asm: {}",
                length,
                hexbytes(allbytes),
                asm
            ),
        }
        node[index as usize] = TrieElem::Terminal(Terminal {
            length: length,
            opcode: asm_parser::parse_opcode(&asm).unwrap(),
            asm: if keep_asm { Some(asm) } else { None },
        });
        return true;
    }
    if let TrieElem::Nothing = &node[index as usize] {
        node[index as usize] = TrieElem::Interior(mk_empty_interior())
    }
    match &mut node[index as usize] {
        TrieElem::Nothing => panic!(),
        TrieElem::Terminal(_) => panic!(),
        TrieElem::Interior(sub_node) => inserter(sub_node, allbytes, &bytes[1..], asm, length + 1, keep_asm),
    }
}

fn traverse<F>(node: &Vec<TrieElem>, f: &mut F) -> ()
where
    F: FnMut(&Terminal) -> (),
{
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
        return None;
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
fn sampler(
    node: &mut Vec<TrieElem>,
    mut current: Vec<u8>,
) -> (Option<(Vec<u8>, asm_parser::Opcode)>, bool) {
    // Collect all the indices that are present.
    let mut present: Vec<usize> = Vec::with_capacity(node.len());
    for (i, elem) in node.iter().enumerate() {
        match elem {
            TrieElem::Nothing => (),
            _ => present.push(i),
        }
    }
    assert!(!present.is_empty());

    // Determine an index to sample (from the compacted indices).
    let mut rng = thread_rng();
    let index: usize = *present.choose(&mut rng).unwrap();
    assert!(index <= 255);

    // Push that onto our current byte traversal.
    current.push(index as u8);

    let (result, sub_empty) = match &mut node[index] {
        TrieElem::Nothing => panic!("Should have filtered out Nothing indices."),
        // If we hit a terminal our result is the current byte traversal.
        TrieElem::Terminal(t) => (Some((current, t.opcode)), true),
        TrieElem::Interior(n) => sampler(n, current),
    };

    // If the sub-node is now empty from our sampling, we mark it as nothing.
    if sub_empty {
        node[index] = TrieElem::Nothing;
    }

    // Provide the result and whether this node is now empty.
    (result, present.len() == 1 && sub_empty)
}

#[pyclass]
#[derive(PartialEq, Clone)]
struct PyRecord {
    bytes: Vec<u8>,
    #[pyo3(get)]
    length: u8,
    #[pyo3(get)]
    opcode: u16,
}

#[pymethods]
impl PyRecord {
    #[new]
    fn new(bytes: Vec<u8>, length: u8, opcode: &str) -> PyResult<Self> {
        let opcode: u16 = match asm_parser::parse_opcode(opcode) {
            Some(opcode_enum) => opcode_enum as u16,
            None => return Err(PyErr::new::<exceptions::ValueError, _>(format!("Invalid opcode {:?}", opcode))),
        };
        Ok(PyRecord{bytes: bytes.clone(), length: length, opcode: opcode})
    }

    #[getter]
    fn bytes<'p>(&self, py: Python<'p>) -> PyResult<&'p PyBytes> {
        Ok(PyBytes::new(py, &self.bytes))
    }
}

#[pyproto]
impl pyo3::class::PyObjectProtocol for PyRecord {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Record{{bytes: {:?}, length: {}, opcode: {}}}", self.bytes, self.length, self.opcode))
    }
    fn __richcmp__(&self, other: PyRecord, op: CompareOp) -> PyResult<PyObject> {
        let gil = GILGuard::acquire();
        let py = gil.python();
        Ok(match op {
            CompareOp::Eq => PyBool::new(py, *self == other).to_object(py),
            CompareOp::Ne => PyBool::new(py, *self != other).to_object(py),
            _ => py.NotImplemented()
        })
    }
}

struct Record {
    bytes: Vec<u8>,
    length: u8,
    opcode: u16,
}

impl Record {
    fn to_py(&self) -> PyRecord {
        PyRecord{bytes: self.bytes.clone(), length: self.length, opcode: self.opcode}
    }
}

// "length" indicates the length to which we should pad the sampled bytes
// with random garbage. If we padded it with zeros then the length of the
// sample would be easy to determine by inspection of where the zeros
// start!
fn sample_nr(
    xt: &mut XTrie,
    length: Option<u8>,
) -> Option<Record> {
    if all_nothing(&xt.trie.root) {
        // Whole trie is empty.
        return None;
    }
    let (result, _) = sampler(&mut xt.trie.root, vec![]);
    match result {
        Some((mut v, opcode)) => {
            let orig_len = v.len();
            if let Some(len) = length {
                // Push random bytes on until we hit our target length.
                let mut rng = thread_rng();
                while v.len() < len as usize {
                    let b: u8 = rng.gen();
                    v.push(b)
                }
            }
            Some(Record{bytes: v, length: orig_len as u8, opcode: opcode as u16})
        }
        None => None,
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

    pub fn clone(&self) -> PyResult<XTrie> {
        Ok(XTrie {
            trie: self.trie.clone(),
        })
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
        self.trie.total += inserter(&mut self.trie.root, bytes, bytes, asm, 1, self.trie.keep_asm) as u64;
        Ok(())
    }

    //pub fn lookup(&self, bytes: &[u8]) -> PyResult<String> {
    //    let root = &self.trie.root;
    //    match resolver(root, bytes) {
    //        Some(t) => Ok(t.asm.clone()),
    //        None => Err(PyErr::new::<exceptions::KeyError, _>(
    //            "Could not find bytes in trie",
    //        )),
    //    }
    //}

    pub fn empty(&self) -> PyResult<bool> {
        Ok(all_nothing(&self.trie.root))
    }

    pub fn histo(&self) -> PyResult<Vec<i32>> {
        let mut histo = vec![];
        let mut add_to_histo = |t: &Terminal| {
            if histo.len() <= (t.length as usize) {
                histo.resize((t.length + 1) as usize, 0);
            }
            histo[t.length as usize] += 1;
        };
        traverse(&self.trie.root, &mut add_to_histo);
        Ok(histo)
    }

    pub fn sample_nr(&mut self, length: Option<u8>) -> PyResult<Option<PyRecord>> {
        Ok(sample_nr(self, length).map(|r| r.to_py()))
    }

    pub fn sample_nr_mb(
        &mut self,
        minibatch_size: u8,
        length: u8,
    ) -> PyResult<MiniBatch> {
        let mut mb = MiniBatch{bytes: vec![], sizes: vec![], opcodes: vec![]};
        for _ in 0..minibatch_size {
            match sample_nr(self, Some(length)) {
                Some(r) => {
                    mb.bytes.push(r.bytes);
                    mb.sizes.push(r.length);
                    mb.opcodes.push(r.opcode);
                }
                None => {
                    mb.bytes.push(vec![0u8; length as usize]);
                    mb.sizes.push(0);
                    mb.opcodes.push(0);
                }
            }
        }
        Ok(mb)
    }
}

#[pyfunction]
fn parse_asm(s: &str) -> PyResult<()> {
    asm_parser::parse_opcode(s);
    Ok(())
}

#[pyfunction]
fn get_opcode_count() -> PyResult<u32> {
    Ok(asm_parser::get_opcode_count())
}

#[pyfunction]
fn mk_trie(keep_asm: bool) -> PyResult<XTrie> {
    Ok(XTrie { trie: Trie::new(keep_asm) })
}

#[pyfunction]
fn load_from_path(path: &str) -> PyResult<XTrie> {
    let file = File::open(path)?;
    let trie = bincode::deserialize_from(file).unwrap();
    Ok(XTrie { trie })
}

/// This module is a python module implemented in Rust.
#[pymodule]
fn xtrie(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(mk_trie))?;
    m.add_wrapped(wrap_pyfunction!(load_from_path))?;
    m.add_wrapped(wrap_pyfunction!(parse_asm))?;
    m.add_wrapped(wrap_pyfunction!(get_opcode_count))?;
    m.add_class::<XTrie>()?;
    m.add_class::<PyRecord>()?;

    Ok(())
}
