use std::fs::File;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyBool;
use pyo3::wrap_pyfunction;
use pyo3::class::basic::CompareOp;
use pyo3::{PyErr, Python};

mod asm_parser;
mod trie;
mod trie_sampler;

#[pyclass]
struct XTrie {
    trie: trie::Trie,
}

impl trie_sampler::Record {
    fn to_py(&self) -> PyRecord {
        PyRecord{bytes: self.bytes.clone(), length: self.length, opcode: self.opcode}
    }
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

/// Vector-scaled version of a PyRecord.
#[pyclass]
struct MiniBatch {
    bytes: Vec<Vec<u8>>,
    sizes: Vec<u8>,
    opcodes: Vec<u16>,
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
        self.trie.total += trie::inserter(&mut self.trie.root, bytes, bytes, asm, 1, self.trie.keep_asm) as u64;
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
        Ok(trie::all_nothing(&self.trie.root))
    }

    pub fn histo(&self) -> PyResult<Vec<i32>> {
        let mut histo = vec![];
        let mut add_to_histo = |t: &trie::Terminal| {
            if histo.len() <= (t.length as usize) {
                histo.resize((t.length + 1) as usize, 0);
            }
            histo[t.length as usize] += 1;
        };
        trie::traverse(&self.trie.root, &mut add_to_histo);
        Ok(histo)
    }

    pub fn sample_nr(&mut self, length: Option<u8>) -> PyResult<Option<PyRecord>> {
        Ok(trie_sampler::sample_nr(&mut self.trie.root, length).map(|r| r.to_py()))
    }

    pub fn sample_nr_mb(
        &mut self,
        minibatch_size: u8,
        length: u8,
    ) -> PyResult<MiniBatch> {
        let mut mb = MiniBatch{bytes: vec![], sizes: vec![], opcodes: vec![]};
        for _ in 0..minibatch_size {
            match trie_sampler::sample_nr(&mut self.trie.root, Some(length)) {
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
    Ok(XTrie { trie: trie::Trie::new(keep_asm) })
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
