#![feature(test)]

extern crate test;

use std::fs::File;

use pyo3::prelude::*;
use pyo3::PyTraverseError;
use pyo3::gc::{PyGCProtocol, PyVisit};
use pyo3::exceptions;
use pyo3::types::{PyBytes, PyBool};
use pyo3::wrap_pyfunction;
use pyo3::class::basic::CompareOp;
use pyo3::{PyErr, Python};

use numpy::{PyArray, PyArray1, PyArray2};

mod asm_parser;
mod trie;
mod trie_sampler;

#[pyclass]
#[derive(Clone)]
struct XTrieOpts {
    pub byte: bool,
    pub nibbles: bool,
    pub crumbs: bool,
    pub bits: bool,
    pub keep_asm: bool,
}

impl XTrieOpts {
    fn new() -> XTrieOpts {
        return XTrieOpts{byte: true, nibbles: true, crumbs: true, bits: true, keep_asm: false}
    }
}

#[pyclass]
struct XTrie {
    trie: trie::Trie,
    opts: XTrieOpts,
}

impl trie_sampler::Record {
    fn to_py(&self) -> PyRecord {
        PyRecord{bytes: self.bytes.clone(), length: self.length, opcode: self.opcode, asm: self.asm.clone()}
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
    #[pyo3(get)]
    asm: Option<String>,
}

fn byte_to_float(x: u8) -> f32 {
    f32::from_bits((127 << 23) | (x as u32))
}

fn bytes_to_floats(bytes: &[u8], opts: &XTrieOpts) -> Vec<f32> {
    let mut result = Vec::with_capacity((opts.byte as usize) + 2 * (opts.nibbles as usize) + 4 * (opts.crumbs as usize) + 8 * (opts.bits as usize));
    for &byte in bytes {
        if opts.byte {
            result.push(byte_to_float(byte));
        }
        if opts.nibbles {
            result.push(byte_to_float(byte >> 4));
            result.push(byte_to_float(byte & 0xff));
        }
        if opts.crumbs {
            result.push(byte_to_float((byte >> 6) & 0x3));
            result.push(byte_to_float((byte >> 4) & 0x3));
            result.push(byte_to_float((byte >> 2) & 0x3));
            result.push(byte_to_float((byte >> 0) & 0x3));
        }
        if opts.bits {
            for i in 0..8 {
                result.push(byte_to_float((byte >> i) & 0x1));
            }
        }
    }
    result
}

#[pymethods]
impl PyRecord {
    #[new]
    fn new(bytes: Vec<u8>, length: u8, opcode: &str, asm: Option<String>) -> PyResult<Self> {
        let opcode: u16 = match asm_parser::parse_opcode(opcode) {
            Some(opcode_enum) => opcode_enum as u16,
            None => return Err(PyErr::new::<exceptions::ValueError, _>(format!("Invalid opcode {:?}", opcode))),
        };
        Ok(PyRecord{bytes: bytes.clone(), length: length, opcode: opcode, asm: asm.clone()})
    }

    #[getter]
    fn bytes<'p>(&self, py: Python<'p>) -> PyResult<&'p PyBytes> {
        Ok(PyBytes::new(py, &self.bytes))
    }
}

/// Vector-scaled version of a PyRecord.
#[pyclass]
struct PyMiniBatch {
    mb: trie_sampler::MiniBatch,
    #[pyo3(get)]
    floats: PyObject,
    #[pyo3(get)]
    lengths: PyObject,
}

#[pyproto]
impl PyGCProtocol for PyMiniBatch {
    fn __traverse__(&self, visit: PyVisit) -> Result<(), PyTraverseError> {
        visit.call(&self.floats)?;
        visit.call(&self.lengths)?;
        Ok(())
    }

    fn __clear__(&mut self) {
        let gil = GILGuard::acquire();
        let py = gil.python();
        py.release(&self.floats);
        py.release(&self.lengths);
    }
}

#[pymethods]
impl PyMiniBatch {
    #[getter]
    fn opcodes<'p>(&self, py: Python<'p>) -> PyResult<&'p PyArray1<u16>> {
        Ok(PyArray::from_vec(py, self.mb.opcode.clone()))
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
            opts: self.opts.clone(),
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
        py: Python,
    ) -> PyResult<PyMiniBatch> {
        let mut mb = trie_sampler::sample_nr_mb(&mut self.trie.root, length, minibatch_size);
        for bytes in &mb.bytes {
            mb.floats.push(bytes_to_floats(&bytes, &self.opts));
        }
        let floats = PyArray::from_vec2(py, &mb.floats)?;
        let lengths = PyArray::from_vec(py, mb.length.clone());
        Ok(PyMiniBatch{mb, floats: floats.to_object(py), lengths: lengths.to_object(py)})
    }

    pub fn nop<'p>(&self, py: Python<'p>) -> PyResult<(&'p PyArray2<f32>, &'p PyArray1<u16>)> {
        Ok((PyArray2::zeros(py, [128, 15], false), PyArray1::zeros(py, [128], false)))
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
fn mk_trie(opts: XTrieOpts) -> PyResult<XTrie> {
    Ok(XTrie { trie: trie::Trie::new(opts.keep_asm), opts: opts })
}

#[pyfunction]
fn load_from_path(path: &str, opts: XTrieOpts) -> PyResult<XTrie> {
    let file = File::open(path)?;
    let trie = bincode::deserialize_from(file).unwrap();
    Ok(XTrie { trie, opts })
}

#[pyfunction]
fn mk_opts() -> PyResult<XTrieOpts> {
    Ok(XTrieOpts::new())
}

/// This module is a python module implemented in Rust.
#[pymodule]
fn xtrie(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(mk_trie))?;
    m.add_wrapped(wrap_pyfunction!(mk_opts))?;
    m.add_wrapped(wrap_pyfunction!(load_from_path))?;
    m.add_wrapped(wrap_pyfunction!(parse_asm))?;
    m.add_wrapped(wrap_pyfunction!(get_opcode_count))?;
    m.add_class::<XTrieOpts>()?;
    m.add_class::<XTrie>()?;
    m.add_class::<PyRecord>()?;

    Ok(())
}

#[cfg(test)]
mod tests {

use super::*;
use test::Bencher;

#[bench]
fn py_bench_nop(b: &mut Bencher) {
    let mut xt: XTrie = load_from_path("/tmp/x86.state", mk_opts().unwrap()).unwrap();
    let gil = GILGuard::acquire();
    let py = gil.python();
    b.iter(|| test::black_box(xt.nop(py)));
}

#[bench]
fn py_bench_sample_nr_mb_1(b: &mut Bencher) {
    let mut xt: XTrie = load_from_path("/tmp/x86.state", mk_opts().unwrap()).unwrap();
    let gil = GILGuard::acquire();
    let py = gil.python();
    b.iter(|| test::black_box(xt.sample_nr_mb(1, 15, py)));
}

#[bench]
fn py_bench_sample_nr_mb_32(b: &mut Bencher) {
    let mut xt: XTrie = load_from_path("/tmp/x86.state", mk_opts().unwrap()).unwrap();
    let gil = GILGuard::acquire();
    let py = gil.python();
    b.iter(|| test::black_box(xt.sample_nr_mb(32, 15, py)));
}

#[bench]
fn py_bench_sample_nr_mb_128(b: &mut Bencher) {
    let mut xt: XTrie = load_from_path("/tmp/x86.state", mk_opts().unwrap()).unwrap();
    let gil = GILGuard::acquire();
    let py = gil.python();
    b.iter(|| test::black_box(xt.sample_nr_mb(128, 15, py)));
}

}
