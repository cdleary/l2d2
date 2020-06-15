use rand::thread_rng;

extern crate test;

use rand::Rng;
use rand::seq::SliceRandom;

use crate::trie;

/// A record; e.g. as sampled from the trie.
pub struct Record {
    pub bytes: Vec<u8>,
    pub length: u8,
    pub opcode: u16,
    pub asm: Option<String>
}

pub struct MiniBatch {
    pub bytes: Vec<Vec<u8>>,
    pub length: Vec<u8>,
    pub opcode: Vec<u16>,
    pub asm: Vec<Option<String>>,

    // (batch_size, byteno, float_data)
    // e.g. perhaps (32, 8, 15) for 8-byte-length input data where we expand
    // into 15 floats.
    pub floats: Vec<Vec<Vec<f32>>>,
}

impl MiniBatch {
    fn new(minibatch_size: usize) -> MiniBatch {
        MiniBatch{
            bytes: Vec::with_capacity(minibatch_size),
            length: Vec::with_capacity(minibatch_size),
            opcode: Vec::with_capacity(minibatch_size),
            asm: Vec::with_capacity(minibatch_size),
            floats: vec![],
        }
    }
}

/// Randomly selects a terminal from within "node".
///
/// Returns the terminal, if present, and whether this node is now empty, since
/// we sampled without replacement. If the node is empty, the parent node should
/// make a note of it to avoid recursing into it in future samplings.
fn sampler(
    node: &mut Vec<trie::TrieElem>,
    mut current: Vec<u8>,
) -> (Option<Record>, bool) {
    // Collect all the indices that are present.
    let mut present: Vec<usize> = Vec::with_capacity(node.len());
    for (i, elem) in node.iter().enumerate() {
        match elem {
            trie::TrieElem::Nothing => (),
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
        trie::TrieElem::Nothing => panic!("Should have filtered out Nothing indices."),
        // If we hit a terminal our result is the current byte traversal.
        trie::TrieElem::Terminal(t) => {
            let len = current.len();
            (Some(Record{bytes: current, length: len as u8, opcode: t.opcode as u16, asm: t.asm.clone()}), true)
        }
        trie::TrieElem::Interior(n) => sampler(n, current),
    };

    // If the sub-node is now empty from our sampling, we mark it as nothing.
    if sub_empty {
        node[index] = trie::TrieElem::Nothing;
    }

    // Provide the result and whether this node is now empty.
    (result, present.len() == 1 && sub_empty)
}

/// "target_length" indicates the length to which we should pad the sampled bytes
/// with random garbage. If we padded it with zeros then the length of the
/// sample would be easy to determine by inspection of where the zeros
/// start!
pub fn sample_nr(
    node: &mut Vec<trie::TrieElem>,
    target_length: Option<u8>,
) -> Option<Record> {
    if trie::all_nothing(node) {
        // Whole trie is empty.
        return None;
    }
    let (result, _now_empty): (Option<Record>, bool) = sampler(node, vec![]);
    match result {
        Some(mut r) => {
            if let Some(len) = target_length {
                // Push random bytes on until we hit our target length.
                let mut rng = thread_rng();
                while r.bytes.len() < len as usize {
                    let b: u8 = rng.gen();
                    r.bytes.push(b)
                }
            }
            Some(r)
        }
        None => None,
    }
}

pub fn sample_nr_mb(
    node: &mut Vec<trie::TrieElem>,
    target_length: u8,
    minibatch_size: usize
) -> MiniBatch {
    let mut mb = MiniBatch::new(minibatch_size);
    for _ in 0..minibatch_size {
        match sample_nr(node, Some(target_length)) {
            Some(r) => {
                mb.bytes.push(r.bytes);
                mb.length.push(r.length);
                mb.opcode.push(r.opcode);
                mb.asm.push(r.asm);
            }
            None => {
                mb.bytes.push(vec![0u8; target_length as usize]);
                mb.length.push(0);
                mb.opcode.push(0);
                mb.asm.push(None);
            }
        }
    }
    mb
}

#[cfg(test)]
mod tests {

use std::fs::File;
use super::*;
use test::Bencher;

#[bench]
fn bench_sample_nr(b: &mut Bencher) {
    let file = File::open("/tmp/x86.state").unwrap();
    let mut trie: trie::Trie = bincode::deserialize_from(file).unwrap();
    b.iter(|| test::black_box(sample_nr(&mut trie.root, Some(15))));
}

#[bench]
fn bench_sample_nr_mb_128(b: &mut Bencher) {
    let file = File::open("/tmp/x86.state").unwrap();
    let mut trie: trie::Trie = bincode::deserialize_from(file).unwrap();
    b.iter(|| test::black_box(sample_nr_mb(&mut trie.root, 15, 128)));
}

}
