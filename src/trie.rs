use std::collections::HashSet;
use serde::{Deserialize, Serialize};

use crate::asm_parser;

fn hexbytes(bytes: &[u8]) -> String {
    bytes
        .iter()
        .enumerate()
        .map(|(i, x)| format!("{:02x}", x) + if i + 1 == bytes.len() { "" } else { " " })
        .collect::<Vec<_>>()
        .concat()
}

// Represents a leaf in the x86 assembly trie; having a cached length (in
// bytes) and the assembly text associated with its byte sequence.
#[derive(Serialize, Deserialize, Clone)]
pub struct Terminal {
    pub length: u8,
    pub opcode: asm_parser::Opcode,
    pub asm: Option<String>,
}

// Member of the trie; for any given byte we can have a node that leads to more
// terminals, a terminal, or not have any entry yet.
#[derive(Serialize, Deserialize, Clone)]
pub enum TrieElem {
    Interior(Vec<TrieElem>),
    Terminal(Terminal),
    Nothing,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Trie {
    pub root: Vec<TrieElem>,
    pub total: u64,

    // Hashes of binaries already processed.
    pub binaries: HashSet<String>,

    // Whether to store provided assembly in the trie.
    pub keep_asm: bool,
}

impl Trie {
    pub fn new(keep_asm: bool) -> Trie {
        Trie {
            root: mk_empty_interior(),
            total: 0,
            binaries: HashSet::new(),
            keep_asm: keep_asm,
        }
    }
}

/// Inserts a terminal into the trie under "node".
///
/// Returns whether the terminal was newly added (false if it was already
/// present).
pub fn inserter(
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

fn mk_empty_interior() -> Vec<TrieElem> {
    let mut result = vec![];
    for _i in 0..256 {
        result.push(TrieElem::Nothing);
    }
    return result;
}

pub fn all_nothing(v: &Vec<TrieElem>) -> bool {
    v.iter().all(|t| {
        if let TrieElem::Nothing = t {
            true
        } else {
            false
        }
    })
}

pub fn traverse<F>(node: &Vec<TrieElem>, f: &mut F) -> ()
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

pub fn resolver<'a>(node: &'a Vec<TrieElem>, bytes: &[u8]) -> Option<&'a Terminal> {
    if bytes.len() == 0 {
        return None;
    }
    match &node[bytes[0] as usize] {
        TrieElem::Nothing => None,
        TrieElem::Terminal(t) => Some(t),
        TrieElem::Interior(n) => resolver(&n, &bytes[1..]),
    }
}

