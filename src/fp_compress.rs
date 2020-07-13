// Attempt at an adaptation of "Fast lossless compression of scientific
// floating point data" by Ratanaworabhan, Ke, and Burtscher for fp32 data.

use bitvec::vec::BitVec;
use log::info;

#[derive(Clone, Copy, Debug)]
struct HashTableEntry {
    dcpred1: f32,
    dcpred2: f32,
}

struct Predictor {
    last: f32,
    deltas: [f32;3],
    entries: Vec<HashTableEntry>,
}

// Returns a 20 bit value.
fn hash_func(deltas: &[f32; 3]) -> u32 {
    (deltas[0].to_bits() ^ (deltas[1].to_bits() << 5) ^ (deltas[2].to_bits() << 10)) & 0x000f_ffff
}

// Everything up to two bits of the mantissa.
fn high_bits_same(x: f32, y: f32) -> bool {
    (x.to_bits() >> 21) == (y.to_bits() >> 21)
}

impl Predictor {
    fn predict(&mut self, new_delta: f32) -> f32 {
        info!(" new delta: {:?}", new_delta);
        let hash = hash_func(&self.deltas);
        info!(" hash: {:x}", hash);
        let entry = &mut self.entries[hash as usize];
        info!(" entry before: {:?}", entry);
        let pred_delta: f32 = if high_bits_same(entry.dcpred1, entry.dcpred2) {
            info!(" high bits same; using dcpred1 as delta: {:?}", entry.dcpred1);
            entry.dcpred1
        } else {
            let drift = entry.dcpred1-entry.dcpred2;
            entry.dcpred1 + drift
        };
        entry.dcpred2 = entry.dcpred1;
        entry.dcpred1 = new_delta;
        let pred: f32 = self.last + pred_delta;
        info!(" last: {:?}", self.last);
        info!(" pred_delta: {:?}", pred_delta);
        info!(" pred: {:?} == {:08x}", pred, pred.to_bits());
        info!(" entry after: {:?}", entry);
        pred
    }

    fn predict_and_feed(&mut self, x: f32) -> u32 {
        let new_delta = x - self.last;
        let pred = self.predict(new_delta);
        self.deltas[2] = self.deltas[1];
        self.deltas[1] = self.deltas[0];
        self.deltas[0] = new_delta;
        self.last = x;
        pred.to_bits()
    }

    fn new() -> Predictor {
        Predictor{last: 0.0, deltas: [0.0;3], entries: vec![HashTableEntry{dcpred1: 0.0, dcpred2: 0.0}; 1<<20]}
    }
}

fn push_bits(bits: &mut BitVec, value: u32, lsbs: u32) {
    let mut v = value;
    for _ in 0..lsbs {
        bits.push((v & 1) != 0);
        v >>= 1;
    }
}

pub fn compress(floats: &[f32]) -> BitVec {
    let mut bits = BitVec::new();
    let mut predictor = Predictor::new();
    for f in floats {
        info!("---");
        let u = f.to_bits();
        info!("value: {:08x}", u);
        let pred: u32 = predictor.predict_and_feed(*f);
        info!("pred: {:08x}", pred);
        let diff: u32 = u ^ pred;
        info!("xor-diff: {:08x}", diff);
        // Up to 32 leading zeros = 4 bytes = 8 nibbles = 16 crumbs.
        // If our prediction is perfect we need 4 bits to describe that all 8
        // nibbles are leading zeros.
        let lz = diff.leading_zeros();
        info!("lz: {}", lz);
        let lz_div_4 = lz / 4;
        let to_encode = 32 - lz_div_4 * 4;
        push_bits(&mut bits, lz_div_4, 4);
        push_bits(&mut bits, diff, to_encode);
        info!("enc-bits: {}", 4+to_encode);
    }
    bits
}

#[cfg(test)]
mod tests {
use env_logger;
use crate::fp_compress::compress;

fn init() {
    let _ = env_logger::builder().is_test(true).try_init();
}

#[test]
fn test_compress_zero() {
    assert_eq!(compress(&[0.0]).len(), 4);
}

#[test]
fn test_compress_2x_zero() {
    assert_eq!(compress(&[0.0, 0.0]).len(), 8);
}

#[test]
fn test_compress_one() {
    assert_eq!(compress(&[1.0]).len(), 36);
}

#[test]
fn test_compress_2x_one() {
    assert_eq!(compress(&[1.0, 1.0]).len(), 72);
}

#[test]
fn test_compress_3x_one() {
    init();
    assert_eq!(compress(&[1.0, 1.0, 1.0]).len(), 108);
}

#[test]
fn test_compress_4x_one() {
    init();
    assert_eq!(compress(&[1.0, 1.0, 1.0, 1.0]).len(), 112);
}

#[test]
fn test_compress_3x_seq() {
    assert_eq!(compress(&[1.0, 2.0, 3.0]).len(), 68);
}
}
