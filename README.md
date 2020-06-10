## Installation

Get rustup for the nightly Rust toolchain (required by PyO3):
https://rustup.rs/

```
pip install setuptools_rust
pip install --editable ./
```

Then you should be able to ingest x86 binaries into the x86 trie:

```
python l2d2/ingest_main.py --ingest-limit 2
```

And then train on that ingested data:

```
python l2d2/attempt.py
```

## Rust Benchmarks

```
$ cargo bench --no-default-features
```

## Tips

Ensure that any development on the Rust extension that requires performance
measurements uses:

```
python setup.py install
```

Using `develop` instead will not use "release" mode.
