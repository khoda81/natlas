# natlas

A web navigator for distributions over integer sequences, built as the visual
client for [natwalk](https://github.com/khoda81/natwalk). A tree's children
partition their parent's interval. Each child has a weight under each
*measure*, and what its branch means (a token, a note) comes separately.

![SmolLM2-360M continuing "The most interesting consequence of information theory is"](docs/text.png)

A language model's tree of texts. A box's height follows its probability and
its width follows its surprisal. The child boxes cover their parent's
probability mass times its conditional entropy when one measure sets both
dimensions.

## Quick start

Install [Bun](https://bun.sh), [Rust](https://www.rust-lang.org/tools/install),
[wasm-pack](https://wasm-bindgen.github.io/wasm-pack/installer/), and
[uv](https://docs.astral.sh/uv/).

```bash
bun install
bun run dev       # builds the Rust core and serves http://localhost:5173
```

Open <http://localhost:5173>. Drop a snapshot onto the page, or generate the
Zen of Python character n-gram dataset with `bun run data` from the repository
root. The generator needs a [natwalk checkout beside this one](export/README.md#generate-snapshots).
For live inference, start `bun run live` for language models or
`bun run scribe` for music transcription in a second terminal.

## Documentation

| Guide | Covers |
| --- | --- |
| [Browser](src/README.md) | Reading the picture, comparing measures, controls, and saved views |
| [Rust core](core/README.md) | Trees, exact positions, layout, picking, and demand sessions |
| [Protocol](proto/README.md) | WebSocket messages, search, and snapshot format |
| [Servers and exports](export/README.md) | Live language models, model scheduling, and snapshot generation and migration |
| [Music transcription](export/scribe.md) | MuScriptor, audio recordings, and the piano roll |

## Development

[Build scripts](package.json) are run from the repository root. The
[Rust core](core/Cargo.toml) builds to WebAssembly for the browser and to a
native Python module through PyO3 and maturin. The Python scripts declare
the local core dependency, so uv builds it automatically. Generated WebAssembly
files and Python wheels are build outputs.

```bash
bun run test       # TypeScript, native Rust, and Python tests
bun run typecheck
bun run build      # static site in dist/, including WebAssembly and datasets
bun run bench      # browser frame timings and canvas/inspector snapshots
bun run proto      # regenerate TypeScript and Python bindings; needs protoc
```

The PStore Criterion suite measures construction and batches of 256 queries
on the same 150,000-branch fixtures. For browser Wasm, put a
`wasm-bindgen-test-runner` matching `wasm-bindgen` in `core/Cargo.lock` and a
compatible WebDriver on your PATH. The Wasm suite runs in a dedicated
browser worker so the driver can read progress during sampling. The full
suite spends about five minutes warming up and sampling per target.

```bash
cargo bench --manifest-path core/Cargo.toml --no-default-features --bench pstore
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
WASM_BINDGEN_TEST_TIMEOUT=600 \
cargo bench --manifest-path core/Cargo.toml --no-default-features --bench pstore \
    --target wasm32-unknown-unknown
```

For targeted checks after building the core:

```bash
bun test
cargo test --manifest-path core/Cargo.toml
uv run tests/server_test.py
uv run tests/scribe_test.py
```

The [server tests](tests/server_test.py) use deterministic distributions.
To try the same fixture in the browser, run
`PORT=8427 uv run tests/server_test.py --serve` and open
`http://localhost:5173/?server=http://127.0.0.1:8427/`.
