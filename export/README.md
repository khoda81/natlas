# Live servers and snapshot exports

[Project setup](../README.md) · [Wire protocol](../proto/README.md) · [Music transcription](scribe.md)

## Run a live language model

From the repository root, run `bun run live`. From this directory, the
equivalent is `uv run serve.py`. The server listens at http://localhost:8421
and reads [serve.toml](serve.toml). To choose a different configuration,
run `uv run serve.py --config path/to/config.toml` from this directory.

A local browser page automatically looks for a server on port 8421 and lists
it first. Use `?server=URL` to choose another server. Edit text in the
[path inspector](../src/README.md#canvas-navigation) and select a result to
fly there.

[serve.py](serve.py) offers the sources in its configuration. They all share
a tokenizer, which the server reads before loading any model. Models load
when the page asks. The root is the empty text, with any start-of-text token
handled internally by the model.

A model is a source on a device at a precision, such as
`SmolLM2-360M/cuda/float32`. A measure spec adds optional settings to that
name. For example, `SmolLM2-360M/cuda/float32/context=4` reads only the last
four tokens. Other settings include `temperature=T` and, for GGUF models,
`control=NAME`. Precision `auto` uses the checkpoint's saved precision.
See [lm.py](lm.py) for the model and measure implementations.

### Qwen chat markers

The live server tokenizes the text you enter in the path inspector directly.
In Qwen3 chat, `<|im_start|>user` and `<|im_start|>assistant` open turns,
and `<|im_end|>` closes a turn. Paste a prefix like this to explore the
assistant's next token.

```text
<|im_start|>user
What is 2 + 2?<|im_end|>
<|im_start|>assistant
```

Keep a newline after `assistant`. Use Shift+Enter to insert a newline in
the inspector and Enter to jump to the path.

`<think>` opens the assistant's reasoning block and `</think>` closes it
before the answer. To start inside the block, append `<think>` followed by
a newline. The Qwen3 non-thinking template appends an empty block,
`<think>\n\n</think>\n\n`, with real newlines.
[Qwen3-4B chat template](https://huggingface.co/Qwen/Qwen3-4B/blob/main/tokenizer_config.json)

## Live endpoints

The [shared service](service.py) implements the browser's WebSocket session
and the following HTTP endpoints.

| Request | Result |
| --- | --- |
| `GET /manifest.json` | Datasets, model presets, and accepted spec parameters |
| `GET /labels.json` | Branch labels, as token bytes for language models or event descriptions for music |
| `GET /held` | Models loading, loaded, or failed, and measures that are open |
| `POST /models/SPEC` | Load a model |
| `DELETE /models/SPEC` | Unload a model and release its memory |
| `POST /measures/SPEC` | Open a measure of a loaded model |
| `POST /natlas.Natlas/Search` | Look up text on a server that provides text search |

## Scheduling

In [service.py](service.py), one scheduler thread owns the shared Rust tree
and the clients' sessions. WebSocket handlers post demands without waiting
for inference. The scheduler handles a burst together, keeping each client's
latest view, then sends any newly available data and chooses the next pass.
It fills a pass with as many distributions as the measure can compute at once,
taking a walk from each client in turn. Within each client, the session's
[view priorities](../core/README.md#tree-and-demand) determine the order.

A model thread owns inference and its KV cache. It can have one pass running
and one queued, so it can start the next while the scheduler packs the last.
The next pass is chosen before the running pass's result is available. A new
demand or a disconnect withdraws obsolete queued work. Running work can
finish into the shared cache.

Models load on separate threads. Measures open on the model thread between
passes. A demand for a measure whose model is loading waits. If the measure
cannot be opened, the client receives the reason.

The scheduler keeps packed distributions within the `--cache` budget,
1 GiB by default, at roughly four bytes per vocabulary entry. Transformers
models reuse the KV cache of the last path and read ahead in chunks. Their
float32 results can differ slightly from computing a position by itself.

## GGUF models

llama.cpp uses the precision stored in the GGUF file. The config's
`gpu_layers` sets how many layers go on the GPU, with all offloaded by
default. Use a CPU model spec to run on the CPU. The `n_ctx` setting limits
the KV cache cells available for a deep path and the paths kept beside it.
[serve.py](serve.py) configures uv to build llama-cpp-python with CUDA on
first install.

The KV cache holds a sequence per path, sharing cells for common prefixes.
The model reads on from the sequence with the longest shared prefix, or
branches off into a sequence of its own. Descending one cached path costs one
new token per step, and other paths can take turns without reading the same
prefix again. The least recently read sequences give way when cells run out.

At startup, a llama.cpp model times batches of 1, 2, 4, and so on, with paths
branching off one token. It chooses the batch width that maximizes
$k / t(k)^2$, where $t(k)$ is the time to read $k$ paths. On a GTX 1660 Ti,
Qwen3-4B Q6_K chose four paths. Four took 29 ms and one took 19 ms with an empty
KV cache. A full 4096-cell cache added 12–17 ms.

Batched logits round differently from a token read alone. In the same tests,
that made a difference of about $10^{-3}$ nats of KL divergence, and the CPU
and GPU backends differed by about $4 \times 10^{-2}$ nats. One token's logits
also shifted by up to 0.3 when 1200 KV cells held other paths. On the tested
GTX 1660 Ti, float32 Transformers inference was about five times as fast as
float16. Float16 uses less memory for larger models.

```toml
presets = ["Qwen3-4B-Q6_K/cuda/auto"]
n_ctx = 4096

[sources]
Qwen3-4B-Q6_K = "~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-GGUF/snapshots/<hash>/Qwen3-4B-Q6_K.gguf"

[controls]
evil = "evil.gguf"
```

Each entry in `controls` makes a llama.cpp control vector available through
`control=NAME`, read in a context of its own. [steer.py](steer.py) builds
a vector from MLP neurons. Each neuron's dose is multiplied by its layer's
median residual norm along the neuron's `W_down` column.

```bash
uv run steer.py Qwen/Qwen3.5-4B L14N8949+1 L13N7956-1 --out evil.gguf
```

## Generate snapshots

The generators use a [natwalk](https://github.com/khoda81/natwalk) checkout
beside this repository. Live servers use the dependencies declared in
their script headers, which uv installs.

From the repository root, generate the Zen of Python character n-gram dataset
with `bun run data`. It uses [synthetic.py](synthetic.py) and needs no model
inference. For a language model, run [hf.py](hf.py) from this directory.

```bash
uv run hf.py --config serve.toml --p SmolLM2-360M/cuda/float32 \
  --q SmolLM2-360M/cuda/float32/context=4 --q SmolLM2-360M/cuda/float32/temperature=1.6
```

Both generators use natwalk's search to discover the tree, then evaluate
the other measures on the same paths. The primary measure lists enough
branches to include the discovered children and meet the coverage target
within the listing limit. The other measures rank their own branches and
round their surprisal cutoff up to the next bit. The language-model exporter
roots the tree at the empty text and records the path to the prompt where
discovery begins. The recorded measures use the same specs as the live server.

[natlas_export.py](natlas_export.py) writes datasets under the git-ignored
`public/data/`. Each dataset has an entry in `manifest.json`, one `.pb`
snapshot per measure, and a `.labels.json` file containing the branches'
bytes as a base64 JSON array.

### Migrate older snapshots

From the repository root, use [migrate-snapshots.py](../scripts/migrate-snapshots.py)
to convert snapshots with log-mass-below edges into runs with exact tops and
weights. The converter needs no model inference.

```bash
uv run scripts/migrate-snapshots.py           # all public/data/*.pb
uv run scripts/migrate-snapshots.py tree.pb   # selected files
```

The original goes into `.snapshot-backups/edges/` beside each converted
file. You can also regenerate the snapshots.
