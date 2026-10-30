# Categorical partition playground

An isolated experiment over one categorical distribution. It does not change
the hierarchical renderer or the v1/v2 services.

`proto/natlas/partition/v1/partition.proto` defines the experimental RPC. A
mask/value predicate identifies token IDs: `(id & mask) == value`. Adding up
to four previously unfixed bits splits the group into at most sixteen children.
Each response carries group mass bounds under the original normalized measure;
Q additionally returns extrema of individual log probabilities. A group is
never assigned uniform token weights or a guessed rank.

The TypeScript mock endpoints use the generated protobuf codecs. Budgets count
the exact uncompressed request **plus** response message lengths, separately
for P and Q. The simulation admits complete exchanges atomically. A prospective
reply that does not fit is prepared inside the mock endpoint but never decoded
by the client, and no partial exchange is counted. HTTP, TLS, compression,
fixture download, and the optional reference overlay are outside this simulated
budget; this is not a measurement of the browser's actual network traffic.

P requests prioritize observed mass times log token count. Q then follows the
measured P partition, prioritizing P mass times clipped width spread. Both use
the chosen fixed bit order. Q stops when its extrema already describe every P
leaf, since finer Q alone cannot allocate unknown P mass inside that leaf.
Changing a slider replays a fresh client; server computation caches can be reused.

For each P group of mass p, Q supplies a valid interval [a,b] for every token's
surprisal, using the nearest known ancestor if Q has less evidence. The CDF of
surprisal under P lies between sum(p for b <= x) and sum(p for a <= x).
Probability intervals and normalization tighten these bounds. Their inverses
bound the sorted silhouette. Individual token ranks remain unresolved.
Cross-entropy is bounded by sum(p*a) and sum(p*b). The common P/Q partition also
gives the chain-rule lower term sum(-P(group)*ln Q(group)).

The mock measure is the exact mathematical normalization of positive binary64
weights obtained from the captured logs. Positive-sum forward-error bounds
cover aggregation/normalization; log extrema include a floating-point margin.
The fixtures are safely inside binary64's exponent range. Other inputs that
would underflow positive weights are rejected explicitly. These are numerical
bounds, not confidence intervals or bounds on model calibration.

## Real Qwen captures

The checked-in `public/playground/qwen3-4b/` contains two complete, gzip-compressed
float64 log-probability vectors and shared token labels. The prompt, token IDs,
source revision, model/data SHA-256 values, and capture provenance are recorded
in `manifest.json`. Q4_K_M was **re-quantized from the cached Q6_K weights**,
not independently quantized from FP16. Both were evaluated on CPU, with the same
prompt and temperature 1, without token generation. No model download or GPU
inference is needed to replay the demo.

To reproduce with the Python environment used by `export/serve.py`:

```sh
python scripts/capture-categorical.py --model /path/to/Qwen3-4B-Q6_K.gguf \
  --requantized /path/to/Qwen3-4B-Q4_K_M-from-Q6_K.gguf
```

The capture script performs one model re-quantization if needed and two prompt
evaluations. It uses four CPU threads by default. It does not run a server.

Validate the codec, budget gate, and bounds with `bun test tests/partition.test.ts`.
Run the bounded real-data sweep with `bun scripts/partition-benchmark.ts`.
