# V5 experiments and reproduction

This runbook covers the roadmap's V5 GPU experiments. The task workloads run through AgentServe's normal application and step ledger. Each arm starts its own loopback vLLM and CPU resource processes and uses a new task database.

## Data and runtime

From the repository root, prepare the synthetic orders and the approved knowledge snapshots. Use Python 3.11. The experiment-only gold generation also imports NumPy.

```bash
python -m pip install -e '.[web,test]'
python -m pip install numpy==2.4.6
python scripts/setup/prepare_public_demo.py
python -m pytest -q
```

For the retail comparison, import the official [UCI Online Retail II](https://doi.org/10.24432/C5CG6D) archive using the repository importer. It expects 1,067,371 source lines and writes the schema and a data receipt beside SQLite.

```bash
python scripts/setup/prepare_online_retail_ii.py "$UCI_ARCHIVE" data/databases/online_retail_ii
```

GPU measurements use the Linux vLLM `0.32.0+cu129` environment, Qwen2.5-7B-Instruct, the model's tokenizer, Qwen3-Embedding-0.6B for CPU resources, and one RTX 4090 UUID. See the [serving setup](../serving/vllm/README.md) and the [Qwen file manifest](../configs/models/qwen2.5-7b-instruct.manifest.json). Set the variables below to the corresponding paths on your machine. Use a fresh `RUN` directory. The Python used to start vLLM can be in a separate environment.

To prepare the tested model versions, run these commands from the repository root. A prepopulated model directory is also accepted when its files match the pinned manifests.

```bash
python scripts/setup/download_model.py
"$VLLM_PYTHON" scripts/setup/prepare_memory_model.py --source modelscope --revision dd3e37eb506b3addbd930037796e25d349ae9010
```

```bash
export PYTHONPATH="$PWD/src"
mkdir -p "$RUN"
gpu=(--model-dir "$QWEN_DIR" --embedding-dir "$EMBEDDING_DIR"
     --vllm-python "$VLLM_PYTHON" --gpu-uuid "$GPU_UUID")
```

## P0 mixed business workload and recovery

```bash
python scripts/experiments/v6_phase_benchmark.py freeze "$RUN/parent.json"
python scripts/experiments/run_v5_phase.py pilot "$RUN/parent.json" "$RUN/pilot" "${gpu[@]}"
python scripts/experiments/run_v5_phase.py matrix "$RUN/parent.json" "$RUN/matrix" "${gpu[@]}"
python scripts/experiments/summarize_v5_phase.py "$RUN/matrix" "$RUN/pilot" "$RUN/parent.json" "$RUN/phase-receipt.json"
python scripts/experiments/run_v5_faults.py "$RUN/parent.json" "$RUN/faults" "${gpu[@]}"
```

The four ten-task blocks produced 120/120 correct outcomes and 228/228 linked model requests. P0-2 verified controlled response timeout, process exit between SQL execution and commit, duplicate submission, queued and running cancellation, and SQL/knowledge permissions. Recovery preserved the task ID and absolute deadline, reused two model outputs, and executed the targeted read-only query twice. See the [fault results and UCI correction](v5-reliability.md) for expected states, evidence boundaries and the regression replay.

## Expanded business workload

```bash
python scripts/experiments/prepare_v5_expansion.py "$RUN/parent.json" "$RUN/expansion.json"
EXPANSION_SHA=$(sha256sum "$RUN/expansion.json" | cut -d ' ' -f 1)
python scripts/experiments/run_v5_expansion.py smoke "$RUN/parent.json" "$RUN/expansion.json" "$RUN/expansion-smoke" --manifest-sha "$EXPANSION_SHA" "${gpu[@]}"
python scripts/experiments/run_v5_expansion.py matrix "$RUN/parent.json" "$RUN/expansion.json" "$RUN/expansion-matrix" --smoke "$RUN/expansion-smoke" --manifest-sha "$EXPANSION_SHA" "${gpu[@]}"
python scripts/experiments/v5_workbench_contract.py "$RUN/expansion.json" "$RUN/workbench-review.json"
python scripts/experiments/summarize_v5_expansion.py "$RUN/expansion-matrix" "$RUN/expansion.json" "$RUN/expansion-receipt.json" --workbench-contract "$RUN/workbench-review.json"
python scripts/experiments/run_v5_concurrency.py "$RUN/expansion-matrix" "$RUN/parent.json" "$RUN/expansion.json" "$RUN/concurrency8" "${gpu[@]}"
```

The 239 cases contain 144 SQL turns in 12 complete conversations, 47 knowledge questions, 24 metric queries, and 24 briefs. Four three-arm blocks total 2,868 submissions. Original per-arm scores were 230/239. A separately registered workbench review gives 238/239 by accounting for city aliases, unknown-city handling and the expected cross-database rejection. The remaining question consistently retrieved the correct document fifth under a four-chunk budget.

On this workload, full-stage SAM lowered within-block JCT P95 by 14.4%–20.6% versus AR. Submission throughput increased by 11.3%–12.1% at concurrency one and declined by 3.5%–4.0% at concurrency four. Each arm had 42 knowledge model requests and 196 accepted tasks that used no model. The C8 run keeps the vLLM maximum at four sequences and varies workbench concurrency.

The C8 pilot passed 76/76. The two formal AR and full-stage SAM blocks total 956 submissions and 168 linked model requests. AR throughput stayed near 15.2 submissions/s, while its P95 rose from 0.642–0.679 s at C4 to 0.873–0.884 s at C8. Full-stage SAM throughput overlapped the C4 range, and its P95 rose from 0.549–0.569 s to 0.757–0.764 s. Application queue and model-slot waiting increased. Mean engine queue time remained under 4 ms across these blocks.

## P1 native Suffix and Session-SAM

Install the native Suffix extension in an isolated directory built with GCC 11 and Python 3.11. Use that directory as `ARCTIC_SITE`. The [Suffix setup notes](vllm-suffix-baseline.md) describe the tested extension version and ABI. The runner pins the Qwen file hashes, launch settings and extension hash.

```bash
python scripts/experiments/run_v5_suffix.py both "$RUN/suffix" "${gpu[@]}" --arctic-site "$ARCTIC_SITE"
```

The pilot checks four protocol families and a separate retail development slice. Formal blocks run the full 48-turn retail suite, using one warmup request per arm. The experimental arms share a four-token draft cap. Native Suffix cache capacity is zero for request-local mode and 256 requests for cross-request mode. The Session-SAM arm uses the project's session history policy. Request IDs, SQL results, exact prompt/output token IDs and performance counters remain available in each attempt directory.

The protocol pilot passed 48/48 checks with exact output-token parity against AR. Both formal blocks gave every arm 47/48 correct retail turns and 50/50 complete request traces. The same German January customer-count question used the invoice column in all eight formal runs. Each paired comparison had the same SQL, rows and task status for all 48 turns. Draft counts below exclude the warmup and are identical across the two blocks.

| Arm | JCT P95, block 1 / 2 (s) | Accepted / proposed, per block |
| --- | --- | ---: |
| Pure AR | 3.510 / 3.495 | — |
| Request-local Suffix | 2.395 / 2.397 | 1,629 / 3,377 |
| Cross-request Suffix | 1.551 / 1.533 | 2,543 / 3,397 |
| Session-SAM | 2.620 / 2.634 | 1,470 / 3,113 |

Cross-request Suffix reduced within-block JCT P95 by about 56% on this fixed single-concurrency workload. The cache can reuse continuations from earlier requests in the same engine. This workload keeps the size, task order and reuse opportunity fixed.

The V8 release adds a separate [concurrency-four retail workload and feedback replay](v8-suffix-evaluation.md).

## Reading the results

Use the per-arm summaries and the paired SQL and model token evidence. Raw events, failed tasks and Prometheus snapshots are written under the run directory. JCT uses `ended_at - created_at`. Rejections contribute to submission counts, and JCT has a separate sample count. The Suffix experiment engine snapshots include the one retail warmup. The retail task and request summaries cover only formally measured tasks. Compare ITL event intervals with TPOT per-output-token timing using their separate denominators.
