# SWE100 / H200 experiment protocol

Status: implementation and correctness validation, not completed reproduction.

## Frozen inputs and scope

- Source traces: SHA256 `420e77f513f6676cb1db8ae14c6773a3752ce73416a75e65886024547b369d19`, all 100 fixed SWE-bench test trajectories. No replacement, truncation, or success filtering.
- DeepSeek Flash generated the traces. Llama-3.1-8B-Instruct on one shared H200 executes the replay. This is an alternative-workload/hardware evaluation, not recovery of the paper's GPT-5/B200/A100 measurements.
- Primary replay includes visible responses, protocol-rejected responses, recovery responses, and measured tool execution gaps. Hidden DeepSeek reasoning is not a reusable prefix and is excluded from primary Llama replay. API/network delays are not Llama execution times. Failed HTTP attempts with no response cannot be assigned a known decode length; list them separately.
- Tokenize the actual messages using the archived Llama tokenizer. Force every visible response token, including its chat-template terminator, while executing the real model forward pass. Check exact generated tokens and actual cache-hit lengths. Do not freely generate different outputs then assume their KV matches the recorded next prompt.
- Tool gaps are original recorded durations. No future tool duration or future session length is supplied to policy decisions. Histories update only on tool return/session completion.
- Causal runtime-v3: terminal status is revealed only when that response completes, not while it is running; returned/queued jobs have known return probability1 rather than a censored tool survival estimate; the current admission's prior pin owner is excluded from Elastic/conditional-whole reclamation candidates and transferred after successful APC acquisition. Record both owner-reference revocation and net newly endangered unique blocks excluding current reserved hits/shared refs. Earlier v1/v2 runs are development evidence, not pooled into the v3 comparison.
- All main comparisons use identical Poisson arrival draws, trace order, model, block size 16, chunk budget 2048, and max sequences 64. Use seeds 42/43/44 for repeated runs where feasible; distinguish single-run evidence from replicated confidence intervals.
- Primary no-offload study: 24 GiB KV budget. This is an experimental capacity, not a disk/budget stop threshold. Additional capacities/rates must be reported even when they give no improvement.

First real-replay matrix, fixed before full-run results: JPS 0.12 / 24 GiB, primary `vllm-fcfs`, `continuum-paper`, `elastic` with seeds 42, 43, 44; auxiliary `continuum-public`, `static-25`, `static-50`, `static-75`, `conditional-whole` with seed 42. Primary rows have three arrival-seed trials; auxiliary rows are single-trial descriptive controls, not replicated superiority claims. This matrix tests one load/capacity point and is NOT a completed paper load/capacity sweep. `conditional-whole` uses the same conditional return probability at reclamation, but is not the fully periodic Dynamic-Continuum proposed in the idea.

## Systems and provenance

The public `Hanchenli/vllm-continuum` checkout at `316a58794a6ff86b216e579b74fd56ed0c5a911f` explicitly says **without the estimation in the paper**. Its tool estimator uses a fixed two-second threshold and its public tree does not contain the paper's PLAS/InferCept implementations or a complete trace replay harness.

Name rows accurately:

1. `vllm-fcfs`: FCFS path of the public vLLM 0.10.2 Continuum fork, with APC, not an independently installed pristine wheel.
2. `continuum-public`: published fixed-threshold estimator and scheduler, shared instrumentation/correctness fixes documented separately.
3. `continuum-paper`: reconstructed paper CDF TTL objective with online histories and H200 prefill profile. Not the unavailable author estimator.
4. `static-25/50/75`: same reconstructed TTL/predictor and scheduler; fixed fraction protected.
5. `elastic`: same TTL/predictor/scheduler, admission-targeted prefix-boundary reclamation using conditional return probability times measured marginal prefill cost.
6. Additional PLAS, InferCept, CacheWise rows require verified implementations. Never label a home-made heuristic as the official system.

Shared runtime correctness changes: iterate over a snapshot when expiring/removing pins; transfer ownership on successful return admission after APC references are acquired (Algorithm 1 lines 32–34); robust victim fallback when no non-final running request exists; disable cascade attention, whose refcount-based common-prefix inference is incompatible with extra pin-owner refs. The public row is therefore a fixed-threshold **public-policy baseline in this shared safety-patched harness**, not an untouched checkout timing claim. All systems use the same instrumentation. Allocation recovery retries immediately after releasing protection instead of wasting an empty scheduler iteration.

## Elastic operationalization

Use the optimized idea's prefix-closed protection and the motivation document's admission-targeted release. Unpin means decrement this request's reference, NOT delete KV/hash. Opportunistic blocks survive until normal vLLM allocation overwrites them. Reclaim only enough **physically reclaimable** blocks for an actual failed allocation; shared refs do not count as released capacity. Protecting a returned prefix and unpinning its prior owner must be reference-safe.

At admission, boundary utility is `q(elapsed,H) * [T(k-1,N-k+1)-T(k,N-k)]`; `H=1 s` is fixed before outcomes. The memory price is common to equal-sized boundary blocks and cancels from the constrained minimum-utility ranking, so the admission deficit supplies the constraint; do not claim an independently tuned lambda mechanism. Main version uses GPU recompute cost only; queue-cost and predictor ablations are separate, not silently attributed to granularity. No CPU tiering in this version.

## Measurements and interpretation

Record per-job JCT, P50/P95, makespan throughput, request queue delay, actual prefix hits, scheduled/recomputed tokens, preemptions, unpin vs physical eviction, waiting-protected KV memory-time, admission deficits and RAF, and scheduler overhead. Store configuration, code hash, hardware snapshot, token checks and errors with every run. Report negative results and incomplete runs.

Offline H200 prefix-fraction profiling is characterization, not JCT reproduction. Any discrete-event model must be explicitly marked simulation and validated against measured replay before using it to extrapolate. Missing BFCL/70B/multi-GPU/distributed/offload experiments remain missing, not inferred from SWE8B results.

Do not modify the original traces, official checkout, user manuscripts, other users' files/processes, or existing environment. New experiment files live in an isolated study directory; model weights are reused in place. No API calls are needed for replay.
