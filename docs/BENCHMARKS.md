# Benchmarks

This page records TurboFieldfare measurements on an 8 GB M2 MacBook Air, a
24 GB M5 Pro, a 32 GB M5 MacBook Air for the 8-bit variant, and a 36 GB M4 Max
for Qwen3.6 beside Gemma 4. Each number belongs to the workload shown. Prompt length,
generated length, cache state, and hardware all change throughput, so ranges
across workloads are not run-to-run variation.

Each table states its workload and decoding settings. TurboFieldfare uses the
model installed by the [command-line instructions](../README.md#command-line-interface).
Decode rate excludes model installation, model loading, and prompt prefill.

## Results at a glance

| Host and runtime | Decode rate | Reported memory |
| --- | ---: | ---: |
| 8 GB M2, TurboFieldfare | 5.10-6.30 tok/s | ~1.9-2.1 GB footprint |
| 24 GB M5 Pro, TurboFieldfare | 31-35 tok/s | ~2.1 GB footprint |
| 24 GB M5 Pro, mlx-lm | 76.33-82.07 tok/s | 8.3-9.8 GB RSS; 14.7-15.3 GB GPU allocation |
| 32 GB M5, TurboFieldfare 8-bit | 7.65-9.69 tok/s | ~3.4-3.5 GiB footprint |
| 32 GB M5, TurboFieldfare 4-bit | 15.86-17.95 tok/s | ~2.1 GiB footprint |
| 36 GB M4 Max, Gemma 4 4-bit | 39.38-44.85 tok/s | ~2.2 GiB footprint |
| 36 GB M4 Max, Qwen3.6 4-bit | 14.15-27.73 tok/s | ~1.7 GiB footprint at 4K |

Rows for different hosts, runtimes, or model variants come from different
workloads and are not one series.

## M2 measured decode

These rows ran on a `Mac14,15` M2 MacBook Air with 8 GB of memory. No
experiment, profiler, or trace mode was active.

| Prompt / generated tokens | Prefill | TTFT | Decode | Peak RSS / footprint |
| --- | ---: | ---: | ---: | ---: |
| 6 / 32 | 7,025 ms | 7,979 ms | 6.30 tok/s | 1,304 / 1,791 MiB |
| 121 / 64 | 7,934 ms | 8,862 ms | 5.10 tok/s | 1,528 / 1,776 MiB |
| 527 / 64 | 21,736 ms | 22,649 ms | 5.90 tok/s | 1,535 / 1,886 MiB |
| 1,017 / 128 | 36,729 ms | 37,656 ms | 5.38 tok/s | 1,455 / 1,971 MiB |

Each workload ran once in a fresh process. The file cache was warm but
uncontrolled, and every row produced the same token IDs as its validation
control. These four points show the production path running under the 8 GB
rule; they do not form a confidence interval or describe sustained long
generation.

### Where the short M2 row spent its time

A separate diagnostic pass on the six-token prompt divided a 162.8 ms decode
step into four broad parts:

| Work | ms/token |
| --- | ---: |
| Expert reads | 83.1 |
| Waiting in the command-buffer pipeline | 55.6 |
| Tied output head | 14.2 |
| Other runtime work | 9.9 |

The diagnostic instrumentation disabled the normal command-buffer pipeline
and reduced throughput to 4.23 tok/s. The breakdown explains where that run
spent time; it does not describe independent speedups or a performance bound.

## M5 measured decode

These rows ran on 2026-07-20 on a 24 GB M5 Pro (`Mac17,8`) with macOS 26.5.1,
Xcode 26.6, and Swift 6.3.3. No profiler or trace mode was active.

The benchmark uses chat-framed prompts and fixed, non-repeating natural
continuations. This keeps the generated text and expert-routing workload stable
without rewarding a model repetition loop. The complete production sampling
and decode path still runs for every token.

One warmup preceded three fresh-process measurements per workload. The table
reports medians; the file cache was warm but uncontrolled. A separate
free-generation smoke reached the end of each model turn without a repetition
loop.

| Prompt / generated tokens | Prefill / TTFT | Decode | Peak RSS / footprint |
| --- | ---: | ---: | ---: |
| 61 / 256 | 5,096 / 5,668 ms | 35.17 tok/s | 1,834 / 2,126 MiB |
| 430 / 256 | 6,762 / 7,325 ms | 34.72 tok/s | 1,851 / 2,142 MiB |
| 3,015 / 256 | 23,038 / 23,610 ms | 31.01 tok/s | 1,835 / 2,126 MiB |

## M5 8-bit and 4-bit measured decode

These rows ran on 2026-11-08 on a 32 GB M5 MacBook Air (`Mac17,3`, 10 cores)
with macOS 27.2 (26B123) and Swift 6.4.2, connected to power with Low Power
Mode off, at commit `a250079`. No experimental control, profiler, or trace
mode was active.

Both variants ran the [community benchmark protocol](COMMUNITY_BENCHMARKS.md)
unchanged: one discarded warmup, then one measured fresh-process run per case,
with temperature `0.2`, Top-K `64`, Top-P `0.95`, the fixed seeds, a 4,096-token
context and the default 16 expert-cache slots. Every measured run ended with
`stop=endOfTurn`, and every answer was complete, without loops or repeated
blocks. The variants generate different text, so compare a row only with the
same variant's rows.

| Variant | Prompt / generated tokens | Decode |
| --- | ---: | ---: |
| 8-bit | 61 / 525 | 9.694 tok/s |
| 8-bit | 430 / 620 | 8.624 tok/s |
| 8-bit | 3,015 / 555 | 7.645 tok/s |
| 4-bit | 61 / 497 | 17.952 tok/s |
| 4-bit | 430 / 721 | 16.757 tok/s |
| 4-bit | 3,015 / 610 | 15.859 tok/s |

On this Mac the 8-bit model decodes at 0.48-0.54 times the 4-bit rate. That is
consistent with its byte counts: each routed-expert miss reads about 1.9 times
the bytes, and every resident projection, the output head included, reads
twice the weight bytes.

Peak physical footprint was measured separately with `/usr/bin/time -l`, using
the long-synthesis prompt, 256 generated tokens and the same sampling
settings:

| Variant | Context | Expert-cache slots | Peak footprint |
| --- | ---: | ---: | ---: |
| 8-bit | 4K | 16 | 3,485 MiB |
| 8-bit | 8K | 16 | 3,559 MiB |
| 8-bit | 4K | 32 | 6,385 MiB |
| 8-bit | 8K | 32 | 6,470 MiB |
| 4-bit | 4K | 16 | 2,111 MiB |
| 4-bit | 8K | 16 | 2,191 MiB |
| 4-bit | 4K | 32 | 3,662 MiB |
| 4-bit | 8K | 32 | 3,742 MiB |

Without the KV cache and the slots above 16, the 8-bit runtime measured
3.07-3.09 GiB and the 4-bit runtime 1.73-1.75 GiB. The 8-bit memory check uses
3.5 GiB for that part; see [Runtime controls](RUNTIME_CONTROLS.md#context-length-and-memory).
A 32 GB host never comes under memory pressure, so these rows do not show how
the 8-bit model behaves on a 16 GB Mac.

## M4 Max measured decode, Gemma 4 and Qwen3.6

Both families on one host, following the
[community protocol](COMMUNITY_BENCHMARKS.md) exactly: a discarded warmup and
one measured run per case, fresh processes, `--max-new 1024 --max-context 4096
--temperature 0.2 --top-k 64 --top-p 0.95` with the protocol's seeds, no
experimental controls. Host: Mac16,9, Apple M4 Max, 14 cores, 36 GB, AC power,
Low Power Mode off. Full records are in the
[S27 report](plans/qwen-migration/reports/S27.md).

Gemma 4 26B-A4B, 4-bit:

| Case | Prompt | Generated | Decode | Rate | Stop |
| --- | ---: | ---: | ---: | ---: | --- |
| short-explanation | 61 | 492 | 10.97 s | 44.845 tok/s | endOfTurn |
| medium-review | 430 | 731 | 17.45 s | 41.901 tok/s | endOfTurn |
| long-synthesis | 3,015 | 589 | 14.96 s | 39.378 tok/s | endOfTurn |

Qwen3.6 35B-A3B, 4-bit, with `--no-thinking`:

| Case | Prompt | Generated | Decode | Rate | Stop |
| --- | ---: | ---: | ---: | ---: | --- |
| short-explanation | 62 | 497 | 17.93 s | 27.725 tok/s | endOfTurn |
| medium-review | 426 | 710 | 28.31 s | 25.082 tok/s | endOfTurn |
| long-synthesis | 2,940 | 585 | 41.34 s | 14.151 tok/s | endOfTurn |

Qwen3.6 needs `--no-thinking` to produce a protocol-valid row. With reasoning
on, which is its default, all three cases spend the whole 1,024-token budget
inside a think block that never closes, so they end at `stop=maxTokens` with no
answer written at all. Those runs decode at 13.76-25.36 tok/s, but the protocol
requires `endOfTurn` on every measured footer, so they are not results.

Qwen3.6's rate falls further with prompt length than Gemma's does, from 27.7 to
14.2 tok/s between 62 and 2,940 prompt tokens against Gemma's 44.8 to 39.4.
That is consistent with the two geometries — ten full-attention layers over the
whole context against twenty-five ring-capped sliding-window layers — but one
measured run per case does not separate it from run-to-run variation.

Peak physical footprint on the same host, with a 3,015-token prompt and 256
generated tokens at 16 slots: Gemma 4-bit 2.210 GiB at a 4K context; Qwen3.6
1.696 GiB at 4K, 2.245 GiB at 32K and 4.111 GiB at 128K. Net of the computed
KV term, Qwen's non-KV residual is constant at 1.611-1.620 GiB, so the context
accounts for all of its growth.

## Same-host MLX comparison

The same M5 Pro ran MLX 0.32.0 and mlx-lm 0.31.3 against the same checkpoint,
prompt-token IDs, and generated-token counts. MLX measured 82.07, 80.25, and
76.33 tok/s for the 121-, 527-, and 1,017-token prompts.

Treat this as throughput context, not a complete engine comparison:

- The engines ran in separate blocks rather than a balanced, interleaved order.
- Their first-token clocks started at different points, so TTFT is not comparable.
- Generated IDs matched for the shortest prompt but diverged for the two longer prompts.
- TurboFieldfare recorded a 1.89-2.09 GiB physical footprint. MLX reported
  14.66-15.31 GB of peak GPU allocation and 8.27-9.79 GB of peak process RSS.
  Those counters measure different things and should not be compared as a
  direct memory ratio.

The MLX process required the larger host and is not an 8 GB TurboFieldfare
deployment path.

## Reproduce and contribute a result

The [community benchmark guide](COMMUNITY_BENCHMARKS.md) uses short, medium,
and long chat-framed prompts with fixed seeds. It requires coherent output and
a normal end of turn, so a repetition loop cannot become a published speed
result. The public CLI's timing footer reports decode-only throughput without a
separate research harness.

Community runs generate their own output, while the reference table uses fixed
non-repeating continuations for token-for-token stability. Compare community
submissions only when their prompt and generated token counts match.

A current checkout may not reproduce a historical number after the runtime,
compiler, or operating system changes. Report the commit and all three rows
rather than presenting one run as a general hardware result.

Read [System design](SYSTEM_DESIGN.md) for the runtime and resource split,
[Experiments](OPTIMIZATION_JOURNEY.md) for the main wins and failures, and the
[measurement lessons](experiments/summaries/09-validation-and-measurement-lessons.md)
for the rules used to evaluate performance changes.
