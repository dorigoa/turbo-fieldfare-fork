# Local OpenAI-compatible server

`TurboFieldfareServer` exposes a local Chat Completions API for one model,
Gemma 4 or [Qwen3.6](#qwen36). It binds to `127.0.0.1` by default and has no
authentication or TLS. Do not expose it through a proxy or tunnel; to reach it
from another machine on a network you trust, see
[Listen on another address](#listen-on-another-address).

## Start the server

First, install the model with the Mac app or `TurboFieldfareRepack`. Then check
that no other TurboFieldfare model process is running:

```bash
pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

If the command prints a match, do not start the server.

```bash
swift build -c release --product TurboFieldfareServer
.build/release/TurboFieldfareServer \
  --model scratch/gemma4.gturbo \
  --port 8080 \
  --max-context 16384
```

The server loads the model before opening the port. Wait for
`TurboFieldfareServer ready`, then keep the process running while clients use
it. The ready line names the API model ID (`model=`), and at its end the
model's variant (`variant=4-bit` or `variant=8-bit`) and the model ID its
manifest records (`manifest_model_id=`). Before that, the
[log](#logs) reports `model opened` with the same two fields once the weights
are open.

To serve the experimental 8-bit variant, pass its directory. The server reads
the variant from the model's manifest; there is no variant flag:

```bash
.build/release/TurboFieldfareServer \
  --model scratch/gemma4-8bit.gturbo \
  --port 8080 \
  --max-context 16384
```

Check the server from another terminal:

```bash
curl --silent --show-error http://127.0.0.1:8080/health
curl --silent --show-error http://127.0.0.1:8080/v1/models
curl --silent --show-error http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-4-26b-a4b-it",
    "messages": [{"role": "user", "content": "Reply with exactly READY."}],
    "temperature": 0,
    "max_completion_tokens": 16
  }'
```

Requests must name the server's API model ID, which `GET /v1/models` lists.
See [Model ID](#model-id) for the 8-bit value. `model` is required: a request
without it gets HTTP 400 with `missing required field "model"`. A body that
does not decode is answered with code `invalid_json`, a message naming the
field and what it must be, and the field's path in `param`, for example
`"seed" must be a non-negative integer` or `request body is not valid JSON
(line 1, column 3)`.

By default, the server runs one generation and admits up to four additional
requests for preparation or queueing. The limit is enforced before prompt
rendering and tokenization. Use `--queue-limit` to change it. Press Control-C
to stop the server.

## Listen on another address

`--host` sets the address the server listens on, so other machines can reach
it without a port forwarder such as `socat`. It takes an IPv4 or IPv6 address
or `localhost`; `0.0.0.0` listens on every IPv4 interface and `::` on every
IPv6 one. A host name is refused, because it would be resolved only after the
model has loaded.

```bash
.build/release/TurboFieldfareServer \
  --model scratch/gemma4.gturbo \
  --host 0.0.0.0 \
  --port 8080
```

The server still has no authentication or TLS. Anyone who can reach the
address can run the model, read its replies, and keep the queue full, and
requests and replies cross the network in clear text. Use it only on a network
you trust, and allow the port in the macOS firewall if it asks. For any
address other than loopback the server prints a warning to stderr before it
opens the port, and the ready line names the address it bound, for example
`ready at http://0.0.0.0:8080`. Clients on other machines use this Mac's
address in the base URL, such as `http://192.168.1.20:8080/v1`.

## Metrics

`GET /metrics` reports the server's state in the Prometheus text format, so
Prometheus, Grafana Agent, or any compatible scraper can read it, and a quick
look needs only `curl`:

```bash
curl --silent --show-error http://127.0.0.1:8080/metrics
```

Every figure comes from what the server already measures for each request: the
token counts and the prefill and generation times are the ones the `completed`
log line prints. Counters start at zero when the server opens its port and
cover chat completion requests only.

| Metric | Type | Meaning |
| --- | --- | --- |
| `turbofieldfare_model_info` | gauge | Always 1. Labels name the API model ID, variant, family, manifest model ID, image support, prompt-cache mode and, for Gemma, the expert-cache slots and early expert read. |
| `turbofieldfare_context_max_tokens` | gauge | The `--max-context` the server started with. |
| `turbofieldfare_last_context_tokens` | gauge | Context the last completed request filled: prompt plus generated tokens. |
| `turbofieldfare_last_prefill_tokens_per_second` | gauge | Prefill speed of the last request that prefilled any token; cached tokens are not counted as prefilled. |
| `turbofieldfare_last_decode_tokens_per_second` | gauge | Generation speed of the last request that generated any token. |
| `turbofieldfare_requests_running`, `turbofieldfare_requests_waiting` | gauge | The request generating now (0 or 1) and the prepared requests queued behind it. |
| `turbofieldfare_queue_limit` | gauge | The `--queue-limit`. |
| `turbofieldfare_requests_total{result}` | counter | Requests that `completed`, were `cancelled` by the client, or `failed`. |
| `turbofieldfare_request_failures_total{status}` | counter | Failed requests by HTTP status, for example `429` when the queue is full. |
| `turbofieldfare_completions_total{finish_reason}` | counter | Completed requests by `stop`, `length` or `tool_calls`. |
| `turbofieldfare_prompt_tokens_total` | counter | Prompt tokens, prompt-cache hits included. |
| `turbofieldfare_prompt_cached_tokens_total` | counter | Prompt tokens reused from the prompt cache. |
| `turbofieldfare_prefill_tokens_total` | counter | Prompt tokens actually prefilled: prompt minus cached. |
| `turbofieldfare_generation_tokens_total` | counter | Generated tokens. |
| `turbofieldfare_prefill_seconds_total`, `turbofieldfare_decode_seconds_total` | counter | Time spent prefilling and generating. |
| `turbofieldfare_time_to_first_output_seconds` | histogram | From accepting a request to its first content or tool call, queueing and prefill included. |
| `turbofieldfare_request_duration_seconds` | histogram | From accepting a request to its completion. |
| `turbofieldfare_memory_footprint_bytes` | gauge | Memory charged to the process: runtime buffers and KV cache, without the mapped weights. |
| `turbofieldfare_memory_mapped_resident_bytes` | gauge | Mapped weights currently resident in RAM, which the kernel can drop under pressure. |
| `process_resident_memory_bytes` | gauge | Resident memory as the kernel reports it. |
| `turbofieldfare_start_time_seconds` | gauge | Unix time the server opened its port. |

The `last_*` gauges are absent until a request has done that work, rather than
reporting a speed nobody measured. `turbofieldfare_memory_footprint_bytes` is
the same measure as the memory figure in the Mac app's HUD; it leaves out the
mapped weights, which `turbofieldfare_memory_mapped_resident_bytes` counts, and
the two together are what the process occupies in RAM. Speeds and context fill
over a window come from the counters, for example in PromQL:

```text
rate(turbofieldfare_generation_tokens_total[5m]) / rate(turbofieldfare_decode_seconds_total[5m])
rate(turbofieldfare_prefill_tokens_total[5m]) / rate(turbofieldfare_prefill_seconds_total[5m])
turbofieldfare_last_context_tokens / turbofieldfare_context_max_tokens
```

A scrape reads counters the server already keeps and two kernel calls, and it
never waits for a generation to finish. Like every other endpoint it has no
authentication: with [`--host`](#listen-on-another-address) it is visible to
whoever can reach the server.

## Runtime settings

The server accepts the same runtime flags as the CLI, with the same values and
defaults. See [Runtime controls](RUNTIME_CONTROLS.md) for what each one does.

```bash
.build/release/TurboFieldfareServer \
  --model scratch/gemma4.gturbo \
  --expert-cache-slots 32 \
  --expert-cache-policy lru \
  --prefill on \
  --prefill-chunk-tokens 64 \
  --rdadvise bounded
```

Without these flags the server runs the production defaults: 16 expert-cache
slots, LFU eviction, chunked prefill on with 128-token chunks, and prefetch and
read advice off. `--early-expert-read router` opts in at server startup; it is
not a Chat Completions request field.

`--prefill-chunk-tokens auto` runs at the cap,
256, on the server: a per-request size is the smallest allowed size covering the
span being prefilled, so the cap prefills every prompt in exactly those spans,
and it costs about 32.5 MB of prefill scratch against about 16.4 MB at the 128
default. Values are validated before the model loads, so an unsupported one
exits with the usage text rather than failing partway through startup. Chunked
prefill needs at least 16 expert-cache slots, so `--expert-cache-slots 8`
requires `--prefill off`.

The settings are fixed for the life of the process. Restart the server to
change them.

## Model ID

Every request's `model` field must match the server's API model ID; any other
value returns HTTP 404 with `model_not_found`. Without `--model-id`, the ID
follows the model's variant:

| Variant | API model ID |
| --- | --- |
| 4-bit | `gemma-4-26b-a4b-it` |
| 8-bit | `mlx-community/gemma-4-26b-a4b-it-8bit` |

The 4-bit model keeps the ID it always had, so existing client settings keep
working. The 8-bit model answers to the model ID in its manifest, its source
repository, so a client still set up for the 4-bit name is refused instead of
being served a different variant without notice. `--model-id <id>` replaces
the ID for either variant. `GET /v1/models` lists the ID in use, and responses
echo it in their `model` field.

## Qwen3.6

The server also serves Qwen3.6-35B-A3B (4-bit). Pass its model directory; the
server reads the family from the manifest, and the `model opened` and ready
lines name it with `family=qwen3_5_moe`:

```bash
.build/release/TurboFieldfareServer \
  --model scratch/qwen36.gturbo \
  --port 8080 \
  --max-context 16384
```

The API model ID is the one its manifest records,
`mlx-community/Qwen3.6-35B-A3B-4bit`; `--model-id` replaces it as for Gemma.
The differences from Gemma:

- **Sampling defaults.** A request that omits `temperature`, `top_k` or
  `top_p` gets the checkpoint's reference values: 1.0, 20 and 0.95. Values the
  request sets are used as given.
- **Reasoning.** Thinking is on by default. The reasoning text never reaches
  `content`. Send `"chat_template_kwargs": {"enable_thinking": false}` to answer
  without it; that object accepts only the boolean `enable_thinking`. Under
  Gemma the field is still an unrecognized request field.
- **Reasoning depth.** `reasoning_effort` takes `low`, `medium` or `high` and
  caps the tokens the model may spend reasoning at 256, 1,024 or 4,096. When
  the budget runs out the runtime closes the thinking block and the model
  answers; omitting the field leaves the reasoning unbounded, as it was before
  the field was accepted. The chat template has no such parameter, so this is
  enforced during decode, not in the prompt — the cached prefix is the same
  either way. Any other value returns `unsupported_value`, combining it with
  `enable_thinking: false` returns `invalid_value`, and under Gemma, which has
  no reasoning phase, the field still returns `unsupported_value`.
- **Tool calls.** Tools are rendered by the model's own chat template, and the
  model's XML tool calls come back in the OpenAI `tool_calls` field with
  `finish_reason` `tool_calls`. A call to a tool the request did not declare
  fails the request instead of reaching the client.
- **Prompt reuse.** A request is reused only as a continuation at the exact
  end of the cached prompt; any other request runs a full prefill. The
  model's state cannot be rewound to an earlier point.
- **Images.** Not available. A request with an image gets HTTP 400 with
  `image_not_supported` and the message "image input is not available for
  this model"; `--vision-pack` with a Qwen model is refused at startup.

## Connect a client

The base URL is `http://127.0.0.1:8080/v1`. Some client libraries require an
API key, but the server ignores it. The examples below use the 4-bit model ID;
with the 8-bit model, use `mlx-community/gemma-4-26b-a4b-it-8bit` wherever
they say `gemma-4-26b-a4b-it`.

Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
response = client.chat.completions.create(
    model="gemma-4-26b-a4b-it",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(response.choices[0].message.content)
```

OpenCode:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "turbofieldfare": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "TurboFieldfare",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "local"
      },
      "models": {
        "gemma-4-26b-a4b-it": {
          "name": "Gemma 4 26B-A4B IT",
          "attachment": true,
          "modalities": {
            "input": ["text", "image"],
            "output": ["text"]
          },
          "limit": {
            "context": 16384,
            "output": 4096
          }
        }
      }
    }
  }
}
```

Select `turbofieldfare/gemma-4-26b-a4b-it` in OpenCode.

`attachment` and `modalities` are what make images work. OpenCode decides whether
a model accepts images from this configuration, not from the `capabilities` field
`/v1/models` returns, so without them it refuses an image with "this model does
not support image input" before sending anything. Images then reach the server
through OpenCode's `read` tool, which returns image files as attachments; ask it
to read the file explicitly.

Pi uses its `openai-completions` adapter:

```json
{
  "providers": {
    "turbofieldfare": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "compat": {
        "supportsReasoningEffort": false,
        "supportsStrictMode": false,
        "supportsUsageInStreaming": true
      },
      "models": [{
        "id": "gemma-4-26b-a4b-it",
        "name": "Gemma 4 26B-A4B IT",
        "reasoning": false,
        "input": ["text", "image"],
        "contextWindow": 16384,
        "maxTokens": 4096
      }]
    }
  }
}
```

Keep the client context setting at or below the server's `--max-context`.

Pi's `input` field declares image support the same way OpenCode's `modalities`
does. With it, `pi -p @photo.png "What is in this image?"` sends the image as a
base64 data URL in an `image_url` part, which is the shape this server accepts;
its RPC interface takes an explicit `images` array and produces the same
request.

## Prompt reuse

Single-prefix KV reuse is on by default. Send the complete message history with
every request. When a request continues the retained conversation exactly, the
server reuses the verified KV prefix and reports the number of reused tokens in:

```text
usage.prompt_tokens_details.cached_tokens
```

The server retains one prefix. A different or incompatible history replaces
it. Use `--prompt-cache-mode off` to disable reuse.

Cache misses and rejected prefix publication include a reason code in the
server log, such as `history-diverged` or `stop-string-filtered`. These cache
diagnostics contain no prompt, response, tool payload or image content. Cache
diagnostics are silent when prompt reuse is disabled.

## Tool calls

The server can return OpenAI-style function calls, but it cannot authorize or
execute them. The client runs the tool loop:

1. Send function schemas in `tools`.
2. When `finish_reason` is `"tool_calls"`, inspect each function name and JSON
   argument object. Apply the client's normal permission checks before running
   the function.
3. Append the assistant message, including its unchanged `tool_calls`.
4. Append each result as a `role: "tool"` message. Its `tool_call_id` must
   match the call it resolves.
5. Send the complete history and tool schemas again.

The server accepts only function tools. Omit `tool_choice` or set it to `auto`
to allow calls. Set it to `none` to disable them. The server does not support
`required`, named tool selection, or `parallel_tool_calls: false`.

Tool schemas need a non-null object at the top level and explicit JSON Schema
types. Nested properties and items may use nullable forms with one concrete
type plus `null`, including equivalent two-branch `anyOf` and disjoint `oneOf`
forms. Unions of string constants are also supported. Overlapping `oneOf`,
mixed-type unions such as `string | object`, and `allOf` return HTTP 400 with
`invalid_tool_schema`; the server does not guess which branch the model should
use.

## Supported API

Endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Chat Completions supports JSON and Server-Sent Events responses. Set
`"stream": true` for streaming. Set
`"stream_options": {"include_usage": true}` to receive a final usage chunk.

Requests may contain system, developer, user, assistant, and tool messages.
Multiple leading system messages retain their order. System and developer
guidance must precede the conversation; guidance inserted after a user,
assistant or tool message is rejected.
Supported options include `temperature`, `top_p`, `top_k`,
`repetition_penalty`, `seed`, `stop`, `max_tokens`,
`max_completion_tokens`, and function-tool fields.

Unknown top-level request fields return HTTP 400 with `code`
`unknown_parameter` and the field name in `param`, so a misspelled option is
refused rather than silently ignored. `response_format` is accepted only as
`{"type": "text"}`; `json_object` and `json_schema` return
`unsupported_value`, as do `logit_bias`, `top_logprobs`,
`verbosity`, `modalities`, `audio`, `prediction`, `web_search_options`, and
the legacy `functions` and `function_call`. `reasoning_effort` is read only
under Qwen3.6 (see [Qwen3.6](#qwen36)) and returns `unsupported_value` for
every other family. `response_format` must be an object; any
other JSON value returns `invalid_value`. `user`, `store`, `metadata`,
`service_tier`, `prompt_cache_key`, and `safety_identifier` are accepted and
ignored. A top-level field set to `null` is treated as absent. Fields inside
`messages`, `tools`, and `stream_options` are not checked for extras. Inside
`stream_options` only `include_usage` is read, so a misspelled key there is
ignored rather than refused.

The server supports one model and one choice. It does not support the Responses
API, legacy Completions, embeddings, structured output,
batching, log probabilities, or remote model switching.

Context length can be 4K, 8K, 16K, 32K, 64K, 96K, 128K, 192K, or 256K.
The default is 16K. Before loading the model, the server checks whether the
selected context and expert cache fit its memory estimate for your Mac and the
model's variant, which it reads from the model's manifest. The 8-bit model
needs more memory than the 4-bit one at every context, and an 8 GB Mac is
refused it even at 4K; the refusal names the variant. Larger contexts need
more memory for the FP16 KV cache. See
[context length and memory](RUNTIME_CONTROLS.md#context-length-and-memory)
for the calculation and diagnostic override, and the
[long-context report](experiments/summaries/10-long-context.md) for measured
results and validation limits. On an 8 GB Mac, run one model process at a
time and watch memory pressure.

## Logs

The server logs one line per event to stdout, with an ISO 8601 timestamp, so
`> server.log` captures it together with the ready line. Pass
`--log-output stderr` to keep the log apart from the ready line, for a script
that reads stdout. Startup errors, the usage text, and the
[`--host`](#listen-on-another-address) warning always go to stderr.

```bash
.build/release/TurboFieldfareServer --model scratch/gemma4.gturbo >> server.log
```

Each accepted request logs `accepted`, `prepared`, `queued` when it waits, and
`generating`, then ends with one of these:

- `completed`, with the prompt, cached, and generated token counts, prefill
  time and speed (`pp=`, `pp_tok_s=`), generation time and speed (`tg=`,
  `tg_tok_s=`), and `finish=`.
- `cancelled by client`, with the phase it reached.
- `failed`, with the phase, HTTP status, and error.

While a request runs, a progress line every 3 seconds gives the speed over
just those seconds, so a stall shows as a slow interval instead of being
averaged into the final figure:

```text
request chatcmpl-… prefill prompt=12800/30000 pp_tok_s=452.3 window=3.00s
request chatcmpl-… decode generated=345 context=30345/262144 tg_tok_s=41.2 window=3.00s
```

`prompt=` counts prompt tokens in the KV so far, prompt-cache hits included,
of the prompt's total; `generated=` counts reasoning tokens too, since Qwen3.6
decodes them at the same cost. `context=` is prompt plus generated tokens
against `--max-context`. The counts are the runtime's own progress reports,
per prefill chunk and per token. Set the interval with
`--progress-log-interval <seconds>`, or `0` to turn the lines off.

A request refused before it is accepted logs `request rejected` with the HTTP
status, method, path, error code, `param`, and the reason the client received,
for example a missing field or an unknown route. Prompt-cache misses and model
loading events are logged too. Lines carry token counts, timings, and field
names, never prompt text, tool arguments, headers, or request bodies.

## Images

`messages[].content` accepts `image_url` parts in user messages. An
`image_url` on any other role returns HTTP 400. The URL must be a data URL,
and `detail` must be absent or set to `auto`.

Image bytes go to disk as the request body arrives, so a large upload does not
sit in memory. Before any pixel is decoded, the server works out how many
tokens the images will occupy and rejects the request if they do not fit the
context. Bodies and images over their limits return HTTP 413.

`GET /v1/models` reports `capabilities` as `["text", "image"]` when a valid
companion pack is loaded and `["text"]` otherwise. `GET /health` reports the
same state under `vision`, as `ready`, `missing`, `invalid`, or `unsupported`.
`unsupported` means the image tower cannot run on this Mac; it requires M2 or
newer. Text requests remain available.

Choose the pack and the residency policy with `--vision-pack <dir>` and
`--vision-residency on-demand|keep-ready`.
