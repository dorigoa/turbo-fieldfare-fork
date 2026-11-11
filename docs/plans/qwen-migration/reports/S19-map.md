# Mappa S19 — runner token per token

Formule: piano 3.5. Tolleranze: C8. Golden: C7/Appendice D. Cache: D5/D6.
Offset buffer in byte; i wrapper Qwen hanno `= 0` per default su tutti gli
offset (omessi qui). Geometria reale: hidden 2048, vocab 248320, 40 layer
(mask 1 nei layer 3,7,…,39), DeltaNet Hk 16/Hv 32/dk dv 128/conv 8192,
attention 16 Q/2 KV/256, MoE 256 expert top-8, inter 512, eps 1e-6.

## Protocolli (Runtime/Generation/LogitProducer.swift)
- `LogitProducer` :6 — `reset()` :8; `produce(token: Int32, position: Int, into logits: MTLBuffer) async throws` :10.
- `ContinuableLogitProducer` :13 — `continuationPosition: Int`; `prepareForContinuation(expectedPosition: Int) throws` :15. D6: la continuazione in coda è sempre legale → verifica `expectedPosition == cache.position`.
- `ContextWindowReporting` :18 — `maxContext: Int`.
- Classe: `public final class QwenHybridForwardRunner: …, @unchecked Sendable` (come RealForwardRunner:209); `QwenHybridCache` non è Sendable.
- Senza `ChunkedPrefillRunner` il prefill arriva token per token (RawCompletion.swift:172): `produce` serve prefill e decode.

## Init
- `Model.load(directoryURL:device:)`; `model.config == .qwen36_35B_A3B`, `model.qwenHybridConfig == .qwen36_35B_A3B`. `MetalContext()` :123 (MetalContext.swift): `.device` :93, `.queue` :94.
- `QwenHybridCache(device:config:hybrid:maxContext:) throws` (KVCache/Qwen/QwenHybridCache.swift:111).
- Wrapper, tutti `init(context:) throws`: `QwenNorms` QwenNorms.swift:37; `QwenDeltaNetPrep` QwenDeltaNetPrep.swift:26; `QwenDeltaNetScan` QwenDeltaNetScan.swift:27; `QwenAttention` QwenAttention.swift:36; `QwenRouter` QwenRouter.swift:28; `QwenMoE` QwenMoE.swift:50; `DequantInt4GEMV` Quant/DequantInt4GEMV.swift:24; `DequantInt8GEMV` Quant/DequantInt8GEMV.swift:37; `EmbedLookupInt4` Quant/EmbedLookupInt4.swift:14.
- Quant reale (manifest): attention/embedding/routedExpert/sharedExpert 4 bit, router 8 bit → dense con `DequantInt4GEMV`, router e `shared_expert_gate` con `DequantInt8GEMV`.
- `TensorView` (ModelTypes.swift:221): `buffer/offset/length/scaleOffset/scaleLength/biasOffset/biasLength/shape/dtype`. GEMV tipo: `gemv.encode(commandBuffer: cb, weights: v.buffer, weightsOffset: Int(v.offset), scales: v.buffer, scalesOffset: Int(v.scaleOffset), biases: v.buffer, biasesOffset: Int(v.biasOffset), x: xb, xOffset: 0, y: yb, yOffset: 0, m: M, n: N)`.
- `normWeightsShifted = true` → sempre `QwenNorms.WeightMode.plain` (QwenNorms.swift:12-15): i pesi sono già +1.

## Accessori Model
- Gemma riutilizzabili (nomi identici in Qwen): `embedding` Model.swift:98; `qProj/kProj/vProj/oProj` :106-115; `inputNorm/postAttnNorm` :130-134; `finalNorm` :139; `qNorm/kNorm` :150-153.
- Qwen (Inference/Qwen/Model+Qwen.swift): `qwenHybridConfig` :29; `qwenLMHead` :37; `qwenInProjQKV/Z/A/B` :41-52; `qwenConv1D` :53; `qwenALog` :56; `qwenDtBias` :59; `qwenDeltaNetNorm` :62; `qwenOutProj` :65; `qwenRouter` :71; `qwenSharedExpertGate/Up/Down` :74-82; `qwenSharedExpertGateScalar` :84.
- NON usare `lmHead`/`router`/`sharedExpert*` di Model.swift (sono Gemma, tied).

## produce(token:position:into:) — per token (h FP16 [2048])
1. `EmbedLookupInt4.encode(model.embedding…, tokenId: UInt32(token), d: 2048, outScale: 1.0)` (EmbedLookupInt4.swift:22) — senza sqrt(hidden) (3.5).
2. Layer 0…39 secondo `config.fullAttentionLayerMask` (sezioni sotto).
3. `QwenNorms.encodeRow(x: h, weight: model.finalNorm, d: 2048, rows: 1, eps: 1e-6, mode: .plain)` (QwenNorms.swift:48).
4. `DequantInt4GEMV.encode(model.qwenLMHead…, m: 248320, n: 2048, y: logits)` — logit FP16 nel buffer del chiamante; niente softcap.
5. `cache.advance(by: 1)`.

## Layer lineare L (DeltaNet)
1. `x = encodeRow(h, inputNorm(L), .plain)` [2048] (= golden `x_norm`).
2. `qkv = Int4GEMV(qwenInProjQKV(L), 8192×2048)`; `z = …(qwenInProjZ(L), 4096×2048)`; `a`, `b = …(qwenInProjA/B(L), 32×2048)`.
3. `encodeConvSilu(qkv, convWeight: qwenConv1D(L), stateIn/Out: cache.convStateBuffer + cache.convStateByteOffset(L), y: convOut, channels: 8192, tokens: T)` (QwenDeltaNetPrep.swift:38) — stato conv FP32 [3 tap][8192], tap 0 = più vecchio (S14).
4. Split per offset su convOut (2 B/elem): q [0,2048), k [2048,4096), v [4096,8192).
5. `encodeL2PerHead` (QwenNorms.swift:131) su q e k: headDim 128, numHeads 16, rows T, eps 1e-6.
6. `encodeScale(q, scale: 1/sqrt(128), n: 2048·T)` (QwenDeltaNetPrep.swift:104) — solo q.
7. `encodeGates(a, b, aLog: qwenALog(L), dtBias: qwenDtBias(L), heads: 32, tokens: T)` (QwenDeltaNetPrep.swift:70) → beta, decay FP32 [T][32].
8. `QwenDeltaNetScan.encodeScan(q, k, v: convOut (vOffset 8192, vRowStride 8192), beta, decay, state: cache.deltaStateBuffer + cache.deltaStateByteOffset(L) IN-PLACE, o: coreOut, tokens: T, valueHeads: 32, keyHeads: 16, keyDim: 128, valueDim: 128)` (QwenDeltaNetScan.swift:45).
9. `encodeGatedPerHead(o: coreOut, z, weight: qwenDeltaNetNorm(L), headDim: 128, numHeads: 32, rows: T, eps: 1e-6)` (QwenNorms.swift:103) → gated_out [T][4096]; peso sempre w.
10. `mixer = Int4GEMV(qwenOutProj(L), 2048×4096)`; `encodeAdd(h, mixer, out: h, n: 2048·T)` (QwenDeltaNetScan.swift:96).

## Layer di attention L
1. `x = encodeRow(h, inputNorm(L), .plain)`.
2. `qg = Int4GEMV(qProj(L), 8192×2048)` [T][16·512]; `k = …(kProj(L), 512×2048)`; `v = …(vProj(L), 512×2048)`.
3. `encodeSplitQG(qg, q, gate, queryHeads: 16, headDim: 256, rows: T)` (QwenAttention.swift:47).
4. `encodePerHead` (QwenNorms.swift:74, `.plain`) su q (w `qNorm(L)`, 16×256) e k (w `kNorm(L)`, 2×256).
5. `encodeRopePartial(data: q, positionBase: position, headDim: 256, numHeads: 16, rows: T, rotatedPairs: 32, theta: 1e7)` (QwenAttention.swift:77) in-place; idem k (numHeads 2).
6. `layerKV = cache.kvByteOffset(layer: L, position: 0)!`; `encodeKVWrite(k, v, kvK: cache.kBuffer + layerKV, kvV: cache.vBuffer + layerKV, rows: T, kvHeads: 2, headDim: 256, kvStart: position, posStride: 512, headStride: 256)` (QwenAttention.swift:115).
7. `encodeAttention(q, kvK, kvV, out: attnOut, queryHeads: 16, kvHeads: 2, headDim: 256, queryTokens: T, kvStart: position, posStride: 512, headStride: 256, scale: 1/sqrt(256))` (QwenAttention.swift:161) — causale, query t vede [0, kvStart+t].
8. `encodeGate(attnOut, gate, out: gated [T·4096], count: T·4096)` (QwenAttention.swift:213).
9. `mixer = Int4GEMV(oProj(L), 2048×4096)`; `encodeAdd(h, mixer, h)`.

## Blocco MoE (tutti i layer)
1. `x = encodeRow(h, postAttnNorm(L), .plain)` (= golden `moe_x`).
2. `routerLogits = Int8GEMV(qwenRouter(L), 256×2048)` FP16 [256]; `encodeTopK(logits: routerLogits, indices, weights, experts: 256, topK: 8)` (QwenRouter.swift:36) → ids UInt32[8], w FP32[8] ricalibrati.
3. Routed: `blobs = try await model.fetchRoutedExperts(layer: L, experts: ids)` (Inference/ModelExpertIO.swift:126; default `.pread(slotCount: 16)` Model.swift:348); i buffer slot partono dall'inizio del blob (offset 0). `offsets = model.routedExpertOffsets(layer: L)` :21 (`MoEExpertOffsets`, Kernels/MoE/MoE.swift:5 — stessi 9 campi di `QwenMoE.ExpertOffsets`). `argBuf = moe.makeBlobsArgumentBuffer(blobs: blobs.map(\.buffer), slots: 8)` (QwenMoE.swift:97).
4. `encodeGateUpSilu(argBuf, blobs, offsets, x, acts [8·512], hidden: 2048, inter: 512, slots: 8)` (QwenMoE.swift:114); `encodeDown(…, routedOut [8·2048], …)` :148.
5. Shared expert: tre tensori residenti `qwenSharedExpertGate/Up/Down(L)` (affine 4 bit) → staging in UNA `MTLBuffer` C5 da 1,769,472 B con gli offset di `QwenMoE.expertOffsets(hidden: 2048, inter: 512)` (QwenMoE.swift:66); poi gli stessi due kernel con `slots: 1` (`sharedActs [512]`, `sharedOut [2048]`).
6. `sg: Int8GEMV(qwenSharedExpertGateScalar(L), 1×2048)` → FP16 [1]; `encodeSigmoid(count: 1)` (QwenRouter.swift:63).
7. `encodeResidual(residual: h, routedOut, sharedOut, routingW, sg, y: h2, hidden: 2048, slots: 8)` (QwenMoE.swift:183) — scrive h + Σw·routed + sg·shared (= golden `layer_out`).

## Prese di debug (solo test, passi 2-7)
- Tap del hidden state: dopo l'embedding (layer -1) o dopo un layer scelto → buffer FP16 [T][2048] leggibile dal test (per `hidden_p0.json`).
- Tap delle grandezze interne del layer (nomi = chiavi golden): `x_norm, qkv_raw, z, a, b, conv_out, q, k, v, beta, g, core_out, gated_out, mixer_out, moe_x, router_top_ids, router_top_weights, routed_out, shared_out, layer_out, state_heads0to3` (layer 0) e `x_norm, q_pre_norm, gate, k_pre_norm, v, q_rope, k_rope, attn_out, gated, mixer_out, layer_out` (layer 3).
- Il kernel gates scrive solo beta/decay: `g` si ricava nel tap in CPU: `g = -exp(A_log)·softplus(a + dt_bias)` (a FP16, A_log/dt_bias BF16), o si aggiunge un buffer.

## Golden (Tests/QwenFixtures/model/) — forme verificate
- `prompts.json`: lista `{name, text, kind, ids}`; T: p0 5, p1 15, p2 10, p3 5, p4 28, p5 15. p0 ids [760, 6511, 314, 9338, 369]. "Primi 8 token di p0" = tutti i 5 → **T = 5** nei file di interni.
- `hidden_p0.json`: `embedding, layer0, layer3, layer39, final_norm` — [5,2048] f32b64.
- `layer0_internals.json`: `x_norm [5,2048]; qkv_raw [5,8192]; z [5,4096]; a,b [5,32]; conv_out [5,8192]; q,k [5,16,128] (dopo L2; q già ÷sqrt(128)); v [5,32,128]; beta,g [5,32]; core_out,gated_out [5,32,128]; mixer_out [5,2048]; moe_x [5,2048]; router_top_ids [5]×8 int; router_top_weights [5,8]; routed_out, shared_out [5,2048] (shared_out già ×sg); layer_out [5,2048]; state_heads0to3 [4,128,128]`.
- `layer3_internals.json`: `x_norm; q_pre_norm, gate [5,16,256]; k_pre_norm, v [5,2,256]; q_rope, k_rope; attn_out [5,16,256] (pre-gate); gated [5,4096]; mixer_out, layer_out [5,2048]`.
- `logits_topk.json`: per prompt `top_ids` (20 int per posizione) + `top_logits [T,20]` f32. `logits_last.json`: p0/p1 → `logits [248320]` f16b64. `greedy.json`: 32 token per prompt.
- ATTENZIONE: la cache DeltaNet è [testa][colonna c][riga i], il golden `state_heads0to3` è [testa][riga i][colonna c] → trasporre prima del confronto.
- Numeri C7: oggetto `{"shape": [...], "f32b64"|"f16b64": base64}`.

## Tolleranze C8
- Kernel vs CPU: `Tolerance` esistenti (fp16Reduction 5e-3, fp16ChainedReduction 1e-2, quantInt4 1.5e-3). Stato DeltaNet: rel err ≤ 1e-4.
- Modello: coseno logit completi ≥ 0,999 (ultime posizioni p0/p1); top-1 uguali ≥ 98% delle posizioni; greedy identici ≥ 4/6 prompt; diagnostica layer: coseno stati ≥ 0,995 (layer 0, 3), ≥ 0,99 (layer 39); grandezze interne layer 0/3 coseno ≥ 0,995; embedding passo 2 coseno ≥ 0,9999.

## Test (C9)
- `QwenTestResources` (Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTestResources.swift): `.modelURL`, `.fixtures`, `.hasModel`; gating `@Test(.enabled(if: QwenTestResources.hasModel))`.
- Helper da scrivere nel test: decodifica `f32b64`/`f16b64` → [Float]/[Float16], coseno, lettura buffer GPU (`buffer.contents()`).
- Test del passo 2: embedding dei 5 token di p0 contro `hidden_p0.json → embedding` (coseno ≥ 0,9999).

## Note per i passi successivi
- Passo 2 completato (44d19cc): `QwenHybridForwardRunner` in Runtime/Inference/Qwen/ (file `QwenHybridForwardRunner.swift`).
  - `init(model:context:maxContext:) throws`; famiglia non Qwen → `ModelError.archMismatch`. `maxContext` è `public let`; `continuationPosition` = `cache.position`; `prepareForContinuation` butta `PrefillError.prefillCursorMismatch` (D6).
  - `produce`: guardie di posizione (`prefillCursorMismatch` / `ModelError.contextExceedsModel`), embedding in `hiddenBuffer` (`outScale: 1.0`), tap, (layer qui, passi 3–6), final norm → `normedBuffer` (`.plain`, eps `hybrid.rmsNormEps`), tap, lm_head `gemv4.encode(x: normedBuffer, y: logits, m: vocab, n: hidden)`, `commit()` + `await commandBuffer.completed()` + `cache.advance(by: 1)`. In contesto async `waitUntilCompleted` è vietato: usare `await completed()`.
  - Buffer di lavoro T=1 (campi `…Buffer`, storageModeShared): `hidden/normed/xNorm/mixer/moeX` [2048] FP16; `qkv/convOut` [8192]; `z/coreOut/gatedOut/q/gate/attnOut/gated` [4096]; `a/b` [32] FP16; `beta/decay` [32] FP32; `routerLogits` [256] FP16; `routerTopIds` [8] UInt32; `routerTopWeights` [8] FP32; `expertActs` [8·512]; `routedOut` [8·2048]; `sharedBlob` 1,769,472 B (C5); `sharedActs` [512]; `sharedOut` [2048]; `sharedGate` [1]; `qg` [8192]; `attnK/attnV` [512]. Derivate: `hiddenSize/vocabSize/valueDim/qkvRows/queryDim/qgRows/kvDim`. Kernel già inizializzati: embedding, gemv4, gemv8, norms, deltaPrep, deltaScan, attention, router, moe.
  - Tap (solo test): `debugHiddenTap = HiddenTap(points:rows:)` con punti `.afterEmbedding/.afterLayer(L)/.afterFinalNorm`; riga = punto i, token t → `i*rows+t` di `tappedHidden` (FP16; blit GPU dentro l'encode, ordine garantito); `tappedHiddenCount`. `debugInternalsLayer = L` + `recordInternals(layer:entries:)` nel codice del layer (chiavi = nomi C7) → `tappedInternals: [String: MTLBuffer]`: sono buffer di lavoro, il layer tapato deve essere l'ultimo eseguito. `reset()` azzera cache e tap.
- Passo 3 completato (1759178): mixer DeltaNet in `runLinearLayer(L, in:)`, chiamato nel loop dei layer di `produce` dove `fullAttentionLayerMask[L] == 0`; `recordHiddenTap(.afterLayer(L))` dopo ogni layer.
  - `debugStopAfterLayer: Int?` (solo test): ferma il loop dopo il layer L (prima di final norm/lm_head) così i buffer di lavoro del layer tapato sopravvivono a `tappedInternals`; `produce` fa comunque `cache.advance(by: 1)`.
  - Buffer nuovi: `dnQBuffer`/`dnKBuffer` [Hk·dk] FP16 — q e k dopo L2 (q anche ÷sqrt(128)); `convOutBuffer` resta il conv+SiLU crudo (tap `conv_out`), `v` è la sua slice a 2·qkBytes.
  - Test: i buffer di lavoro reggono 1 riga → per confrontare T righe si **ri-gira ogni prefisso** di p0 (`reset()`, poi `produce` 0…end, end = 0..<T): la riga tapata al prefisso end è quella del token end (stato deterministico, uguale alla run completa). Helpers nel test: `readFP16/readFP32/readBF16(buffer, byteOffset, count)`, `softplus`, `cosine`; `golden()` ora casta loose a `[String: Any]` perché i file di interni mescolano oggetti C7 e array grezzi (`router_top_ids`) — per quelli scrivere una lettura dedicata. `cache` è `internal let` per gli offset di stato.
- Passo 4: MoE con i buffer del passo 2 (`moeX`, `routerLogits`, `routerTopIds`, `routerTopWeights`, `expertActs`, `routedOut`, `sharedBlob` 1,769,472 B, `sharedActs`, `sharedOut`, `sharedGate`); shared expert = staging dei 3 tensori residenti in `sharedBlobBuffer` con `QwenMoE.expertOffsets(hidden: 2048, inter: 512)`; verificare la firma di `encodeSigmoid` (in-place su `sharedGateBuffer` o buffer separato). Il blocco MoE va dopo il mixer nello stesso loop (tutti i 40 layer); tap `moe_x, router_top_ids, router_top_weights, routed_out, shared_out, layer_out` nel layer tapato. `router_top_ids` è un array grezzo di interi nel golden (non un oggetto C7). Test: ri-gioco dei prefissi come nel passo 3 + `debugStopAfterLayer = 0`; hidden dopo il layer 0 da `tappedHidden` (`.afterLayer(0)`) vs `hidden_p0.json → layer0` (coseno ≥ 0,995).
- Passo 4 completato (438e2cb): `runMoELayer(L, in: &commandBuffer)` dopo ogni mixer, in tutti i layer. Il router chiude il command buffer, legge gli id sulla CPU (ordinati come il golden: peso crescente, a pari peso id crescente), `fetchRoutedExperts`, poi riparte con un command buffer nuovo (per questo `inout`). Shared expert: `stageSharedExpert` (blit dei 3 tensori nel blob C5) rifatto quando cambia il layer. `routedSumBuffer` (golden `routed_out`) si calcola solo nel layer tapato. Tap `shared_out` = proiezione prima del gate sg (il coseno non cambia).
  - Presa nascosta corretta: ogni `produce` scrive la riga `tappedHiddenCount` per tutti i punti e la chiude con `finishHiddenTapRow()`: con più punti (es. `.afterLayer(0)`, `.afterLayer(3)`) le righe restano allineate.
- Passo 5: `runAttentionLayer(L, in:)` dove `fullAttentionLayerMask[L] == 1`, seguito da `runMoELayer`; buffer già allocati: `qg/q/gate/attnK/attnV/attnOut/gated`. KV in `cache.kBuffer/vBuffer` a `cache.kvByteOffset(layer:position:)`.
- Passo 5 completato (23b549e): `runAttentionLayer(L, position:in:)` dove `fullAttentionLayerMask[L] == 1`, poi `runMoELayer` come per i layer lineari. Buffer nuovi `qNormedBuffer`/`kNormedBuffer` (norma per testa, poi RoPE in place); `qPreRopeTapBuffer`/`kPreRopeTapBuffer` solo per le prese (golden `q_pre_norm`/`k_pre_norm` = dopo norma, prima del RoPE). KV: `cache.kvByteOffset(layer: L, position: 0)`, `posStride` = kvHeads·headDim = 512, `headStride` = 256 (elementi); `rotatedPairs` = headDim·partialRotaryFactor/2 = 32, theta `cfg.ropeTheta`.
- Passo 6: il ciclo dei layer è già completo (lineari + attention + MoE); resta da verificare il modello intero: layer 39, norma finale, top-20 su tutti i prompt, logit completi di p0/p1.
