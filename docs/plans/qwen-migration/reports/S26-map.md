# Mappa S26 — prestazioni del prefill Qwen

Geometria e golden: come `S20-map.md` (chiusa) e `S19-map.md`. T = chunk
(128 default, 256 max). Questa mappa sostituisce la rilettura del codice:
per i passi 2-5 leggere solo gli intervalli indicati qui.

## Costi del prefill (file:riga, dopo il passo 1)

1. **Lettura expert a gruppi + attesa (passo 4)** — `+Prefill` :627-684:
   loop dei gruppi di ≤16 (union degli id del blocco); `fetchRoutedExperts`
   :627 **seriale** (pread ~1,77 MB/expert con GPU ferma); commit+await
   :683-684 dopo ogni gruppo; union ~90-120 expert a 256 token → 6-8 gruppi
   per layer per span. Decode: un solo fetch + commit/await prima.
2. **Arg buffer per coppia (token, slot)** — risolto al passo 2: uno per
   expert del gruppo (`groupArgBuffers` :641-654), shared uno per call.
3. **Chiamate `rows: 1` (passo 5)** — risolto al passo 2 per topK e L2 q/k.
   Restano: coppie (t,j) gate/up+down monoriga e la catena shared+residual
   per riga `+Prefill` :705-748 (5 kernel/riga: `encodeGateUpSilu`/
   `encodeDown`/gemv8/`encodeSigmoid`/`encodeResidual`, tutti con
   count/offset per riga).
4. **Staging shared expert** — risolto al passo 3: 40 blob persistenti
   all'init, nessuno staging per layer né in prefill né in decode.
5. **Attese GPU per layer (quota del passo 4)** — commit+await per la
   lettura CPU degli id :576-577 (uno per layer per span) più :683-684 per
   gruppo. Fine span :155-159 (necessaria: cache advance + onProgress).

## Firme `ModelExpertIO.swift` (non si tocca)

- `planRoutedExperts(layer:experts:avoidingSlots:)` :61 — `(layer: Int,
  experts: [Int], avoidingSlots: Set<Int> = []) throws -> RoutedExpertFetchPlan?`
- `planRoutedExpertsIfPossible(…) -> RoutedExpertFetchPlan?` :72 (nil se non
  sta negli slot evitando quelli dati)
- `fetchRoutedExperts(plan:)` :108 — `(plan: RoutedExpertFetchPlan) async
  throws -> [TensorView]`
- `adviseRoutedExperts(plan:)` :102 — `throws -> ExpertIOAdviceResult`
  (prefetch dei miss, non blocca)
- `routedExpertCacheSlotCount(layer:)` :88 — `Int?` (16 per Qwen, `.pread`)
- `RoutedExpertFetchPlan` :5-18: `experts/misses/hits/assignedSlots`
- Nota: un fetch può scrivere solo in slot non usati dalla GPU; con 16 slot
  totali l'overlap (passo 4) usa due metà da 8 (avoidingSlots).

## MPPPrefillInt4QMM sui blob C5 (passo 5)

Sì. Il blob expert è affine MLX gruppo 64, lo stesso layout che MPP serve per
le proiezioni (C5 = planner Gemma). `MPPPrefillInt4QMM.encode`
(`Kernels/TensorCore/MPPPrefillInt4QMM.swift` :119) prende weights/scales/
biases come buffer+offset: passare il buffer dello slot con `Int(v.offset) +
<offset C5>` (v da `fetchRoutedExperts`). Offset da
`QwenMoE.expertOffsets(hidden: 2048, inter: 512)` (`Kernels/Qwen/QwenMoE.swift`
:66): gate w/s/b 0 / 524288 / 557056; up 589824 / 1114112 / 1146880; down
1179648 / 1703936 / 1736704; stride blob 1769472. Guardie MPP :145-152:
scales/biases % 2 (✓), k % 64 (2048, 512 ✓), x/y % 2 ✓. Gate/up: n=512 k=2048;
down: n=2048 k=512. Path Apple10 veloce con m ≥ 64 e k % 128 == 0 (✓);
`mpp4` del runner è `.control` (path F16, tileM 64). Servono kernel nuovi
(D4) per: gather righe token→buffer contiguo, SiLU·mul su [T][512]×2,
somma pesata in `routedOut[t][j]` nello stesso ordine di S19.

## Contatori S26 (passo 1, solo test)

- `S26PrefillCounters` in `QwenHybridForwardRunner.swift` :1070-1200:
  cpuEncode (make→scheduled), gpu (Σ gpuEnd−gpuStart), await (scheduled→
  completed), per layer groups/expertsFetched/dispatches/argumentBuffers/
  expertFetchSeconds; `waitForDrain()` prima di leggere.
- Attivazione: `runner.s26Counters = Counters()` (+Prefill nota ai blocchi:
  embedding/head layer -1, linear `12`, attention `14`, MoE norm+router+topK
  `3`, coppie ×2, shared+residual `5T`).
- Test: `prefillAndDecodeThroughputAreMeasured` (misure + breakdown, log
  `scratch/logs/S26-measure.log`); `s26LogitHashBaselineIsRecorded`
  (SHA-256 logit → `scratch/S26/baseline.json`, hash anche nel report S26).

## Intervalli da leggere per passo (dopo il passo 1; righi aggiornati al 2)

- Passi 2-3 (fatti, 90d60ae e db27c44): vedi le "Aggiunte" in fondo.
- Passo 4 (overlap): `+Prefill` :576-686 (readback id :576-577, poi loop
  gruppi: fetch :627, arg buffer :641-654, commit+await :683-684);
  ModelExpertIO (firme sopra).
- Passo 5 (GEMM expert): sezione C5 sopra; `+Prefill` :657-686 (loop
  coppie); kernel D4 in `Metal/Qwen/` + wrapper in `Kernels/Qwen/`.

## Aggiunte passo 2 (commit 90d60ae, hash identici)

- `qwen_l2norm_perhead` accetta `xRowStride` (buffer(5); default compatto →
  decode invariato): loop L2 q/k su convOut = 2 chiamate/span-layer
  (`+Prefill` :225-242). `qwen_router_topk` multiriga (grid height = rows;
  regola Metal: i built-in di posizione tutti scalari o tutti vettori
  uguali); `encodeTopK` ha `rows: Int = 1`, il decode non lo passa.
- Arg buffer routed: uno per expert del gruppo (≤16) riusato da tutte le
  righe (`groupArgBuffers` :641-654). Contatori: argumentBuffers = gruppi,
  linear 12, router 3. Misure 4096/128: 73,0 tok/s, GPU/span 1,47 s,
  arg buffer 3.943/span, decode 12,82.

## Aggiunte passo 3 (commit db27c44, hash verificati identici)

- `sharedBlobBuffers: [MTLBuffer]` (40 blob C5, 1.769.472 B l'uno = 70,8 MB
  residentes in più, per S27) creati e riempiti una sola volta in
  `stageAllSharedExpertBlobs` (funzione file-locale in fondo a
  `QwenHybridForwardRunner.swift`, blit unico + `waitUntilCompleted`
  nell'init). `sharedBlobLayer`/`stageSharedExpert` rimossi; decode (main
  :884-899) e prefill (`+Prefill` :687-701) usano `sharedBlobBuffers[L]`,
  bit-identico allo staging vecchio (intermediateSize == moeInter == 512).
- Misura: +0/1% sul muro (staging non era un collo: GPU busy 84%, attese
  per gruppo = passo 4). moeDispatches/span 108.140 (−40 staging). Decode
  12,83 invariato. Prossimo: passo 4 (overlap fetch gruppo N+1 con GPU
  gruppo N, due metà da 8 slot).

## Aggiunte passo 1 (misure di partenza; tabella completa nel report S26)

- Baseline: hash in `scratch/S26/baseline.json` (greedy passo 0 = hash p0 ✓,
  chunk 4 = chunk 128 ✓); 4096/128 = 2,47 s/span con 121k dispatch
  (coppie (t,j) 81.920, shared+residual 25.600, topK 5.120, mixer 8.180),
  costo medio ≈ 15 µs/dispatch: il prefill è **dispatch-bound**. Chunk 256
  non cambia i dispatch per token.
- Per i passi 2-5: l'unità da attaccare è la coppia (t,j) monoriga;
  l'overlap (passo 4) da solo vale poco (fetch 0,16 s/span a 4096, 0,49 a 512
  con cache fredda). Passo 5 (GEMM per colonne) è dove sta il fattore T.
- Riepilogo passo 1 (75e4625): mappa, contatori, baseline hash; dispatch-bound.
- Riepilogo passo 2 (90d60ae): L2/topK batchati, arg buffer per expert;
  hash identici, 73,0 tok/s a 4096, decode invariato.
- Riepilogo passo 3 (db27c44): 40 blob shared persistenti (+70,8 MB);
  hash identici, 72,5 tok/s (rumore), decode 12,83; prossimo: passo 4.
