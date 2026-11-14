# Mappa S26 — prestazioni del prefill Qwen

Geometria e golden: come `S20-map.md` (chiusa) e `S19-map.md`. T = chunk
(128 default, 256 max). Questa mappa sostituisce la rilettura del codice:
per i passi 2-5 leggere solo gli intervalli indicati qui.

## Costi del prefill (file:riga, dopo il passo 1)

1. **Lettura expert a gruppi + attesa (passo 4)** — risolto al passo 4
   (de45c58): gruppi da ≤8 (metà dei 16 slot), fetch del gruppo successivo
   sovrapposto alla GPU del gruppo corrente (`+Prefill` :616-753, vedi
   "Aggiunte passo 4"). Resta: commit+await per gruppo e readback id
   :576-577.
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

- Passi 2-4 (fatti, 90d60ae, db27c44 e de45c58): vedi le "Aggiunte" in
  fondo.
- Passo 5 (GEMM expert): FATTO E SALTATO, vedi «Aggiunte passo 5» sotto
  (non rileggere C5 per riprovarci: il path MPP viola C8).

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

## Aggiunte passo 4 (commit de45c58, hash identici)

- Loop gruppi `+Prefill` :616-753: gruppi da `cacheSlotCount/2` = 8
  (`fetchGroups` :629). Il fetch del gruppo i+1 parte subito dopo il commit
  del CB del gruppo i con `planRoutedExpertsIfPossible(avoidingSlots:
  Set(plan.assignedSlots))` + `fetchRoutedExperts(plan:)` in un `Task`
  (`startFetch` :638): i miss finiscono solo in slot non letti dalla GPU,
  gli hit non si riscrivono mai. Il piano di i+2 nasce solo dopo
  l'await del CB di i (`inFlightGroupBuffer`, await in testa ad ogni
  iterazione; l'ultimo CB si chiude :749-753 prima dello shared expert).
  Fallback seriale se il piano non sta (impossibile con gruppi da 8 su 16
  slot; usato solo per errori inattesi). `adviseRoutedExperts` non serve:
  il fetch è già un giro in anticipo. Decode e shared expert non toccati.
- Misure 4096/128: **78,8 tok/s** (72,5 dopo p3), gruppi/span 505 (erano
  262), command buffer/span ~546, expertFetch 0,175 s/span ormai nascosto
  (muro−GPU ~0,20 s/span); 512/128 (cache fredda) **85,0** (71,4). Decode
  12,83 invariato, hash passo 1 identici. GPU busy ~87%: MoE coppie
  monoriga = voce dominante → **passo 5 conviene**. Prossimo: GEMM per
  colonne (C5 in alto, kernel D4 nuovi).

## Aggiunte passo 5 (TENTATO E SALTATO; nessun commit)

- Implementato il path a colonne per gli expert routed: kernel nuovi D4
  (`qwen_moe_gather_rows`, `qwen_moe_silu_mul`, `qwen_moe_scatter_rows` in
  `qwen_moe.metal`), wrapper `encodeGatherRows/encodeSiluMul/
  encodeScatterRows`, buffer scratch nel runner, loop gruppi con
  gather → gate/up (`encodePrefillProjection` + `shifted` per gli offset
  C5) → SiLU·mul in-place → down → scatter su `routedOut[t][slot]` (somma
  pesata della residual invariata). Kernel test verdi (4/4, riferimento
  CPU, piccole + dimensioni reali). Tentativo completo in
  `scratch/s26-step5-attempt/` (patch + test).
- **Risultato: soglie C8 fuori portata.** Il kernel di controllo MPP
  dequantizza i pesi INT4 in FP16 (`tensorops.metal:80`
  `half(fma(q,scale,bias))`), il GEMV per coppia usa FP32: errore di peso
  ~5e-4 per riga → MoE layer 0 vs golden 0,99822 (prima 0,99997), hidden
  layer 3 0,903 (prima 0,99909), logit p0 vs golden 0,414, top-1 8/12.
  Amplificazione nota dai quasi-pareggi top-8 (problema 1 di S20, ~50×
  più forte). Il path a colonne NON è recuperabile con MPP `.control`;
  `.apple10V1` richiede m ≥ 64 (gli expert ne hanno ~10) e gli stessi
  pesi FP16. Anelli di salvataggio: GEMV multi-riga nuovo (aritmetica
  identica ai GEMV, non è il GEMM della scheda — non provato).
- Revert completo a de45c58: hash passo 1 ripristinati e riverificati
  (`scratch/logs/S26-step5-verify.log`, tutti identici). Il prefill
  resta a 78,8 tok/s a 4096/128. Per il passo 6: la tabella "dopo" è
  quella del passo 4; nessuna misura nuova da fare sul path MoE.

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
- Riepilogo passo 4 (de45c58): fetch gruppo i+1 sovrapposto alla GPU del
  gruppo i (2 metà da 8 slot, avoidingSlots); hash identici, 78,8 tok/s a
  4096/128, decode 12,83; prossimo: passo 5 (GEMM per colonne, ne vale la
  pena: GPU 87%, coppie monoriga dominate).

## Aggiunte passo 6 (chiusura)

- Misure finali (hash passo 1 riverificati identici): 87,3 / 79,0 / 102,1 /
  82,7 tok/s (512/4096 × chunk 128/256), decode 12,82; tabella prima/dopo e
  proposta default 256 token per Qwen nel report (non applicata:
  `defaultChunked` è condiviso con Gemma, serve override per famiglia).
- Suite completa: verde tranne i 7 problemi baseline S1 in
  `TurboFieldfareMacPresentationTests`. Sezione chiusa, nessuna misura
  MoE residua da fare.
