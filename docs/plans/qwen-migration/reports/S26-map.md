# Mappa S26 — prestazioni del prefill Qwen

Geometria e golden: come `S20-map.md` (chiusa) e `S19-map.md`. T = chunk
(128 default, 256 max). Questa mappa sostituisce la rilettura del codice:
per i passi 2-5 leggere solo gli intervalli indicati qui.

## Costi del prefill (file:riga, dopo il passo 1)

1. **Lettura expert a gruppi + attesa** — `QwenHybridForwardRunner+Prefill.swift`
   (d'ora in poi `+Prefill`) :627-683: loop dei gruppi di ≤16 (union degli id
   del blocco); `fetchRoutedExperts(layer:experts:)` :632 è **seriale**: la CPU
   legge i blob (pread, ~1.77 MB/expert) mentre la GPU è ferma; commit+await
   :680-681 dopo ogni gruppo. Con 256 token l'union arriva a ~90-120 expert →
   6-8 gruppi per layer per span. Decode: `QwenHybridForwardRunner.swift`
   :859 un solo fetch (8 expert) + commit/await :834 prima (lettura id).
2. **Un `makeBlobsArgumentBuffer` per coppia (token, slot)** — `+Prefill` :653
   (alloc `MTLBuffer` nuovo per ogni coppia (t, j) del gruppo, :648-679);
   lo shared ne fa uno per call :694. Decode: uno per layer :860, :891.
3. **Chiamate `rows: 1` dei kernel per testa** — L2 per testa q/k su slice di
   convOut: loop :232-243 (2 dispatch per riga = 2T per layer lineare);
   `encodeTopK` un token per call :557-567 (T dispatch). Il kernel accetta
   già `rows:` (QwenNorms.swift `encodeL2PerHead` :131) ma `xOffset` è solo
   la base → serve un **passo di riga** per chiamare con rows: T (pericolo
   xOffset S20). Shared expert + residual: loop per riga :702-745 (5 kernel
   per riga, tutti monoriga: `encodeGateUpSilu`/`encodeDown`/gemv8/
   `encodeSigmoid`/`encodeResidual` accettano count/offset per riga).
4. **Staging shared expert** — `stageSharedExpert` (main) :972-1001: 9 blit
   (gate/up/down × w,s,b) in `sharedBlobBuffer` (1.769.472 B) **a ogni cambio
   di layer**: prefill = 40 staging per span (:689-691), decode = 40 per
   token (:887-889). Passo 3: 40 blob persistenti all'init (~71 MB).
5. **Attese GPU per layer** — commit+await per la lettura CPU degli id :581-582
   (uno per layer per span: spezza il command buffer del chunk); più :680-681
   per gruppo. Decode: :834 per layer. Fine span :155-159 (necessaria: cache
   advance + onProgress).

## Firme `ModelExpertIO.swift` (non si tocca)

- `planRoutedExperts(layer:experts:avoidingSlots:)` :61 — `public func
  planRoutedExperts(layer: Int, experts: [Int], avoidingSlots: Set<Int> = [])
  throws -> RoutedExpertFetchPlan?`
- `planRoutedExpertsIfPossible(…) -> RoutedExpertFetchPlan?` :72 (nil se non
  sta negli slot evitando quelli dati)
- `fetchRoutedExperts(plan:)` :108 — `public func fetchRoutedExperts(plan:
  RoutedExpertFetchPlan) async throws -> [TensorView]`
- `adviseRoutedExperts(plan:)` :102 — `throws -> ExpertIOAdviceResult`
  (prefetch dei miss, non blocca)
- `routedExpertCacheSlotCount(layer:)` :88 — `Int?` (slotCount in `.pread`;
  16 per Qwen, `Model.swift` :348)
- `RoutedExpertFetchPlan` :5-18: `experts/misses/hits/assignedSlots`
- Nota: un fetch può scrivere solo in slot non usati dalla GPU; con 16 slot
  totali l'overlap (passo 4) usa due metà da 8 (avoidingSlots).

## MPPPrefillInt4QMM sui blob C5 (passo 5)

Sì. Il blob expert è affine MLX gruppo 64, lo stesso layout che MPP serve per
le proiezioni (C5 = planner Gemma). `MPPPrefillInt4QMM.encode`
(`Kernels/TensorCore/MPPPrefillInt4QMM.swift` :119) prende
weights/scales/biases come buffer+offset: passare il buffer dello slot con
`Int(v.offset) + <offset C5>` dove v è il TensorView di
`fetchRoutedExperts`/`routedExpertBuffers`. Offset da
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
  embedding/head layer -1, linear `12+2T`, attention `14`, MoE norm+router+
  topK, coppie ×2, shared staging 1, shared+residual `5T`).
- Test: `prefillAndDecodeThroughputAreMeasured` (misure + breakdown, log
  `scratch/logs/S26-measure.log`); `s26LogitHashBaselineIsRecorded`
  (SHA-256 logit → `scratch/S26/baseline.json`, hash anche nel report S26).

## Intervalli da leggere per passo (dopo il passo 1)

- Passo 2 (arg buffer + dispatch): `+Prefill` :627-745, :232-243, :557-567;
  `QwenMoE.swift` :97-177 (wrapper, `makeBlobsArgumentBuffer` riusabile:
  encoder e PSO non cambiano tra call); `QwenNorms.swift` :131; `QwenRouter`
  topK :36 (un token per call anche nel decode — non cambiare il decode).
- Passo 3 (shared 40 blob): `+Prefill` :689-694; main :888-895, :972-1001,
  buffer/init :283-288 (`sharedBlobBuffer`) e :141 (`sharedBlobLayer`).
- Passo 4 (overlap): `+Prefill` :581-683; ModelExpertIO (firme sopra).
- Passo 5 (GEMM expert): sezione C5 sopra; `+Prefill` :627-683; kernel D4 in
  `Metal/Qwen/` + wrapper in `Kernels/Qwen/`.

## Aggiunte passo 1 (misure di partenza, release, Mac16,9)

- Hash di riferimento in `scratch/S26/baseline.json` + report S26: al passo 0
  del greedy il logit hash di p0 = hash(p0 prefill) ✓; chunk 4 = chunk 128 su
  tutti i 6 prompt (bit-determinismo ✓). Decode baseline 12,77 tok/s.
- Steady state 4096/128: 2,47 s/span → gpu 2,14 s, encode CPU 0,13, fetch
  expert 0,16, muro−gpu ≈ 0,33 (13%). **121k dispatch per span**: coppie
  token-expert (t,j) gateUp+down = 81.920 (68%), shared+residual per riga
  25.600 (21%), topK per riga 5.120, mixer 8.180, altro 130. Costo medio
  dispatch ≈ 15 µs: il prefill è **dispatch-bound**, non banda.
- Chunk 256 (di default non cambia): 54,0 vs 51,8 tok/s a 4096; ma coppie
  raddoppiate per span → dispatch per token invariati.
- Per i passi 2-5: l'unità da attaccare è la coppia (t,j) monoriga; il
  riuso arg buffer (passo 2) vale 41k alloc/span a 128; l'overlap (passo 4)
  da solo vale poco (fetch 0,16 s/span a 4096, ma 0,49 a 512 con cache fredda
  per span). Passo 5 (GEMM per colonne) è dove sta il fattore T.
