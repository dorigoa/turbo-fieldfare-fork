# S25 map — file grandi mappati (passo 1)

Riferimenti `file:riga` e firme esatte. Sostituisce la rilettura dei file grandi.

## MultimodalConversation.swift (852 righe, Sources/TurboFieldfare/Runtime/Generation/)
- `public actor MultimodalConversation` :282. Stored :283-291: `model: Model`,
  `context: MetalContext`, `tokenizer: GFTokenizer`, `runner: RealForwardRunner`,
  `scratch`, `visionRuntime: VisionRuntime?`, `visionRuntimeError: Error?`,
  `visionResidency`, `maxContext`. init :316-334 (tutti i parametri, default su
  visionRuntime/Error/Residency). Stato :294-314: `kvTokenIDs`, `pending`, `closed`,
  `generating`, `resetting`, `lineageBroken`, `uncommittedBoundary`, `boundaryNeedsReplay`.
- `send(parts:images:config:prefillConfig:checkCancellation:shouldStop:onProgress:)` :452.
- `generate` :480: guardia immagini :497-502 (visionRuntime == nil →
  `.imageUnavailable(reason: visionRuntimeError…)`); `positionBefore =
  runner.continuationPosition` :544; `runRawCompletion(producer: runner,
  tokenizer: tokenizer, …, start: cached == 0 ? .reset : .resume(cachedPromptTokens:
  cached))` :548-578; `.token`/`.tail` → `text`.
- catch :579-601: se KV avanzata, `restoreAfterFailure(positionBefore:generationError:
  rewind: runner.rewind(to:) reset: runner.reset)`; fallito → `lineageBroken`, throw.
  Stop-string :618-629: `trimHiddenStopTokens(kvBacked, withheld: rewind:
  runner.rewind(to:) reset: runner.reset)`; fallito → `lineageBroken`. I DUE soli
  call site di `rewind` sono :593 e :623.
- `MultimodalConversationKVRecovery` :184-229: `restoreAfterFailure(positionBefore:
  generationError:rewind:(Int)throws->Void reset:()->Void)` :185 (positionBefore==0 →
  reset); `trimHiddenStopTokens(_:withheld:rewind:reset:)` :204 (target==0 → reset).
  rewind che lancia → `lineageRecoveryFailed` → oggi `lineageBroken = true`. Qui entra
  il fallback D6 del passo 3 (reset + rigioco, non "condanna").
- `reset()` :410-436 (attende decode, `runner.reset()`); `invalidate()` :344; `close()` :438.
- `restore(lineage:prefillConfig:checkCancellation:onProgress:)` :669-792: KV vuota :678;
  `runRawPrefill(producer: runner, promptIds: lineage.tokenIDs, multimodalInput:,
  start: .reset, outputMode: .greedyIfAvailable, …)` :763-775; errore → `runner.reset()`
  + stato azzerato :782-786. Pattern "reset + rigioco" del passo 3 (token nudi, non
  `MultimodalTurnRecord`).
- `encodeTurn(parts:images:checkCancellation:)` :799-851 — astrazione del renderer:
  primo turno :807-811 `tokenizer.encode(try tokenizer.applyChatTemplate(
  [GFTokenizer.Message(role: .user, content: text)]), addBOS: false)`; continuazione
  :812 `tokenizer.encodeTextContinuation(userContent: text)`; immagini :828
  `tokenizer.encodeMultimodalUserContinuation(textAndImages:imageTokenCounts:
  openingConversation: kvTokenIDs.isEmpty)` (solo Gemma).
- Immagini nel loop: RawCompletion.swift:125 `producer is any MultimodalPrefillRunner`,
  altrimenti errore :152. Errori: `MultimodalConversationError` :14-46.

## RealInferenceClient.swift (1042 righe, Sources/TurboFieldfareApp/Core/Inference/)
- `public final class RealInferenceClient: AppModelLifecycleClient` :60; `generate`
  :175 → `AsyncThrowingStream<AppInferenceEvent, Error>`; `restoreConversation` :115;
  `resetConversation(epoch:)` :101. `actor RealInferenceSession` :256;
  `conversationTokens: Mutex<Int>` :274; vision :288/:291.
- `ensureLoaded(key:onState:)` :396-496: tokenizer :417-425 via `Self.loadTokenizer(for:)`
  :498 = `GFTokenizer.load(forModelDirectory:)` (cache :418/:421 → factory, passo 4);
  `Model.load(directoryURL:device:streamingMode:expertCachePolicy:integrityPolicy:)`
  :438-443; `RealForwardRunner(model:context:maxContext:runtimeConfiguration:)`
  :447-451 → factory.make; scratch :452; `VisionRuntime.open(textModelURL:context:)`
  :459, missing → `visionRuntimeError` :470-474 (con Qwen: non aprire affatto, D12).
- Due creazioni `MultimodalConversation(model:context:tokenizer:runner:scratch:
  visionRuntime:visionRuntimeError:visionResidency:maxContext: runner.maxContext)`:
  `conversationForRestore` :364-384 e `conversationForTurn` :653-682; entrambe fanno
  `runner.reset()` prima (:370/:663).
- `runConversationTurn(request:prefillConfig:progress:memorySampler:report:)` :684-763:
  digest immagini :699-704, `parts.append(.image)` :705, `.text(request.prompt)` :708,
  `conversation.send(…)` :717-724; mapping errori :741-761. Guardia D12 immagini con
  Qwen: prima di :694 (o in conversationForTurn), prima di qualunque lavoro.
- Delta → eventi: closure `report` :803-827: `.prefill`→`.prefillProgress`,
  `.token`→`.token(AppTokenEvent(index:textDelta:elapsedDecodeSeconds:))`,
  `.tail`→`.token(…)`. `AppInferenceEvent` AppDiagnostics.swift:148; `AppTokenEvent` :142.
- `AppGenerationRequest` (AppGenerationRequest.swift:4): modelDirectory, prompt,
  imageAttachments, maxNewTokens, maxContextTokens, temperature, topK?, topP?,
  repetitionPenalty, runtimeOptions, continuesConversation, conversationTokens,
  conversationEpoch?, turnIndex? — il passo 5 aggiunge `enableThinking`.

## DecodeProtocol (410 righe) + DecodeService + client app
- `DecodeGenerationRequest` :65-122 (DecodeProtocol.swift): prompt, imageAttachments?,
  maxNewTokens, maxContextTokens, temperature, topK?, topP?, repetitionPenalty,
  runtimeOptions, generationID, conversationEpoch?, turnIndex?; init :98. Passo 5:
  `enableThinking: Bool? = nil` (Codable opzionale, assente = true).
- `DecodeServiceCommand` :193 (.generate :195); `DecodeServiceEventKind` :203-223
  (…, snapshot, …); `DecodeServiceEvent` :275: testo via `textDelta` :279 + `tokenCount`
  (eventi `.snapshot`); init :322 (default → campi nuovi additivi ok).
- Entry.swift (472 righe): client :55; `.generate` :247; `AppGenerationRequest(…)`
  :320-338; `for try await event in client.generate(generation) { outbox.publish(event) }` :339.
- DecodeServiceOutbox.swift: `publish(_ event: AppInferenceEvent)` :44 (`.token` accumula
  `pendingText`; terminali finished/cancelled/failed/lineageLost); `runWriter` :99.
- DecodeServiceInferenceClient.swift (app): `generate` :218; `DecodeGenerationRequest(`
  :229; snapshot → `.token` :290-309; terminali :310-323.

## App (Sources/TurboFieldfareApp/)
- Sampling UI: `Mac/Diagnostics/InspectorView.swift` `generationSection` :395+ (Slider
  temperature :398, Toggle Top-K :408, Top-P :420); gate immagini UI già in S24 (:17).
- `Core/State/AppModel.swift`: public var temperature/topKEnabled/topK/topPEnabled/topP
  :114-118; `AppGenerationRequest(` :3115; `apply(_ event:generation:)` :3146:
  `.prefillProgress` → `phase = .prefill` :3152, `.token` → `phase = .decode` :3157 e
  `outputText += token.textDelta` (stato "Thinking" nasce qui dal ragionamento);
  `phase: AppGenerationPhase` :156 (enum AppModelLoadState.swift:49); `presentation` :953;
  `isImageInputAvailable` :1411 (`modelVariant.supportsImages && …`, AppModelVariant.swift:89).
- HUD: `Mac/Diagnostics/StatusHUDView.swift` `PhaseLabel` :176-224; content :211-218
  (loading/pulse/steady/quiet da `model.presentation.label` + `model.phase`);
  `Core/State/AppPresentationState.swift` (Snapshot :21, `resolve` :103).
- `Core/Configuration/MacAppSettings.swift`: chiavi opzionali `*Qwen36` :42-52 +
  CodingKeys :76-81 — pattern per `thinkingEnabledQwen36` (assente = true).
  `applyModelVariant`/`savedSampling` per famiglia in AppModel (nota S24).
- Cronologia: `Core/History/ConversationContinuability.swift` `evaluate` :39, regola
  :47-49 `meta.identity.modelID != identity.modelID` → `.cannotReplay(.differentModel)`.
  Qwen modelID `mlx-community/Qwen3.6-35B-A3B-4bit` ≠ Gemma: la separazione Gemma↔Qwen
  esiste già (passo 7: verificare con test, non aggiungere).
- `Core/State/AppModelHistory.swift`: `conversationIdentityProvider` :52 (def
  AppModel.swift:249/:279); `continuability(of:)` :519; `recordedOtherVariant(of:)` :534;
  `openConversation(id:)` :568 (non continuabile = sola lettura); turno assistente
  :451-458 `ConversationTurnRecord(role: .assistant, at:, text: operation.assistantText,
  tokens: generatedTokenIDs, generatedTokens:, stopReason:, boundary:)` — qui passa il
  ragionamento nel campo `thinking`. `ConversationTurnRecord` in
  ConversationRecord.swift:183, `thinking: String?` :196 (mai scritta; init :222).

## API Qwen da usare
- `LanguageModelRunner` (Runtime/Inference/LanguageModelRunnerFactory.swift:9):
  `ContinuableLogitProducer` + `maxContext` + `rewind(to:) throws`; conformi :17/:18.
  Factory: `make(model:context:maxContext:runtimeConfiguration:)` :26 (Qwen senza
  runtimeConfiguration), `loadTokenizer(family:forModelDirectory:) -> any ChatTokenizer` :42;
  famiglia `Model.family(of: model.manifest)` (Model+Qwen.swift:19).
  `QwenRewindError.rewindUnsupported(position: Int)` QwenHybridForwardRunner.swift:6.
- `ChatTokenizer` (Tokenization/ChatTokenizer.swift:20): vocabSize, stopTokenIDs,
  endOfTurnID, structuralMarkerIDs, toolResultWaitID?, encode(_:addBOS:),
  decode(_:skipSpecialTokens:), makeStreamingDetokenizer(barrierTokenIDs:). NON ha
  applyChatTemplate/encodeTextContinuation → serve il renderer del passo 2.
- `QwenTokenizer` (Tokenization/Qwen/QwenTokenizer.swift): `applyChatTemplate(_:
  [GFTokenizer.Message], enableThinking: Bool = true) throws -> String` :337;
  `encodeTextContinuation(userContent: String, enableThinking: Bool = true)` :370
  (inizia con im_end + generation prompt :471); ChatTokenizer :52-63 (endOfTurnID =
  imEndID, toolResultWaitID = nil).
- `GFTokenizer` (Tokenization/Tokenizer.swift): applyChatTemplate :333,
  encodeTextContinuation :399, encodeMultimodalUserContinuation :417, Message :298/Role :273.
- `QwenAssistantDecoder` (Tokenization/Qwen/QwenAssistantDecoder.swift:16, public):
  `init(allowedTools: Set<String>, insideReasoning: Bool, idGenerator:)` :42,
  `consume(String) throws -> [StructuredAssistantEvent]` :53, `finish() throws ->
  [StructuredAssistantEvent]` :61, `onReasoning: (@Sendable (String) -> Void)?` :36,
  `hasToolCalls` :75. Ragionamento solo da `onReasoning` (D11); splitter internal
  QwenReasoningSplitter.swift:18 (`insideReasoning` = prompt finito con `<think>\n`).
- `ScriptedLogitProducer`: Sources/TurboFieldfareValidation/Support/Fixtures/
  (target TurboFieldfareValidationSupport).
- Test esistenti: Tests/TurboFieldfare/Core/Runtime/Generation/{MultimodalConversation
  KVRecovery,ConversationLineageRecord}Tests.swift (l'actor non è costruito direttamente
  da nessun test); …/Generation/Qwen/QwenChatTokenizerGenerationTests.swift (S21: producer
  scriptato con `prefillConfig: .off`); Tests/TurboFieldfareDecodeService/DecodeConversation
  {Gate,Restore}Tests.swift. `runRawCompletion` :273 (Gemma) → :296 (any ChatTokenizer).
  Note S21/S23: token di stop senza `.token`; boundary Qwen = [im_end]; a fine generazione
  pulita runner == KV → `.resume` legale, rewind mai in continuazione (D6).

## Decisioni (piano Parte 3.8)
- D6 (802-808): coda sempre ok; ≤2 snapshot a fine prefill; ogni altro rewind errore →
  prefill completo. D11 (823-828): `<think>` mai nel contenuto; app: "Thinking…" +
  interruttore (solo Qwen). D12 (829-832): immagini Qwen mai accettate; UI già gate da
  S24, runtime fail-closed. D13 (833-837): ChatTokenizer; funzioni esistenti stessa firma.

## Aggiornamenti passo 2 (38ec4db) — per il passo 3
- `MultimodalConversation.swift` dopo il passo 2 (righe nuove): proprietà `renderer` :287,
  `tokenizer: any ChatTokenizer & Sendable` :293, `runner: any LanguageModelRunner` :296;
  init Gemma (firma invariata) :329, init generale :356; guard D12 `generate` :546,
  `restore` :749; downcast GFTokenizer in `encodeTurn` :874.
- Passo 3: i call site di `rewind` sono ora `:642` (`restoreAfterFailure`, catch di
  `generate`, `lineageBroken = true` :645) e `:672` (`trimHiddenStopTokens`, stop-string,
  `lineageBroken = true` :675). Helper `MultimodalConversationKVRecovery` invariato :185–229
  (lancia `.lineageRecoveryFailed`). Fallback D6 = catch di `QwenRewindError.rewindUnsupported`
  (QwenHybridForwardRunner.swift:6): `reset` + rigioco di `kvTokenIDs` nudi con
  `runRawPrefill(start: .reset)` come in `restore` :818-841, non `lineageBroken`.
- Test: `ScriptedConversationRunner` (private, in
  `Tests/…/Generation/Qwen/MultimodalConversationQwenPromptTests.swift`) ha già cursore,
  `prepared`, `resetCount`, `rewind(to:)` = set del cursore: per il passo 3 aggiungere un flag
  `rewindThrows` che lancia `.rewindUnsupported` e riusare il modello del toy
  (`toyModel()` + `ModelLoaderTests.writeToySynthetic()`).

## Aggiornamenti passo 3 (rewind D6) — per il passo 4
- `MultimodalConversation.swift` ora 1003 righe. Nuove righe: helper
  `restoreAfterFailure` :185 e `trimHiddenStopTokens` :210 lanciano
  `QwenRewindError.rewindUnsupported` senza wrapper (:203, :230); call site
  `catch is QwenRewindError` :653 (catch di `generate`) e :698 (trim
  stop-string) → `replayRecordedKV` privato :750: `runRawPrefill` con
  `start: .reset` sui token registrati (niente `uncommittedBoundary`),
  `outputMode: .greedyIfAvailable`, `seedUse: .discarded`; replay fallito →
  reset + registro svuotato + `lineageBroken` + `lineageRecoveryFailed`.
  Spostamenti: actor :291, init Gemma :338, init generale :365, guard D12
  generate :555 e restore :837, `encodeTurn` :942.
- Test passo 3 nello stesso file del passo 2: `ScriptedConversationRunner` ha
  `setRewindThrows(_:)` e `failNextProduction()` (throws AFTER commit →
  `kvAdvanced` vera) + `ScriptedDecodeFailure` (internal, stesso file). Tre
  test nuovi (rigira, trim, propagazione helper). Per il passo 4: nessuna
  API pubblica nuova nel runtime; la factory per `RealInferenceClient` resta
  quella in mappa (`make` :26, `loadTokenizer` :42, `Model.family(of:)`).

## Aggiornamenti passo 4 (caricamento decode service) — per il passo 5
- `RealInferenceClient.swift` ora 1169 righe. `RealInferenceSession`:
  `tokenizer: (any ChatTokenizer & Sendable)?` :261, `promptRenderer:
  (any ConversationPromptRenderer)?` :264, `loadedFamily: ModelFamily?` :266,
  `runner: (any LanguageModelRunner)?` :270 (azzerati in `ensureLoaded` e
  `unload`). `ensureLoaded` :406: legge il manifest con
  `ManifestReader.load(directoryURL:expecting: .gemma4_26B_A4B)` :433 →
  `Self.family(of:)` (mirror di `Model.family`, è internal) → tokenizer via
  factory con cast family→concreto (no cast su marker Sendable: vietato dal
  compilatore) in `loadTokenizer(family:for:)` :538 →
  `LanguageModelRunnerFactory.make` :475 + `promptRenderer(family:tokenizer:)`
  (Qwen: `QwenConversationPromptRenderer` default `enableThinking: true` —
  il passo 5 deve far arrivare il campo fino a qui) + VisionRuntime aperto
  solo se `Self.opensImagePack(for:)` :488 (mai con Qwen, D12).
  `nonisolated static`: `family(of:)` :565, `opensImagePack` :572,
  `promptRenderer` :579 (testabili senza attore).
- Conversazioni create con l'init generale (`renderer:promptRenderer`):
  `conversationForRestore` :373, `conversationForTurn` :764 (guard += promptRenderer).
- Guardia D12 `request.imageAttachments.isEmpty || loadedFamily == .gemma4`
  prima di qualunque lavoro: `runConversationTurn` :806 e `runOneShot` :656.
- `runOneShot` :642: parametri `any LanguageModelRunner` +
  `any ChatTokenizer & Sendable` + `renderer`; ramo testo
  `renderer.firstTurnTokens(text:)` :665 (stesso template di prima), ramo
  immagini downcast GFTokenizer :698 (D12 lo rende irraggiungibile con Qwen).
  Diagnostics Gemma-only: snapshot contatori `(runner as?
  RealForwardRunner).map` :932, `runnerDiagnostics` :1082 downcast (Qwen: nil).
- Test nuovi `Tests/TurboFieldfareApp/Core/Inference/
  RealInferenceClientFamilySelectionTests.swift` (3, senza modello: manifest
  Qwen/Gemma scritti a mano → `family(of:)`, `opensImagePack`, e
  `promptRenderer` che rifiuta un tokenizer di altra famiglia con
  `MismatchedTokenizer` + `NullDetokenizer` file-locali; riutilizzabili).
- Passo 5: campo `enableThinking` in `DecodeGenerationRequest` (opzionale,
  assente = true) → `Entry.swift` :320 → `AppGenerationRequest`
  (AppGenerationRequest.swift:4) → `runConversationTurn`/renderer: il renderer
  è creato una volta per load in `ensureLoaded`; per il thinking per-richiesta
  servirà ricostruirlo (o ricrearlo alla creazione della conversazione) quando
  `enableThinking` cambia — il conversation actor lo riceve all'init.
  Pipeline D11: `QwenAssistantDecoder`/`onReasoning` in `MultimodalConversation`
  (i `.token`/`.tail` oggi vanno dritti in `text` in `generate`), delta →
  eventi app nella closure `report` :922-949 (`.token`→`AppTokenEvent`).
