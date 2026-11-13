import Foundation
import TurboFieldfare
import TurboFieldfareAppCore
import TurboFieldfareDecodeProtocol

final class DecodeServiceOutbox: @unchecked Sendable {
    private struct PrefillProgress {
        var done: Int
        var total: Int
    }

    private struct State {
        var pendingText = ""
        var latestPrefill: PrefillProgress?
        var latestToken: AppTokenEvent?
        var terminal: DecodeServiceEvent?
        var terminalCommitted = false
        var finished = false
        var sequence: UInt64 = 0
        /// Thinking-block transitions waiting for the next drain (S25 step 5,
        /// D11). Coalesced rather than dropped: a true/false pair inside one
        /// 100 ms window still has to end with the block closed, and one
        /// bool cannot say which.
        var thinkingQueue: [Bool] = []
    }

    private let condition = NSCondition()
    private var state = State()
    private let generationID: UUID
    private let memorySampler = AppMemorySampler()

    /// Bytes of image tower currently held mapped, or nil when there is no
    /// vision runtime. Sampled per event so Keep Ready is visible while a run
    /// is happening, not only in its final diagnostics.
    private let towerBytes: @Sendable () -> UInt64?
    /// Nil outside conversation mode, so a one-shot turn reports no gauge
    /// rather than a misleading zero.
    private let conversationTokens: @Sendable () -> Int?

    init(generationID: UUID,
         towerBytes: @escaping @Sendable () -> UInt64? = { nil },
         conversationTokens: @escaping @Sendable () -> Int? = { nil }) {
        self.conversationTokens = conversationTokens
        self.generationID = generationID
        self.towerBytes = towerBytes
        memorySampler.resetPeak()
    }

    func publish(_ event: AppInferenceEvent) {
        condition.lock()
        switch event {
        case .memorySample:
            // The writer samples on its own schedule; an inbound reading is the
            // runtime's, and there is nothing to queue.
            break
        case .prefillProgress(let done, let total):
            state.latestPrefill = PrefillProgress(done: done, total: total)
            condition.signal()
        case .token(let token):
            state.pendingText += token.textDelta
            state.latestToken = token
        case .thinking(let active):
            state.thinkingQueue.append(active)
            condition.signal()
        case .finished(let diagnostics):
            if !state.terminalCommitted {
                state.terminal = terminal(.finished, diagnostics: diagnostics)
                state.terminalCommitted = true
            }
        case .cancelled(let diagnostics):
            if !state.terminalCommitted {
                state.terminal = terminal(.cancelled, diagnostics: diagnostics)
                state.terminalCommitted = true
            }
        case .failed(let error, let diagnostics):
            if !state.terminalCommitted {
                // A broken lineage is not the same terminal state as a failed
                // turn. The turn can be retried; the conversation cannot, and
                // the app has to clear it rather than offer a retry that would
                // fail identically forever.
                let kind: DecodeServiceEventKind
                if case .conversationLineageLost = error { kind = .lineageLost }
                else { kind = .failed }
                state.terminal = terminal(
                    kind, diagnostics: diagnostics, error: error.userMessage)
                state.terminalCommitted = true
            }
        }
        if state.terminal != nil { condition.signal() }
        condition.unlock()
    }

    func finish(error: Error? = nil) {
        condition.lock()
        if !state.terminalCommitted, let error {
            state.terminal = DecodeServiceEvent(
                kind: .failed, generationID: generationID, error: "\(error)")
            state.terminalCommitted = true
        }
        state.finished = true
        condition.broadcast()
        condition.unlock()
    }

    func runWriter(to handle: FileHandle) throws {
        while true {
            condition.lock()
            if state.terminal == nil, !state.finished {
                _ = condition.wait(until: Date().addingTimeInterval(0.1))
            }
            let prefill = state.latestPrefill
            let text = state.pendingText
            let token = state.latestToken
            let terminal = state.terminal
            let done = state.finished
            let thinking = state.thinkingQueue
            state.latestPrefill = nil
            state.pendingText = ""
            state.latestToken = nil
            state.terminal = nil
            state.thinkingQueue = []
            var prefillSequence: UInt64?
            if prefill != nil {
                state.sequence &+= 1
                prefillSequence = state.sequence
            }
            var tokenSequence: UInt64?
            if !text.isEmpty || token != nil {
                state.sequence &+= 1
                tokenSequence = state.sequence
            }
            condition.unlock()

            _ = memorySampler.sample()

            if prefill == nil, text.isEmpty, token == nil, terminal == nil,
               thinking.isEmpty, !done {
                let snapshot = DecodeServiceEvent(
                    kind: .memory, generationID: generationID,
                    currentMemoryBytes: memorySampler.sample(),
                    peakMemoryBytes: memorySampler.peakBytes,
                    visionTowerMappedBytes: towerBytes())
                try handle.write(contentsOf: DecodeFrameCodec.encode(snapshot))
                continue
            }
            if let prefill, let prefillSequence {
                let snapshot = DecodeServiceEvent(
                    kind: .prefill, generationID: generationID,
                    sequence: prefillSequence,
                    prefillDone: prefill.done, prefillTotal: prefill.total,
                    currentMemoryBytes: memorySampler.sample(),
                    peakMemoryBytes: memorySampler.peakBytes,
                    visionTowerMappedBytes: towerBytes())
                try handle.write(contentsOf: DecodeFrameCodec.encode(snapshot))
            }
            if !text.isEmpty || token != nil {
                let elapsed = token?.elapsedDecodeSeconds ?? 0
                let count = (token?.index ?? -1) + 1
                let snapshot = DecodeServiceEvent(
                    kind: .snapshot, generationID: generationID,
                    sequence: tokenSequence ?? 0, textDelta: text, tokenCount: count,
                    decodeSeconds: elapsed,
                    tokensPerSecond: elapsed > 0 ? Double(count) / elapsed : 0,
                    currentMemoryBytes: memorySampler.sample(),
                    peakMemoryBytes: memorySampler.peakBytes,
                    visionTowerMappedBytes: towerBytes())
                try handle.write(contentsOf: DecodeFrameCodec.encode(snapshot))
            }
            // Thinking transitions ride ahead of the terminal event so an
            // open block can never outlive its run on the client: the
            // conversation closes it during the turn's unwind, before the
            // terminal is published, and the queue preserves that order.
            for active in thinking {
                let event = DecodeServiceEvent(
                    kind: .thinking, generationID: generationID,
                    currentMemoryBytes: memorySampler.sample(),
                    peakMemoryBytes: memorySampler.peakBytes,
                    visionTowerMappedBytes: towerBytes(),
                    thinkingActive: active)
                try handle.write(contentsOf: DecodeFrameCodec.encode(event))
            }
            if let terminal {
                try handle.write(contentsOf: DecodeFrameCodec.encode(terminal))
            }
            if terminal != nil { return }
            if done { return }
        }
    }

    private func terminal(_ kind: DecodeServiceEventKind,
                          diagnostics: AppDiagnostics?,
                          error: String? = nil) -> DecodeServiceEvent {
        DecodeServiceEvent(
            kind: kind, generationID: generationID,
            tokenCount: diagnostics?.generatedTokens ?? 0,
            promptTokenCount: diagnostics?.promptTokenCount,
            computedPrefillTokens: diagnostics?.computedPrefillTokens,
            prefillSeconds: diagnostics?.prefillSeconds,
            timeToFirstTokenSeconds: diagnostics?.timeToFirstTokenSeconds,
            decodeSeconds: diagnostics?.decodeSeconds ?? 0,
            tokensPerSecond: diagnostics?.tokensPerSecond ?? 0,
            stopReason: diagnostics?.stopReason.rawValue,
            error: error,
            currentMemoryBytes: memorySampler.sample(),
            peakMemoryBytes: memorySampler.peakBytes,
            visionTowerMappedBytes: diagnostics?.visionTowerMappedBytes,
            cachedPromptTokens: diagnostics?.cachedPromptTokens,
            conversationTokenCount: conversationTokens(),
            // The turn's own record, carried once on the event that ends it.
            // The snapshots this outbox sends per 100 ms carry text, which is
            // what the transcript shows; these are what the model saw, and the
            // app cannot store the conversation without them.
            promptTokenIDs: diagnostics?.promptTokenIDs,
            generatedTokenIDs: diagnostics?.generatedTokenIDs,
            boundaryTokenIDs: diagnostics?.boundaryTokenIDs,
            boundaryNeedsReplay: diagnostics?.boundaryNeedsReplay,
            prefill: diagnostics?.prefill.map(Self.prefillDiagnostics),
            runner: diagnostics?.runner.map(Self.runnerDiagnostics))
    }

    private static func prefillDiagnostics(_ value: PrefillExecutionDiagnostics)
        -> DecodePrefillDiagnostics {
        DecodePrefillDiagnostics(
            requestedMode: value.requestedMode.rawValue,
            executedMode: value.executedMode.rawValue,
            kvStorageMode: value.kvStorageMode?.rawValue,
            chunkCompleteness: value.chunkCompleteness.rawValue,
            unsupportedReason: value.unsupportedReason)
    }

    private static func runnerDiagnostics(_ value: AppRunnerDiagnostics)
        -> DecodeRunnerDiagnostics {
        DecodeRunnerDiagnostics(
            cb1MillisecondsPerToken: value.cb1MillisecondsPerToken,
            ioMillisecondsPerToken: value.ioMillisecondsPerToken,
            cb2MillisecondsPerToken: value.cb2MillisecondsPerToken,
            headMillisecondsPerToken: value.headMillisecondsPerToken,
            rdadviseMillisecondsPerToken: value.rdadviseMillisecondsPerToken,
            rdadviseCallsPerToken: value.rdadviseCallsPerToken,
            rdadviseMegabytesPerToken: value.rdadviseMegabytesPerToken,
            rdadviseSkippedPerToken: value.rdadviseSkippedPerToken,
            rdadviseFailures: value.rdadviseFailures)
    }
}
