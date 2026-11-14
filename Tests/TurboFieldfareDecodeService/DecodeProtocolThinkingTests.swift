import Foundation
import Testing
import TurboFieldfareDecodeProtocol

/// S25 step 5: the thinking switch crosses the wire. The request's field is
/// optional because absent means enabled, and a thinking transition is its
/// own event kind whose field stays invisible until a family sets it.

@Suite struct DecodeProtocolThinkingTests {
    @Test("enableThinking round-trips when set and is absent from the wire when nil")
    func enableThinkingRoundTrip() throws {
        for enabled in [true, false] {
            let request = DecodeGenerationRequest(prompt: "hi", maxNewTokens: 8,
                                                  maxContextTokens: 512,
                                                  temperature: 0.2,
                                                  enableThinking: enabled)
            let decoded = try JSONDecoder().decode(
                DecodeGenerationRequest.self,
                from: try JSONEncoder().encode(request))
            #expect(decoded.enableThinking == enabled)
        }
        let untouched = DecodeGenerationRequest(prompt: "hi", maxNewTokens: 8,
                                                maxContextTokens: 512,
                                                temperature: 0.2)
        let encoded = try JSONEncoder().encode(untouched)
        // Absent must stay absent — not `"enableThinking": null`: the wire
        // message for a peer that predates the switch has to be byte-identical
        // to the message that peer sends, and "absent" is what the service
        // reads as "enabled".
        #expect(!String(decoding: encoded, as: UTF8.self).contains("enableThinking"))
        #expect(try JSONDecoder().decode(DecodeGenerationRequest.self,
                                         from: encoded).enableThinking == nil)
    }

    @Test("the reasoning budget round-trips when set and is absent from the wire when nil")
    func reasoningBudgetRoundTrip() throws {
        for tokens in [256, 1_024, 4_096] {
            let request = DecodeGenerationRequest(prompt: "hi", maxNewTokens: 8,
                                                  maxContextTokens: 512,
                                                  temperature: 0.2,
                                                  reasoningBudgetTokens: tokens)
            let decoded = try JSONDecoder().decode(
                DecodeGenerationRequest.self,
                from: try JSONEncoder().encode(request))
            #expect(decoded.reasoningBudgetTokens == tokens)
        }
        let untouched = DecodeGenerationRequest(prompt: "hi", maxNewTokens: 8,
                                                maxContextTokens: 512,
                                                temperature: 0.2)
        let encoded = try JSONEncoder().encode(untouched)
        // Reasoning depth (D11): absent, not null, for the same reason as the switch above — a
        // peer that predates the cap sends exactly this message, and the
        // service reads absence as unbounded reasoning.
        #expect(!String(decoding: encoded, as: UTF8.self).contains("reasoningBudgetTokens"))
        #expect(try JSONDecoder().decode(DecodeGenerationRequest.self,
                                         from: encoded).reasoningBudgetTokens == nil)
    }

    /// The two fields are independent on the wire: a message may carry either,
    /// both or neither, and a reader of one must not need the other.
    @Test("the switch and the budget cross the wire independently")
    func thinkingAndBudgetAreIndependent() throws {
        let request = DecodeGenerationRequest(prompt: "hi", maxNewTokens: 8,
                                              maxContextTokens: 512,
                                              temperature: 0.2,
                                              enableThinking: true,
                                              reasoningBudgetTokens: 256)
        let decoded = try JSONDecoder().decode(
            DecodeGenerationRequest.self, from: try JSONEncoder().encode(request))
        #expect(decoded.enableThinking == true)
        #expect(decoded.reasoningBudgetTokens == 256)

        let switchOnly = DecodeGenerationRequest(prompt: "hi", maxNewTokens: 8,
                                                 maxContextTokens: 512,
                                                 temperature: 0.2,
                                                 enableThinking: false)
        let switchJSON = try JSONEncoder().encode(switchOnly)
        #expect(!String(decoding: switchJSON, as: UTF8.self)
            .contains("reasoningBudgetTokens"))
    }

    @Test("a thinking transition is its own kind and other events omit the field")
    func thinkingEventWire() throws {
        let on = DecodeServiceEvent(kind: .thinking, generationID: UUID(),
                                    thinkingActive: true)
        let back = try JSONDecoder().decode(
            DecodeServiceEvent.self, from: try JSONEncoder().encode(on))
        #expect(back.kind == .thinking)
        #expect(back.thinkingActive == true)

        let plain = DecodeServiceEvent(kind: .snapshot, generationID: UUID(),
                                       textDelta: "hi", tokenCount: 1)
        let plainJSON = try JSONEncoder().encode(plain)
        #expect(!String(decoding: plainJSON, as: UTF8.self).contains("thinkingActive"))
        let plainBack = try JSONDecoder().decode(DecodeServiceEvent.self, from: plainJSON)
        // A missing field is no state change, for either generation of peer.
        #expect(plainBack.thinkingActive == nil)
    }
}
