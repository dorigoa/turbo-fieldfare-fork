import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// S23 step 2: the request surface branches on the loaded family.
/// `chat_template_kwargs` is a Qwen-only field (D11) and the sampling
/// defaults are the family's reference generation (D10, U4); under Gemma
/// both answers are the long-standing ones, byte for byte.
@Suite("Server Qwen request family")
struct ServerQwenRequestFamilyTests {

    private typealias Rejection = (message: String, param: String?, code: String)

    private static let body =
        #"{"model":"m","messages":[{"role":"user","content":"x"}]}"#

    private static func body(_ fields: String) -> String {
        #"{"model":"m","messages":[{"role":"user","content":"x"}],\#(fields)}"#
    }

    private static func decoder(family: ModelFamily) -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.userInfo[OpenAIChatRequest.modelFamilyUserInfoKey] = family
        return decoder
    }

    private static func decode(
        _ body: String,
        family: ModelFamily
    ) throws -> OpenAIChatRequest {
        try decoder(family: family).decode(
            OpenAIChatRequest.self, from: Data(body.utf8))
    }

    private func decodeRejection(
        _ body: String,
        family: ModelFamily
    ) -> Rejection? {
        do {
            _ = try Self.decode(body, family: family)
            Issue.record("request decoded instead of failing")
            return nil
        } catch ServerRequestError.invalid(let message, let param, let code) {
            return (message, param, code)
        } catch {
            Issue.record("decoding threw \(error) rather than a ServerRequestError")
            return nil
        }
    }

    private func validationRejection(
        _ body: String,
        decodeFamily: ModelFamily,
        validateFamily: ModelFamily
    ) -> Rejection? {
        do {
            _ = try OpenAIRequestValidator.validate(
                try Self.decode(body, family: decodeFamily),
                modelID: "m",
                family: validateFamily)
            Issue.record("request validated instead of failing")
            return nil
        } catch ServerRequestError.invalid(let message, let param, let code) {
            return (message, param, code)
        } catch {
            Issue.record("validation threw \(error) rather than a ServerRequestError")
            return nil
        }
    }

    // MARK: - chat_template_kwargs key sweep

    @Test func chatTemplateKwargsDecodesForQwenOnly() throws {
        let qwen = try Self.decode(
            Self.body(#""chat_template_kwargs":{"enable_thinking":false}"#),
            family: .qwen35Moe)
        #expect(qwen.chatTemplateKwargs != nil)
        // A Gemma-serving decoder answers exactly as it did before the key
        // was declared at all: one unknown field, same message, param, code.
        let gemma = decodeRejection(
            Self.body(#""chat_template_kwargs":{"enable_thinking":false}"#),
            family: .gemma4)
        #expect(gemma?.message == #"unrecognized request field "chat_template_kwargs""#)
        #expect(gemma?.param == "chat_template_kwargs")
        #expect(gemma?.code == "unknown_parameter")
    }

    @Test func absentDecoderFamilyMeansGemma() throws {
        // The production wiring sets the family on every request; a decoder
        // without it must not unlock the Qwen field.
        do {
            _ = try JSONDecoder().decode(
                OpenAIChatRequest.self,
                from: Data(Self.body(#""chat_template_kwargs":{}"#).utf8))
            Issue.record("chat_template_kwargs decoded without a family")
        } catch ServerRequestError.invalid(let message, let param, let code) {
            #expect(message.contains("chat_template_kwargs"))
            #expect(param == "chat_template_kwargs")
            #expect(code == "unknown_parameter")
        }
    }

    @Test func chatTemplateKwargsSortsWithOtherUnknownFieldsUnderGemma() {
        // Pre-Qwen, an undeclared key joined the sorted unknown list; the
        // family branch has to land in the same position.
        let rejection = decodeRejection(
            Self.body(#""zeta":1,"chat_template_kwargs":{}"#),
            family: .gemma4)
        #expect(rejection?.message
            == #"unrecognized request fields "chat_template_kwargs", "zeta""#)
        #expect(rejection?.param == "chat_template_kwargs")
    }

    // MARK: - enable_thinking validation

    @Test(arguments: [
        ( #"{"enable_thinking":false}"#, false),
        ( #"{"enable_thinking":true}"#, true),
        ("{}", true),
    ])
    func qwenEnableThinkingRoundTrips(kwargs: String, expected: Bool) throws {
        let validated = try OpenAIRequestValidator.validate(
            Self.decode(Self.body(#""chat_template_kwargs":\#(kwargs)"#),
                        family: .qwen35Moe),
            modelID: "m",
            family: .qwen35Moe)
        #expect(validated.enableThinking == expected)
    }

    @Test func qwenThinkingIsOnWithoutTheKwargsBlock() throws {
        // U3: the checkpoint's own default is thinking on, so a request that
        // says nothing gets the think frame.
        let validated = try OpenAIRequestValidator.validate(
            Self.decode(Self.body, family: .qwen35Moe),
            modelID: "m",
            family: .qwen35Moe)
        #expect(validated.enableThinking)
    }

    @Test func qwenRejectsForeignKwargsKeys() {
        let rejection = validationRejection(
            Self.body(#""chat_template_kwargs":{"temperature":0.5}"#),
            decodeFamily: .qwen35Moe,
            validateFamily: .qwen35Moe)
        #expect(rejection?.message.contains("temperature") == true)
        #expect(rejection?.param == "chat_template_kwargs")
        #expect(rejection?.code == "unsupported_value")
    }

    @Test func qwenRejectsNonBooleanEnableThinking() {
        let rejection = validationRejection(
            Self.body(#""chat_template_kwargs":{"enable_thinking":"yes"}"#),
            decodeFamily: .qwen35Moe,
            validateFamily: .qwen35Moe)
        #expect(rejection?.message.contains("boolean") == true)
        #expect(rejection?.param == "chat_template_kwargs")
        #expect(rejection?.code == "invalid_value")
    }

    @Test(arguments: ["[]", "7", "\"off\""])
    func qwenRejectsNonObjectKwargs(_ value: String) {
        let rejection = validationRejection(
            Self.body(#""chat_template_kwargs":\#(value)"#),
            decodeFamily: .qwen35Moe,
            validateFamily: .qwen35Moe)
        #expect(rejection?.message.contains("object") == true)
        #expect(rejection?.param == "chat_template_kwargs")
        #expect(rejection?.code == "invalid_value")
    }

    @Test func gemmaValidatorRefusesKwargsReachedBeyondTheSweep() {
        // The validator is reachable without the decode sweep (and the sweep
        // can only be reached through a Qwen decoder); the family check in
        // the validator repeats the sweep's refusal so no path serves a
        // Gemma model with a template escape hatch.
        let rejection = validationRejection(
            Self.body(#""chat_template_kwargs":{"enable_thinking":false}"#),
            decodeFamily: .qwen35Moe,
            validateFamily: .gemma4)
        #expect(rejection?.message
            == #"unrecognized request field "chat_template_kwargs""#)
        #expect(rejection?.param == "chat_template_kwargs")
        #expect(rejection?.code == "unknown_parameter")
    }

    // MARK: - family sampling defaults (D10)

    @Test func qwenAnswersWithTheCheckpointGenerationDefaults() throws {
        let validated = try OpenAIRequestValidator.validate(
            Self.decode(Self.body, family: .qwen35Moe),
            modelID: "m",
            family: .qwen35Moe)
        #expect(validated.generationConfig.temperature == 1.0)
        #expect(validated.generationConfig.topK == 20)
        #expect(validated.generationConfig.topP == 0.95)
    }

    @Test func gemmaSamplingDefaultsAreUntouched() throws {
        let validated = try OpenAIRequestValidator.validate(
            try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(Self.body.utf8)),
            modelID: "m",
            family: .gemma4)
        #expect(validated.generationConfig.temperature == 0.2)
        #expect(validated.generationConfig.topK == 64)
        #expect(validated.generationConfig.topP == 0.95)
    }

    @Test func explicitSamplingOverridesTheFamilyDefaults() throws {
        let validated = try OpenAIRequestValidator.validate(
            Self.decode(
                Self.body(#""temperature":0.3,"top_k":8,"top_p":0.5"#),
                family: .qwen35Moe),
            modelID: "m",
            family: .qwen35Moe)
        #expect(validated.generationConfig.temperature == 0.3)
        #expect(validated.generationConfig.topK == 8)
        #expect(validated.generationConfig.topP == 0.5)
    }

    // MARK: - reasoning_effort

    /// The three levels map to the shared budgets, so the server, the CLI and
    /// the app mean the same token counts by the same words.
    @Test(arguments: ReasoningDepth.allCases)
    func reasoningEffortMapsToTheSharedBudget(_ depth: ReasoningDepth) throws {
        let validated = try OpenAIRequestValidator.validate(
            Self.decode(Self.body(#""reasoning_effort":"\#(depth.rawValue)""#),
                        family: .qwen35Moe),
            modelID: "m",
            family: .qwen35Moe)
        #expect(validated.reasoningBudgetTokens == depth.budgetTokens)
        #expect(validated.enableThinking)
    }

    /// A request that asks for no depth is unbounded, which is the decode this
    /// server has always run.
    @Test func withoutReasoningEffortTheRequestIsUnbounded() throws {
        let validated = try OpenAIRequestValidator.validate(
            Self.decode(Self.body, family: .qwen35Moe),
            modelID: "m",
            family: .qwen35Moe)
        #expect(validated.reasoningBudgetTokens == nil)
    }

    /// Gemma has no reasoning phase, so its answer is the one it always gave:
    /// the key is refused as unsupported, not as a typo, and the message,
    /// param and code are unchanged.
    @Test func reasoningEffortStaysRefusedForGemma() {
        let rejection = decodeRejection(Self.body(#""reasoning_effort":"low""#),
                                        family: .gemma4)
        #expect(rejection?.message == "reasoning_effort is not supported")
        #expect(rejection?.param == "reasoning_effort")
        #expect(rejection?.code == "unsupported_value")
    }

    /// An OpenAI level this server has no budget for is refused rather than
    /// quietly treated as one of the three.
    @Test func reasoningEffortRefusesLevelsWithNoBudget() {
        for value in ["minimal", "none", "xhigh", "LOW", ""] {
            let rejection = validationRejection(
                Self.body(#""reasoning_effort":"\#(value)""#),
                decodeFamily: .qwen35Moe,
                validateFamily: .qwen35Moe)
            #expect(rejection?.param == "reasoning_effort")
            #expect(rejection?.code == "unsupported_value")
            #expect(rejection?.message.contains("low, medium, high") == true)
        }
    }

    /// A depth for a turn that does no reasoning bounds nothing: refused,
    /// rather than silently dropping one of the two controls.
    @Test func reasoningEffortAndThinkingOffAreMutuallyExclusive() {
        let rejection = validationRejection(
            Self.body(#""reasoning_effort":"low","chat_template_kwargs":{"enable_thinking":false}"#),
            decodeFamily: .qwen35Moe,
            validateFamily: .qwen35Moe)
        #expect(rejection?.param == "reasoning_effort")
        #expect(rejection?.code == "invalid_value")
    }

    /// Every other unsupported key stays unsupported for Qwen too: only
    /// `reasoning_effort` was taken back.
    @Test func otherUnsupportedKeysStayRefusedForQwen() {
        for key in ["logit_bias", "verbosity", "modalities", "top_logprobs"] {
            let rejection = decodeRejection(Self.body(#""\#(key)":{}"#),
                                            family: .qwen35Moe)
            #expect(rejection?.message == "\(key) is not supported")
            #expect(rejection?.code == "unsupported_value")
        }
    }
}
