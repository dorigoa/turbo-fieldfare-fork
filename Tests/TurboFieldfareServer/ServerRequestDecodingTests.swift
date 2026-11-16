import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// A body that does not decode used to answer "malformed JSON request" for
/// every cause. Each case here answered exactly that before the fix.
@Suite("Server request decoding")
struct ServerRequestDecodingTests {
    private static func decodingError(_ body: String,
                                      family: ModelFamily = .qwen35Moe) -> ServerRequestError? {
        do {
            _ = try OpenAIChatRequest.decodeRequest(from: Data(body.utf8), family: family)
            return nil
        } catch let error as ServerRequestError {
            return error
        } catch {
            Issue.record("unexpected error type: \(error)")
            return nil
        }
    }

    private static func detail(_ body: String) throws -> OpenAIErrorEnvelope.Detail {
        try #require(decodingError(body)).envelope.error
    }

    /// The request from the bug report: valid JSON, no `model`.
    @Test func missingModelIsNamed() throws {
        let detail = try Self.detail(#"""
        {"messages":[{"role":"user","content":"Spiegami la relatività ristretta in 600 parole"}],"max_tokens":1000}
        """#)
        #expect(detail.message == #"missing required field "model""#)
        #expect(detail.param == "model")
        // The code every decoding failure has always carried, so a client
        // that branches on it is unaffected.
        #expect(detail.code == "invalid_json")
        #expect(detail.type == "invalid_request_error")
    }

    @Test func nestedMissingFieldNamesItsPath() throws {
        let detail = try Self.detail(#"{"model":"m","messages":[{"content":"hi"}]}"#)
        #expect(detail.message == #"missing required field "messages[0].role""#)
        #expect(detail.param == "messages[0].role")
    }

    @Test(arguments: [
        (#"{"model":"m","messages":[],"temperature":"0.7"}"#, "temperature", "a number"),
        (#"{"model":"m","messages":[],"stream":"yes"}"#, "stream", "true or false"),
        (#"{"model":"m","messages":{}}"#, "messages", "an array"),
        (#"{"model":7,"messages":[]}"#, "model", "a string"),
        (#"{"model":"m","messages":[],"max_tokens":"1000"}"#, "max_tokens", "an integer"),
    ])
    func wrongTypeNamesFieldAndExpectedType(body: String, field: String, expected: String) throws {
        let detail = try Self.detail(body)
        #expect(detail.message == "\"\(field)\" must be \(expected)")
        #expect(detail.param == field)
        #expect(detail.code == "invalid_json")
    }

    /// Foundation reports a number that does not fit its type as invalid
    /// JSON with no field; some clients send `"seed": -1` for "random".
    @Test(arguments: [
        (#"{"model":"m","messages":[],"seed":-1}"#, "seed", "a non-negative integer"),
        (#"{"model":"m","messages":[],"max_tokens":1.5}"#, "max_tokens", "an integer"),
        (#"{"model":"m","messages":[],"top_k":1e40}"#, "top_k", "an integer"),
    ])
    func outOfRangeNumberIsNamed(body: String, field: String, expected: String) throws {
        let detail = try Self.detail(body)
        #expect(detail.message == "\"\(field)\" must be \(expected)")
        #expect(detail.param == field)
        #expect(detail.code == "invalid_json")
    }

    @Test(arguments: ["", "   \n\t"])
    func emptyBodyIsCalledEmpty(body: String) throws {
        let detail = try Self.detail(body)
        #expect(detail.message == "request body is empty")
        #expect(detail.code == "invalid_json")
    }

    /// What a shell apostrophe inside `-d '...'` leaves behind. The message
    /// says where the JSON breaks and never quotes the body.
    @Test func brokenJSONSaysWhereNotWhat() throws {
        let detail = try Self.detail("{'model':'m','messages':[]}")
        #expect(detail.message == "request body is not valid JSON (line 1, column 3)")
        #expect(!detail.message.contains("'"))
        #expect(!detail.message.contains("model"))
        #expect(detail.code == "invalid_json")
        #expect(detail.param == nil)
    }

    /// The key sweep's own refusal still reaches the client unchanged.
    @Test func unknownKeyKeepsTheSweepError() throws {
        let error = try #require(Self.decodingError(
            #"{"model":"m","messages":[],"chat_template_kwargs":{}}"#, family: .gemma4))
        #expect(error.envelope.error.code != "invalid_json")
        #expect(error.envelope.error.message != "malformed JSON request")
    }

    @Test func validRequestStillDecodes() throws {
        let request = try OpenAIChatRequest.decodeRequest(
            from: Data(#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#.utf8),
            family: .gemma4)
        #expect(request.model == "m")
    }

    @Test func rejectedLogLineNamesTheRefusalWithoutContent() {
        let detail = ServerRequestError.invalid(
            message: #"missing required field "model""#,
            param: "model", code: "invalid_json").envelope.error
        #expect(ServerLog.rejectedMessage(status: 400, method: "POST",
                                          path: "/v1/chat/completions", error: detail)
            == "request rejected status=400 method=POST path=/v1/chat/completions "
            + #"code=invalid_json param=model reason=missing required field "model""#)
        let routeDetail = OpenAIErrorEnvelope(message: "route not found", code: "not_found").error
        #expect(ServerLog.rejectedMessage(status: 404, method: "GET", path: "/v2/x",
                                          error: routeDetail)
            == "request rejected status=404 method=GET path=/v2/x code=not_found "
            + "reason=route not found")
    }

    @Test func logOutputDefaultsToStdout() throws {
        let parse = { (extra: [String]) in
            try ServerArguments.parse(["--model", "model.gturbo"] + extra,
                                      hostMemoryBytes: 64 << 30, environment: [:],
                                      readModelVariant: { _ in nil })
        }
        #expect(try parse([]).logOutput == .stdout)
        #expect(try parse(["--log-output", "stderr"]).logOutput == .stderr)
        #expect(throws: ServerArgumentError.invalid("--log-output must be stdout or stderr")) {
            _ = try parse(["--log-output", "file"])
        }
        #expect(ServerArguments.usage.contains("--log-output <stdout|stderr>"))
    }
}
