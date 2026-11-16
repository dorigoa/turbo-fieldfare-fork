import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// Claude Code 2.1 sends `role: "system"` turns inside the conversation: a
/// per-turn effort statement with empty content, an instruction that holds
/// until the next user message, and tools added or removed. Before this, the
/// first of them at `messages[1]` failed the whole request.
@Suite("Anthropic system turns")
struct AnthropicSystemTurnTests {
    private static func validated(_ body: String) throws -> ValidatedChatRequest {
        let json = try AnthropicMessages.chatRequestJSON(from: Data(body.utf8), family: .qwen35Moe)
        let decoded = try OpenAIChatRequest.decodeRequest(from: json, family: .qwen35Moe,
                                                          acceptAnyModel: true)
        return try OpenAIRequestValidator.validate(decoded, modelID: "m", family: .qwen35Moe,
                                                   acceptAnyModel: true)
    }

    private static let readTool = #"""
    {"name":"Read","input_schema":{"type":"object","properties":{"file_path":{"type":"string"}}}}
    """#

    @Test func anEffortStatementIsDropped() throws {
        let request = try Self.validated(#"""
        {"max_tokens":10,"messages":[
          {"role":"user","content":"hi"},
          {"role":"system","content":[],"output_config":{"effort":"medium"}}]}
        """#)
        #expect(request.messages.map(\.role) == [.user])
        #expect(request.messages[0].content == "hi")
    }

    @Test func anInstructionBecomesAReminderInTheUserTurn() throws {
        let request = try Self.validated(#"""
        {"max_tokens":10,"system":"Base.","messages":[
          {"role":"user","content":"hi"},
          {"role":"system","content":"Plan mode is on.","clear_at":"next_user_message"}]}
        """#)
        #expect(request.messages.map(\.role) == [.system, .user])
        #expect(request.messages[1].content
            == "hi\n\n<system-reminder>\nPlan mode is on.\n</system-reminder>")
    }

    /// After tool results there is no user text to join, so the reminder is
    /// its own user turn, after the results it follows.
    @Test func anInstructionAfterToolResultsFollowsThem() throws {
        let request = try Self.validated(#"""
        {"max_tokens":10,"tools":[\#(Self.readTool)],"messages":[
          {"role":"user","content":"read it"},
          {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"/a"}}]},
          {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"x"}]},
          {"role":"system","content":[{"type":"text","text":"Be brief.","cache_control":{"type":"ephemeral"}}]}]}
        """#)
        #expect(request.messages.map(\.role) == [.user, .assistant, .tool, .user])
        #expect(request.messages[3].content == "<system-reminder>\nBe brief.\n</system-reminder>")
    }

    @Test func aSystemTurnAheadOfTheConversationJoinsTheSystemPrompt() throws {
        let request = try Self.validated(#"""
        {"max_tokens":10,"system":"Base.","messages":[
          {"role":"system","content":"Extra."},
          {"role":"user","content":"hi"}]}
        """#)
        #expect(request.messages.map(\.role) == [.system, .user])
        #expect(request.messages[0].content == "Base.\n\nExtra.")
    }

    @Test func toolsAreAddedRemovedAndLoadedOnReference() throws {
        let request = try Self.validated(#"""
        {"max_tokens":10,"tools":[
           \#(Self.readTool),
           {"name":"Edit","input_schema":{"type":"object","properties":{}}},
           {"name":"Deferred","defer_loading":true,"input_schema":{"type":"object","properties":{}}},
           {"name":"Searched","defer_loading":true,"input_schema":{"type":"object","properties":{}}},
           {"name":"NeverLoaded","defer_loading":true,"input_schema":{"type":"object","properties":{}}}],
         "messages":[
          {"role":"user","content":"go"},
          {"role":"system","content":[
             {"type":"tool_addition","tool":{"type":"tool_definition","definition":
                {"name":"Late","description":"d","input_schema":{"type":"object","properties":{"x":{"type":"string"}}}}}},
             {"type":"tool_addition","tool":{"type":"tool_reference","name":"Deferred"}},
             {"type":"tool_removal","tool":{"type":"tool_reference","name":"Edit"}}]},
          {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"/a"}}]},
          {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1",
             "content":[{"type":"tool_reference","tool_name":"Searched"}]}]}]}
        """#)
        #expect(request.tools.map(\.name) == ["Read", "Deferred", "Searched", "Late"])
        #expect(request.messages.last?.content == "[Searched is now available]")
    }

    @Test func anUnknownRoleIsNamed() {
        do {
            _ = try AnthropicMessages.chatRequestJSON(
                from: Data(#"{"max_tokens":1,"messages":[{"role":"developer","content":"x"}]}"#.utf8),
                family: .gemma4)
            Issue.record("expected a refusal")
        } catch let error as ServerRequestError {
            #expect(error.envelope.error.message
                == #""messages[0].role" is "developer"; it must be user, assistant, or system"#)
            #expect(error.envelope.error.param == "messages[0].role")
        } catch {
            Issue.record("unexpected error \(error)")
        }
    }
}
