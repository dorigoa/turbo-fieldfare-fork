import Foundation
import Testing
@testable import TurboFieldfare

/// The XML tool-call block of plan 3.6 must lift JSON-typed parameters and
/// keep everything else as a string, after removing exactly one leading and
/// one trailing newline.
@Suite
struct QwenToolCallParserTests {
    private func block(_ inner: String) -> String {
        "\(QwenToolCallParser.callOpen)\n\(inner)\n\(QwenToolCallParser.callClose)"
    }

    private func parse(_ text: String,
                       allowed: Set<String> = ["search"]) throws -> ParsedToolCall {
        try QwenToolCallParser().parse(text, allowedTools: allowed, id: "call_test")
    }

    @Test("a full block parses name, JSON-typed and string parameters")
    func fullBlock() throws {
        let call = try parse(block("""
        <function=search>
        <parameter=query>
        Rome weather
        </parameter>
        <parameter=days>
        3
        </parameter>
        <parameter=flag>
        true
        </parameter>
        <parameter=spot>
        {"city": "Rome", "lat": 41.9}
        </parameter>
        <parameter=ids>
        [1, 2]
        </parameter>
        </function>
        """))
        #expect(call.id == "call_test")
        #expect(call.name == "search")
        let arguments = try #require(call.arguments.objectValue)
        #expect(arguments["query"] == .string("Rome weather"))
        #expect(arguments["days"] == .integer(3))
        #expect(arguments["flag"] == .bool(true))
        #expect(arguments["ids"] == .array([.integer(1), .integer(2)]))
        let spot = try #require(arguments["spot"]?.objectValue)
        #expect(spot["city"] == .string("Rome"))
        #expect(arguments.keys.sorted() == ["days", "flag", "ids", "query", "spot"])
        // argumentsJSON round-trips the lifted values with sorted keys.
        #expect(call.argumentsJSON.hasPrefix(#"{"days":3,"flag":true,"ids":[1,2],"query":"Rome weather""#))
    }

    @Test("only one leading and one trailing newline are removed")
    func singleNewlineTrim() throws {
        let call = try parse("<function=search><parameter=q>\nA\nB\n</parameter></function>")
        #expect(call.arguments.objectValue?["q"] == .string("A\nB"))
        let doubled = try parse("<function=search><parameter=q>\n\nX\n\n</parameter></function>")
        #expect(doubled.arguments.objectValue?["q"] == .string("\nX\n"))
        // No newlines at all: taken verbatim.
        let inline = try parse("<function=search><parameter=q>hi</parameter></function>")
        #expect(inline.arguments.objectValue?["q"] == .string("hi"))
    }

    @Test("text that only looks like JSON stays a string")
    func stringFallback() throws {
        let call = try parse(block("""
        <function=search>
        <parameter=a>
        12abc
        </parameter>
        <parameter=b>
        {broken
        </parameter>
        <parameter=c>

        </parameter>
        </function>
        """))
        let arguments = try #require(call.arguments.objectValue)
        #expect(arguments["a"] == .string("12abc"))
        #expect(arguments["b"] == .string("{broken"))
        #expect(arguments["c"] == .string(""))
    }

    @Test("no parameters means empty arguments")
    func noParameters() throws {
        let call = try parse("<function=search></function>")
        #expect(call.arguments == .object([:]))
        #expect(call.argumentsJSON == "{}")
    }

    @Test("malformed blocks throw a clear error")
    func malformed() {
        let cases: [(String, QwenToolCallParserError)] = [
            // Missing closing tags.
            ("<function=search>", .malformed),
            ("<function=search><parameter=q>hi", .malformed),
            ("<function=search>\n</function>\nstray", .malformed),
            // Duplicated parameter.
            ("<function=search><parameter=q>a</parameter><parameter=q>b</parameter></function>",
             .malformed),
            // Empty or invalid name.
            ("<function=>\n</function>", .malformed),
            ("<function=na me>\n</function>", .malformed),
            // Not a function block at all.
            ("hello", .malformed),
            ("", .malformed),
        ]
        for (text, error) in cases {
            #expect(throws: error, "case \(text.debugDescription)") {
                try parse(text)
            }
        }
    }

    @Test("unknown tool names are rejected before arguments")
    func unknownTool() {
        #expect(throws: QwenToolCallParserError.unknownTool("delete")) {
            try parse("<function=delete></function>")
        }
        // Allowed list opens it up.
        #expect(throws: Never.self) {
            _ = try parse("<function=delete></function>", allowed: ["delete"])
        }
    }
}
