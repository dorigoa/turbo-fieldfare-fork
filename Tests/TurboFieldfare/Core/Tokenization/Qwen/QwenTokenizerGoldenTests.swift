import Foundation
import Testing
@testable import TurboFieldfare

/// S2 golden tokenizer cases (`Tests/QwenFixtures/tokenizer/`
/// `tokenizer_golden.json`): text→ids and ids→decoded with
/// `skip_special_tokens = false`.
///
/// 100% agreement is the S9 exit criterion, so the expectations here are the
/// fixtures themselves, never values transcribed into the test body.
@Suite(.enabled(if: QwenTestResources.hasTokenizer))
struct QwenTokenizerGoldenTests {
    struct GoldenCase: Decodable {
        let text: String
        let ids: [Int]
        let decoded: String
    }

    let tokenizer: QwenTokenizer
    let cases: [GoldenCase]

    init() async throws {
        tokenizer = try await QwenTokenizer.load(
            from: try #require(QwenTestResources.tokenizerFolder))
        let data = try Data(contentsOf: QwenTestResources.fixtures
            .appendingPathComponent("tokenizer/tokenizer_golden.json"))
        cases = try JSONDecoder().decode([GoldenCase].self, from: data)
        #expect(cases.count == 83)
    }

    @Test("golden text encodes to the golden IDs")
    func encode() {
        for (index, testCase) in cases.enumerated() {
            #expect(tokenizer.encode(testCase.text) == testCase.ids.map(Int32.init),
                    "case \(index): \(testCase.text.prefix(40))")
        }
    }

    @Test("addBOS is ignored: Qwen has no BOS")
    func addBOSIgnored() {
        for testCase in cases {
            #expect(tokenizer.encode(testCase.text, addBOS: false)
                    == tokenizer.encode(testCase.text, addBOS: true))
        }
    }

    @Test("golden IDs decode to the golden text")
    func decode() {
        for (index, testCase) in cases.enumerated() {
            #expect(tokenizer.decode(testCase.ids.map(Int32.init), skipSpecialTokens: false)
                    == testCase.decoded, "case \(index): \(testCase.text.prefix(40))")
        }
    }

    @Test("token-by-token streaming decode equals the batch decode")
    func streamingMatchesBatch() {
        for (index, testCase) in cases.enumerated() {
            let ids = testCase.ids.map(Int32.init)
            for skip in [false, true] {
                var stream = QwenDetokenizer(tokenizer: tokenizer, skipSpecialTokens: skip)
                var streaming = ""
                for id in ids { streaming += stream.push(id) }
                streaming += stream.flush()
                #expect(streaming == tokenizer.decode(ids, skipSpecialTokens: skip),
                        "case \(index), skipSpecialTokens=\(skip)")
            }
        }
    }
}
