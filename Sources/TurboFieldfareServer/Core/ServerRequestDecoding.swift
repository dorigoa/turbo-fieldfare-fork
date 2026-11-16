import Foundation
import TurboFieldfare

extension OpenAIChatRequest {
    /// Decodes a chat request body, turning a decoding failure into an error
    /// that names what is wrong.
    ///
    /// Every such failure used to answer "malformed JSON request", whether the
    /// body was not JSON at all or was valid JSON missing `model`, so a client
    /// could not tell which field to fix and the server log had nothing either.
    /// The code stays `invalid_json` for all of them, as before, so a client
    /// that branches on it is unaffected; the message and `param` now name the
    /// field by its JSON path and say what it should be. They never quote a
    /// value from the request, so they are safe to log.
    static func decodeRequest(from body: Data,
                              family: ModelFamily,
                              acceptAnyModel: Bool = false) throws -> OpenAIChatRequest {
        guard body.contains(where: { !Self.isJSONWhitespace($0) }) else {
            throw ServerRequestError.invalid(message: "request body is empty",
                                             param: nil, code: "invalid_json")
        }
        let decoder = JSONDecoder()
        // The key sweep in the request decoder asks the family whether
        // `chat_template_kwargs` is a field at all (D11).
        decoder.userInfo[modelFamilyUserInfoKey] = family
        decoder.userInfo[acceptAnyModelUserInfoKey] = acceptAnyModel
        do {
            return try decoder.decode(OpenAIChatRequest.self, from: body)
        } catch let error as DecodingError {
            throw ServerRequestError.decoding(error, body: body)
        }
    }

    /// The top-level integer fields, for naming the one Foundation refused
    /// as out of range: it reports a number that does not fit its type as
    /// invalid JSON with no field at all. A field missing here still gets an
    /// accurate message, only without its name.
    static let integerFields: [(name: String, unsigned: Bool)] = [
        ("seed", true),
        ("max_tokens", false),
        ("max_completion_tokens", false),
        ("top_k", false),
        ("n", false),
    ]

    private static func isJSONWhitespace(_ byte: UInt8) -> Bool {
        byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D
    }
}

extension ServerRequestError {
    static func decoding(_ error: DecodingError, body: Data) -> ServerRequestError {
        switch error {
        case .keyNotFound(let key, let context):
            let path = jsonPath(context.codingPath + [key])
            return invalidJSON("missing required field \"\(path)\"", param: path)
        case .valueNotFound(let type, let context):
            let path = jsonPath(context.codingPath)
            return invalidJSON("\"\(path)\" must be \(expected(type)), not null", param: path)
        case .typeMismatch(let type, let context):
            let path = jsonPath(context.codingPath)
            return invalidJSON("\"\(path)\" must be \(expected(type))", param: path)
        case .dataCorrupted(let context) where !context.codingPath.isEmpty:
            let path = jsonPath(context.codingPath)
            return invalidJSON("\"\(path)\" has a value out of range or of the wrong form",
                               param: path)
        case .dataCorrupted(let context):
            let descriptions = underlyingDescriptions(context)
            if descriptions.contains(where: { $0.contains("is not representable") }) {
                if let field = outOfRangeField(in: body) {
                    let expectation = field.unsigned ? "a non-negative integer" : "an integer"
                    return invalidJSON("\"\(field.name)\" must be \(expectation)",
                                       param: field.name)
                }
                return invalidJSON("a number in the request does not fit its field", param: nil)
            }
            let location = syntaxLocation(descriptions).map { " (\($0))" } ?? ""
            return invalidJSON("request body is not valid JSON\(location)", param: nil)
        @unknown default:
            return invalidJSON("malformed JSON request", param: nil)
        }
    }

    private static func invalidJSON(_ message: String, param: String?) -> ServerRequestError {
        .invalid(message: message, param: param, code: "invalid_json")
    }

    /// `messages[0].content`: keys joined by dots, array positions in brackets.
    static func jsonPath(_ codingPath: [any CodingKey]) -> String {
        var path = ""
        for key in codingPath {
            if let index = key.intValue {
                path += "[\(index)]"
            } else {
                path += path.isEmpty ? key.stringValue : ".\(key.stringValue)"
            }
        }
        return path.isEmpty ? "body" : path
    }

    /// What a field of this Swift type looks like in JSON.
    static func expected(_ type: Any.Type) -> String {
        switch type {
        case is String.Type: return "a string"
        case is Bool.Type: return "true or false"
        case is UInt.Type, is UInt8.Type, is UInt16.Type, is UInt32.Type, is UInt64.Type:
            return "a non-negative integer"
        case is Int.Type, is Int8.Type, is Int16.Type, is Int32.Type, is Int64.Type:
            return "an integer"
        case is Float.Type, is Double.Type: return "a number"
        default:
            let name = String(describing: type)
            if name.hasPrefix("Dictionary<") || (name.hasPrefix("[") && name.contains(":")) {
                return "an object"
            }
            if name.hasPrefix("Array<") || name.hasPrefix("[") {
                return "an array"
            }
            return "a value of another type"
        }
    }

    private static func underlyingDescriptions(_ context: DecodingError.Context) -> [String] {
        var texts = [context.debugDescription]
        if let underlying = context.underlyingError {
            texts.append(String(describing: underlying))
            if let detail = (underlying as NSError).userInfo[NSDebugDescriptionErrorKey] as? String {
                texts.append(detail)
            }
        }
        return texts
    }

    /// The line and column of a syntax error, and nothing else from its
    /// description, which quotes the offending text.
    private static func syntaxLocation(_ descriptions: [String]) -> String? {
        for text in descriptions {
            if let match = text.firstMatch(of: /line (\d+), column (\d+)/) {
                return "line \(match.1), column \(match.2)"
            }
        }
        return nil
    }

    /// The first top-level integer field whose number is fractional, negative
    /// where it must not be, or too large for its type.
    private static func outOfRangeField(in body: Data) -> (name: String, unsigned: Bool)? {
        guard let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any] else {
            return nil
        }
        for field in OpenAIChatRequest.integerFields {
            guard let number = object[field.name] as? NSNumber else { continue }
            let value = number.doubleValue
            let lowest = field.unsigned ? 0 : Double(Int.min)
            let highest = field.unsigned ? Double(UInt64.max) : Double(Int.max)
            if value != value.rounded() || value < lowest || value > highest {
                return field
            }
        }
        return nil
    }
}
