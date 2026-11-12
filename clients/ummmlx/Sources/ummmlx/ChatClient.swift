// SPDX-License-Identifier: Apache-2.0
// Modified from Voxt internal/llm/openai_http.go and openai.go at commit
// 5645daf0480e9bf572b45a2bba8b8e2f1e642daa. See ThirdParty/Voxt/LICENSE
// and NOTICE. Changes: Swift concurrency, incremental parsing, strict limits,
// required [DONE], and the ummInfer request contract.

import Foundation

struct ChatMessage: Codable, Equatable {
    let role: String
    let content: String
}

struct ChatCompletion {
    let text: String
    let finishReason: String
}

enum ChatClientError: LocalizedError, Equatable {
    case invalidEndpoint(String)
    case modelRequired
    case http(Int, String)
    case invalidResponse(String)
    case streamTruncated

    var errorDescription: String? {
        switch self {
        case .invalidEndpoint(let detail): return detail
        case .modelRequired: return "Model is required."
        case .http(let status, let detail): return "HTTP \(status): \(detail)"
        case .invalidResponse(let detail): return "Invalid server response: \(detail)"
        case .streamTruncated: return "The stream ended before [DONE]."
        }
    }
}

struct SSEParser {
    private var line = Data()
    private var dataLines: [String] = []
    private var dataBytes = 0
    private let limit = 4 * 1024 * 1024

    mutating func feed(_ byte: UInt8) throws -> String? {
        if byte == 0x0A {
            return try consumeLine()
        }
        line.append(byte)
        if line.count > limit + (line.last == 0x0D ? 1 : 0) {
            throw ChatClientError.invalidResponse("an SSE line exceeds 4 MiB")
        }
        return nil
    }

    mutating func finish() throws -> String? {
        if !line.isEmpty {
            if let event = try consumeLine() { return event }
        }
        return try flush()
    }

    private mutating func consumeLine() throws -> String? {
        defer { line.removeAll(keepingCapacity: true) }
        if line.last == 0x0D { line.removeLast() }
        if line.isEmpty { return try flush() }
        guard let text = String(data: line, encoding: .utf8) else {
            throw ChatClientError.invalidResponse("an SSE line is not UTF-8")
        }
        let parts = text.split(separator: ":", maxSplits: 1,
                               omittingEmptySubsequences: false)
        guard parts[0] == "data" else { return nil }
        var value = parts.count == 2 ? String(parts[1]) : ""
        if value.first == " " { value.removeFirst() }
        dataBytes += value.utf8.count + (dataLines.isEmpty ? 0 : 1)
        if dataBytes > limit {
            throw ChatClientError.invalidResponse("an SSE event exceeds 4 MiB")
        }
        dataLines.append(value)
        return nil
    }

    private mutating func flush() throws -> String? {
        guard !dataLines.isEmpty else {
            dataBytes = 0
            return nil
        }
        let event = dataLines.joined(separator: "\n")
        dataLines.removeAll(keepingCapacity: true)
        dataBytes = 0
        return event
    }
}

private final class NoRedirectDelegate: NSObject, URLSessionTaskDelegate,
                                        @unchecked Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}

@MainActor
final class ChatClient {
    private let session: URLSession
    private let responseLimit = 64 * 1024 * 1024

    init(session: URLSession? = nil) {
        if let session {
            self.session = session
            return
        }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 180
        configuration.timeoutIntervalForResource = 300
        configuration.connectionProxyDictionary = [:]
        self.session = URLSession(configuration: configuration,
                                  delegate: NoRedirectDelegate(), delegateQueue: nil)
    }

    static func chatURL(base: String) throws -> URL {
        let text = base.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let components = URLComponents(string: text),
              let scheme = components.scheme?.lowercased(),
              let host = components.host?.lowercased(), !host.isEmpty,
              scheme == "http" || scheme == "https",
              components.user == nil, components.password == nil,
              components.query == nil, components.fragment == nil else {
            throw ChatClientError.invalidEndpoint(
                "Use an HTTP(S) base URL without credentials, query, or fragment.")
        }
        if scheme == "http" && !isLoopback(host) {
            throw ChatClientError.invalidEndpoint(
                "Plain HTTP is allowed only on loopback (127.0.0.0/8, ::1, or localhost).")
        }
        var mutable = components
        var path = mutable.path
        while path.count > 1 && path.last == "/" { path.removeLast() }
        if path.isEmpty || path == "/" { path = "/v1" }
        mutable.path = path + "/chat/completions"
        guard let url = mutable.url else {
            throw ChatClientError.invalidEndpoint("The base URL is invalid.")
        }
        return url
    }

    private static func isLoopback(_ host: String) -> Bool {
        if host == "localhost" || host == "::1" || host == "[::1]" { return true }
        let parts = host.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 4, parts[0] == "127" else { return false }
        return parts.allSatisfy { part in
            guard !part.isEmpty, part.allSatisfy(\.isASCII),
                  part.allSatisfy(\.isNumber), let value = Int(part) else { return false }
            return value >= 0 && value <= 255
        }
    }

    func stream(base: String, model: String, messages: [ChatMessage],
                onDelta: (String) -> Void) async throws -> ChatCompletion {
        let name = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { throw ChatClientError.modelRequired }
        let url = try Self.chatURL(base: base)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": name,
            "messages": messages.map { ["role": $0.role, "content": $0.content] },
            "max_tokens": 64,
            "temperature": 0,
            "stream": true
        ])

        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ChatClientError.invalidResponse("missing HTTP response")
        }
        guard (200..<300).contains(http.statusCode) else {
            let body = try await readLimited(bytes, limit: 64 * 1024)
            throw ChatClientError.http(http.statusCode, Self.errorDetail(body))
        }
        let contentType = http.value(forHTTPHeaderField: "Content-Type")?
            .split(separator: ";").first?.trimmingCharacters(in: .whitespaces)
            .lowercased()
        guard contentType == "text/event-stream" else {
            throw ChatClientError.invalidResponse("expected text/event-stream")
        }

        var parser = SSEParser()
        var text = ""
        var finishReason = ""
        var received = 0
        var done = false

        func consume(_ event: String) throws -> Bool {
            if event == "[DONE]" { return true }
            let object = try JSONSerialization.jsonObject(with: Data(event.utf8))
            guard let root = object as? [String: Any] else {
                throw ChatClientError.invalidResponse("an SSE event is not an object")
            }
            if let error = root["error"], !(error is NSNull) {
                throw ChatClientError.http(500, Self.errorDetail(Data(event.utf8)))
            }
            if let rawChoices = root["choices"] {
                guard let choices = rawChoices as? [Any] else {
                    throw ChatClientError.invalidResponse("choices is not an array")
                }
                for rawChoice in choices {
                    guard let choice = rawChoice as? [String: Any] else {
                        throw ChatClientError.invalidResponse("a choice is not an object")
                    }
                    if let rawDelta = choice["delta"], !(rawDelta is NSNull) {
                        guard let delta = rawDelta as? [String: Any] else {
                            throw ChatClientError.invalidResponse("delta is not an object")
                        }
                        if let rawContent = delta["content"], !(rawContent is NSNull) {
                            guard let content = rawContent as? String else {
                                throw ChatClientError.invalidResponse("delta.content is not a string")
                            }
                            if !content.isEmpty {
                                text += content
                                if text.utf8.count > responseLimit {
                                    throw ChatClientError.invalidResponse("response text exceeds 64 MiB")
                                }
                                onDelta(content)
                            }
                        }
                    }
                    if let rawFinish = choice["finish_reason"], !(rawFinish is NSNull) {
                        guard let finish = rawFinish as? String else {
                            throw ChatClientError.invalidResponse("finish_reason is not a string")
                        }
                        if !finish.isEmpty { finishReason = finish }
                    }
                }
            }
            return false
        }

        stream: for try await byte in bytes {
            try Task.checkCancellation()
            received += 1
            if received > responseLimit {
                throw ChatClientError.invalidResponse("stream exceeds 64 MiB")
            }
            if let event = try parser.feed(byte), try consume(event) {
                done = true
                break stream
            }
        }
        if !done, let event = try parser.finish() {
            done = try consume(event)
        }
        try Task.checkCancellation()
        guard done else { throw ChatClientError.streamTruncated }
        guard !text.isEmpty || finishReason == "length" else {
            throw ChatClientError.invalidResponse("the model returned no text")
        }
        return ChatCompletion(text: text, finishReason: finishReason)
    }

    private func readLimited(_ bytes: URLSession.AsyncBytes, limit: Int) async throws -> Data {
        var data = Data()
        for try await byte in bytes {
            data.append(byte)
            if data.count >= limit { break }
        }
        return data
    }

    static func errorDetail(_ data: Data) -> String {
        var detail = ""
        if let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let error = root["error"] as? String {
                detail = error
            } else if let error = root["error"] as? [String: Any] {
                detail = [error["type"], error["code"], error["message"]]
                    .compactMap { $0 as? String }.filter { !$0.isEmpty }
                    .joined(separator: ": ")
            }
            if detail.isEmpty { detail = root["message"] as? String ?? "" }
            if detail.isEmpty, root["detail"] != nil {
                detail = String(decoding: data, as: UTF8.self)
            }
        } else {
            detail = String(decoding: data, as: UTF8.self)
        }
        detail = detail.trimmingCharacters(in: .whitespacesAndNewlines)
        if detail.isEmpty { detail = "no error detail" }
        return String(detail.prefix(500))
    }
}
