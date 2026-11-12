// SPDX-License-Identifier: Apache-2.0

import XCTest
@testable import ummmlx

final class SSEParserTests: XCTestCase {
    func testSplitMultilineAndComments() throws {
        let input = ": keepalive\r\nevent: message\r\ndata: first\r\n" +
            "data:  second\r\n\r\ndata: [DONE]\r\n\r\n"
        var parser = SSEParser()
        var events: [String] = []
        for byte in Data(input.utf8) {
            if let event = try parser.feed(byte) { events.append(event) }
        }
        XCTAssertNil(try parser.finish())
        XCTAssertEqual(events, ["first\n second", "[DONE]"])
    }

    func testFinalEventWithoutBlankLine() throws {
        var parser = SSEParser()
        for byte in Data("data: final".utf8) {
            XCTAssertNil(try parser.feed(byte))
        }
        XCTAssertEqual(try parser.finish(), "final")
        XCTAssertNil(try parser.finish())
    }

    func testInvalidUTF8Fails() throws {
        var parser = SSEParser()
        XCTAssertNil(try parser.feed(0xFF))
        XCTAssertThrowsError(try parser.feed(0x0A))
    }

    func testLineLimitFails() throws {
        var parser = SSEParser()
        for _ in 0..<(4 * 1024 * 1024) {
            _ = try parser.feed(0x61)
        }
        XCTAssertThrowsError(try parser.feed(0x61))
    }

    func testLineLimitAllowsCRLF() throws {
        var parser = SSEParser()
        for _ in 0..<(4 * 1024 * 1024) {
            _ = try parser.feed(0x61)
        }
        XCTAssertNil(try parser.feed(0x0D))
        XCTAssertNil(try parser.feed(0x0A))
    }
}

private final class SSEProtocol: URLProtocol {
    static let responses = [
        "http://127.0.0.1/v1/chat/completions": Data(#"""
data: {"choices":[{"delta":{"content":""},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"length"}]}

data: [DONE]

"""#.utf8),
        "http://127.0.0.2/v1/chat/completions": Data(#"""
data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}

data: [DONE]

"""#.utf8)
    ]

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url, let data = Self.responses[url.absoluteString],
              let response = HTTPURLResponse(
                url: url, statusCode: 200, httpVersion: nil,
                headerFields: ["Content-Type": "text/event-stream"]
              ) else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

final class ChatClientTests: XCTestCase {
    @MainActor
    func testLoopbackChatURLs() throws {
        let cases = [
            "http://127.0.0.1:30000/v1": "http://127.0.0.1:30000/v1/chat/completions",
            "http://127.0.0.1:30000/v1/": "http://127.0.0.1:30000/v1/chat/completions",
            "http://localhost:30000": "http://localhost:30000/v1/chat/completions",
            "http://127.255.1.2/v1": "http://127.255.1.2/v1/chat/completions",
            "http://[::1]:30000/v1": "http://[::1]:30000/v1/chat/completions",
            "https://example.com/v1": "https://example.com/v1/chat/completions"
        ]
        for (base, expected) in cases {
            XCTAssertEqual(try ChatClient.chatURL(base: base).absoluteString, expected)
        }
    }

    @MainActor
    func testRejectsUnsafeOrUnsupportedURLs() {
        let cases = [
            "http://example.com/v1", "http://127.example.com/v1",
            "http://127.1/v1", "http://256.0.0.1/v1",
            "http://user@127.0.0.1/v1", "http://127.0.0.1/v1?x=1",
            "http://127.0.0.1/v1#part", "ftp://127.0.0.1/v1", ""
        ]
        for base in cases {
            XCTAssertThrowsError(try ChatClient.chatURL(base: base), base)
        }
    }

    @MainActor
    func testErrorDetail() {
        let object = Data(#"{"error":{"type":"invalid_request","message":"bad model"}}"#.utf8)
        XCTAssertEqual(ChatClient.errorDetail(object), "invalid_request: bad model")
        let string = Data(#"{"error":"failed"}"#.utf8)
        XCTAssertEqual(ChatClient.errorDetail(string), "failed")
        XCTAssertEqual(ChatClient.errorDetail(Data("plain".utf8)), "plain")
        XCTAssertEqual(ChatClient.errorDetail(Data()), "no error detail")
    }

    @MainActor
    private func makeClient() -> ChatClient {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [SSEProtocol.self]
        return ChatClient(session: URLSession(configuration: configuration))
    }

    @MainActor
    func testLengthAllowsEmptyCompletion() async throws {
        let result = try await makeClient().stream(
            base: "http://127.0.0.1", model: "test", messages: []
        ) { _ in }
        XCTAssertEqual(result.text, "")
        XCTAssertEqual(result.finishReason, "length")
    }

    @MainActor
    func testStopRejectsEmptyCompletion() async {
        do {
            _ = try await makeClient().stream(
                base: "http://127.0.0.2", model: "test", messages: []
            ) { _ in }
            XCTFail("An empty stop completion must fail")
        } catch {
            XCTAssertEqual(
                error as? ChatClientError,
                .invalidResponse("the model returned no text")
            )
        }
    }
}

final class ChatModelTests: XCTestCase {
    @MainActor
    func testHistoryContainsOnlyCompleteExchanges() {
        let turns = [
            ChatTurn(role: .user, text: "A", state: .complete),
            ChatTurn(role: .assistant, text: "a", state: .complete),
            ChatTurn(role: .user, text: "B", state: .complete),
            ChatTurn(role: .assistant, text: "partial", state: .failed),
            ChatTurn(role: .user, text: "C", state: .complete),
            ChatTurn(role: .assistant, text: "", state: .stopped),
            ChatTurn(role: .user, text: "D", state: .complete)
        ]
        XCTAssertEqual(
            ChatModel.requestMessages(turns: turns, prompt: "E"),
            [
                ChatMessage(role: "user", content: "A"),
                ChatMessage(role: "assistant", content: "a"),
                ChatMessage(role: "user", content: "E")
            ]
        )
    }
}
