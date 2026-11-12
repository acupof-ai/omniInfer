// SPDX-License-Identifier: Apache-2.0

import Foundation

struct ChatTurn: Identifiable {
    enum Role: String {
        case user
        case assistant
    }
    enum State {
        case complete
        case streaming
        case stopped
        case failed
    }

    let id = UUID()
    let role: Role
    var text: String
    var state: State
}

@MainActor
final class ChatModel: ObservableObject {
    @Published var endpoint: String
    @Published var model: String
    @Published var prompt = ""
    @Published private(set) var turns: [ChatTurn] = []
    @Published private(set) var isRunning = false
    @Published private(set) var status = "Ready"
    @Published private(set) var error: String?

    private let client = ChatClient()
    private var requestTask: Task<Void, Never>?
    private var activeTurn: UUID?
    private var stopping = false
    private var receivedDelta = false
    var onFirstDelta: (() -> Void)?
    /// Automation hook: the exact messages sent for one request.
    var onRequest: (([ChatMessage]) -> Void)?

    init() {
        let defaults = UserDefaults.standard
        endpoint = defaults.string(forKey: "endpoint") ?? "http://127.0.0.1:30000/v1"
        model = defaults.string(forKey: "model") ?? "Qwen2.5-0.5B-Instruct-4bit"
    }

    func send() {
        guard !isRunning else { return }
        error = nil
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            error = "Enter a prompt."
            status = "Input required"
            return
        }
        do {
            _ = try ChatClient.chatURL(base: endpoint)
            if model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                throw ChatClientError.modelRequired
            }
        } catch {
            self.error = error.localizedDescription
            status = "Configuration error"
            return
        }

        if !ProcessInfo.processInfo.arguments.contains("--automation") {
            UserDefaults.standard.set(endpoint, forKey: "endpoint")
            UserDefaults.standard.set(model, forKey: "model")
        }
        let history = Self.requestMessages(turns: turns, prompt: text)
        onRequest?(history)

        turns.append(ChatTurn(role: .user, text: text, state: .complete))
        let assistant = ChatTurn(role: .assistant, text: "", state: .streaming)
        turns.append(assistant)
        activeTurn = assistant.id
        prompt = ""
        isRunning = true
        stopping = false
        receivedDelta = false
        status = "Connecting…"

        requestTask = Task { [weak self] in
            guard let self else { return }
            do {
                let result = try await self.client.stream(
                    base: self.endpoint, model: self.model, messages: history
                ) { delta in
                    guard self.activeTurn == assistant.id,
                          let index = self.turns.firstIndex(where: { $0.id == assistant.id }) else {
                        return
                    }
                    self.turns[index].text += delta
                    if !self.receivedDelta {
                        self.receivedDelta = true
                        self.status = "Streaming…"
                        self.onFirstDelta?()
                    }
                }
                guard self.activeTurn == assistant.id,
                      let index = self.turns.firstIndex(where: { $0.id == assistant.id }) else {
                    return
                }
                self.turns[index].text = result.text
                self.turns[index].state = .complete
                self.status = result.finishReason.isEmpty
                    ? "Completed" : "Completed · \(result.finishReason)"
                self.finish(assistant.id)
            } catch {
                guard self.activeTurn == assistant.id else { return }
                let cancelled = self.stopping || Task.isCancelled ||
                    (error as? URLError)?.code == .cancelled || error is CancellationError
                if let index = self.turns.firstIndex(where: { $0.id == assistant.id }) {
                    if self.turns[index].text.isEmpty {
                        self.turns.remove(at: index)
                    } else {
                        self.turns[index].state = cancelled ? .stopped : .failed
                    }
                }
                if cancelled {
                    self.status = "Stopped"
                } else {
                    self.status = "Error"
                    self.error = error.localizedDescription
                }
                self.finish(assistant.id)
            }
        }
    }

    static func requestMessages(turns: [ChatTurn], prompt: String) -> [ChatMessage] {
        var messages: [ChatMessage] = []
        var pendingUser: ChatTurn?
        for turn in turns {
            switch turn.role {
            case .user:
                pendingUser = turn
            case .assistant:
                if let pendingUser, turn.state == .complete {
                    messages.append(ChatMessage(role: "user", content: pendingUser.text))
                    messages.append(ChatMessage(role: "assistant", content: turn.text))
                }
                pendingUser = nil
            }
        }
        messages.append(ChatMessage(role: "user", content: prompt))
        return messages
    }

    func stop() {
        guard isRunning else { return }
        stopping = true
        status = "Stopping…"
        requestTask?.cancel()
    }

    func newChat() {
        requestTask?.cancel()
        requestTask = nil
        activeTurn = nil
        turns.removeAll()
        isRunning = false
        stopping = false
        receivedDelta = false
        error = nil
        status = "Ready"
        onFirstDelta = nil
    }

    private func finish(_ id: UUID) {
        guard activeTurn == id else { return }
        requestTask = nil
        activeTurn = nil
        isRunning = false
        stopping = false
    }
}
