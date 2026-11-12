// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import AppKit
import Darwin

@MainActor
final class Automation {
    private let model: ChatModel
    private let appearance: Binding<String>
    private let arguments = ProcessInfo.processInfo.arguments

    static func requested(model: ChatModel, appearance: Binding<String>) -> Automation? {
        guard ProcessInfo.processInfo.arguments.contains("--automation") else { return nil }
        return Automation(model: model, appearance: appearance)
    }

    private init(model: ChatModel, appearance: Binding<String>) {
        self.model = model
        self.appearance = appearance
        Task { await run() }
    }

    private func option(_ name: String) -> String? {
        let prefix = "--\(name)="
        return arguments.first { $0.hasPrefix(prefix) }.map { String($0.dropFirst(prefix.count)) }
    }

    private func run() async {
        let output = option("output") ?? "/tmp/ummmlx-result.json"
        let screenshots = option("screenshots") ?? "/tmp/ummmlx-screenshots"
        let prompt = option("prompt") ?? ""
        let followup = option("followup-prompt")
            ?? "Reply with one short sentence about the previous reply."
        let errorModel = option("error-model") ?? ""
        if let endpoint = option("endpoint") { model.endpoint = endpoint }
        if let servedModel = option("model") { model.model = servedModel }
        appearance.wrappedValue = option("theme") ?? "light"
        await settle()

        var stopChecked = false
        var stopStatus = "NOT_REQUESTED"
        var passed = true
        var failure = ""
        var responseText = ""
        var followupText = ""
        var errorChecked = false
        var errorStatus = "NOT_REQUESTED"
        var errorText = ""
        var requests: [[String]] = []
        var successTurns: [[String: String]] = []
        model.onRequest = { requests.append($0.map(\.role)) }

        if !prompt.isEmpty {
            var runCompletion = true
            if arguments.contains("--stop-first") {
                model.onFirstDelta = { [weak self] in
                    guard let self else { return }
                    stopChecked = true
                    self.model.onFirstDelta = nil
                    self.model.stop()
                }
                model.prompt = prompt
                model.send()
                if await waitForIdle() {
                    stopStatus = model.status
                    if !stopChecked || model.status != "Stopped" {
                        passed = false
                        failure = "The first real stream did not stop after a delta."
                    }
                } else {
                    passed = false
                    failure = "Timed out waiting for the stopped request."
                    runCompletion = false
                    model.stop()
                }
                model.newChat()
            }

            if runCompletion {
                model.prompt = prompt
                model.send()
                if await waitForIdle() {
                    responseText = model.turns.last { $0.role == .assistant }?.text ?? ""
                    if !model.status.hasPrefix("Completed") || responseText.isEmpty {
                        passed = false
                        failure = model.error ?? "The real request did not complete with text."
                    }
                } else {
                    passed = false
                    failure = "Timed out waiting for the real completion."
                    model.stop()
                }
            }

            // Second turn in the same session. It must carry the first
            // exchange; the runner checks the request roles.
            if runCompletion && passed {
                model.prompt = followup
                model.send()
                if await waitForIdle() {
                    followupText = model.turns.last { $0.role == .assistant }?.text ?? ""
                    if !model.status.hasPrefix("Completed") || followupText.isEmpty {
                        passed = false
                        failure = model.error
                            ?? "The second same-session request did not complete with text."
                    } else {
                        let roles = model.turns.map { $0.role.rawValue }
                        let states = model.turns.map { turn -> String in
                            switch turn.state {
                            case .complete: return "complete"
                            case .streaming: return "streaming"
                            case .stopped: return "stopped"
                            case .failed: return "failed"
                            }
                        }
                        if roles != ["user", "assistant", "user", "assistant"]
                            || states.contains(where: { $0 != "complete" }) {
                            passed = false
                            failure = "The same session did not keep two complete exchanges."
                        }
                    }
                } else {
                    passed = false
                    failure = "Timed out waiting for the second same-session completion."
                    model.stop()
                }
            }
            // Snapshot now: the rejected request below appends one user turn.
            successTurns = model.turns.map {
                ["role": $0.role.rawValue, "state": state($0.state)]
            }
        }

        var paths: [String] = []
        for theme in ["light", "dark"] {
            appearance.wrappedValue = theme
            await settle()
            do {
                paths.append(try capture(name: theme, directory: screenshots))
            } catch {
                passed = false
                failure = error.localizedDescription
            }
        }

        // Visualise a real server rejection: the model name is not served, so
        // the frozen server answers 400 and the client must show its message.
        if passed && !prompt.isEmpty && !errorModel.isEmpty {
            errorChecked = true
            appearance.wrappedValue = "light"
            await settle()
            let restore = model.model
            model.model = errorModel
            model.prompt = prompt
            model.send()
            if await waitForIdle() {
                errorStatus = model.status
                errorText = model.error ?? ""
                if errorStatus != "Error" || errorText.isEmpty {
                    passed = false
                    failure = "The real server error was not visible."
                }
            } else {
                passed = false
                failure = "Timed out waiting for the rejected request."
                model.stop()
            }
            model.model = restore
            await settle()
            do {
                paths.append(try capture(name: "error", directory: screenshots))
            } catch {
                passed = false
                failure = error.localizedDescription
            }
        }

        let result: [String: Any] = [
            "passed": passed,
            "static_only": prompt.isEmpty,
            "status": model.status,
            "error": model.error ?? "",
            "failure": failure,
            "response": responseText,
            "followup_response": followupText,
            "requests": requests,
            "turns": successTurns,
            "stop_checked": stopChecked,
            "stop_status": stopStatus,
            "error_checked": errorChecked,
            "error_status": errorStatus,
            "error_text": errorText,
            "endpoint": model.endpoint,
            "model": model.model,
            "screenshots": paths
        ]
        do {
            let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: output))
        } catch {
            FileHandle.standardError.write(Data("ummmlx automation: \(error)\n".utf8))
            exit(2)
        }
        exit(passed ? 0 : 1)
    }

    private func state(_ value: ChatTurn.State) -> String {
        switch value {
        case .complete: return "complete"
        case .streaming: return "streaming"
        case .stopped: return "stopped"
        case .failed: return "failed"
        }
    }

    private func settle() async {
        for _ in 0..<20 {
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
    }

    private func waitForIdle(timeout: TimeInterval = 120) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        for _ in 0..<2400 {
            if !model.isRunning { return true }
            if Date() >= deadline { return false }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        return !model.isRunning
    }

    private func capture(name: String, directory: String) throws -> String {
        guard let window = NSApp.windows.first(where: { $0.isVisible && $0.contentView != nil }),
              let view = window.contentView else {
            throw NSError(domain: "ummmlx", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "No visible app window."])
        }
        window.layoutIfNeeded()
        view.layoutSubtreeIfNeeded()
        guard let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else {
            throw NSError(domain: "ummmlx", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "Cannot allocate window bitmap."])
        }
        view.cacheDisplay(in: view.bounds, to: rep)
        guard let data = rep.representation(using: .png, properties: [:]) else {
            throw NSError(domain: "ummmlx", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "Cannot encode window PNG."])
        }
        let manager = FileManager.default
        try manager.createDirectory(atPath: directory, withIntermediateDirectories: true)
        let path = (directory as NSString).appendingPathComponent("ummmlx-\(name).png")
        try data.write(to: URL(fileURLWithPath: path))
        return path
    }
}
