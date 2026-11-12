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
        if let endpoint = option("endpoint") { model.endpoint = endpoint }
        if let servedModel = option("model") { model.model = servedModel }
        appearance.wrappedValue = option("theme") ?? "light"
        await settle()

        var stopChecked = false
        var stopStatus = "NOT_REQUESTED"
        var passed = true
        var failure = ""
        var responseText = ""
        let deadline = Date().addingTimeInterval(240)

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
                if await waitForIdle(deadline: deadline) {
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
                if await waitForIdle(deadline: deadline) {
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
        }

        var paths: [String] = []
        for theme in ["light", "dark"] {
            appearance.wrappedValue = theme
            await settle()
            do {
                paths.append(try capture(theme: theme, directory: screenshots))
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
            "stop_checked": stopChecked,
            "stop_status": stopStatus,
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

    private func settle() async {
        for _ in 0..<20 {
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
    }

    private func waitForIdle(deadline: Date) async -> Bool {
        for _ in 0..<2400 {
            if !model.isRunning { return true }
            if Date() >= deadline { return false }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        return !model.isRunning
    }

    private func capture(theme: String, directory: String) throws -> String {
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
        let path = (directory as NSString).appendingPathComponent("ummmlx-\(theme).png")
        try data.write(to: URL(fileURLWithPath: path))
        return path
    }
}
