// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import AppKit

@main
struct UmmMLXApp: App {
    @StateObject private var model = ChatModel()
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        WindowGroup("ummmlx") {
            RootView(model: model)
        }
        .defaultSize(width: 900, height: 680)
        .windowResizability(.contentMinSize)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        !ProcessInfo.processInfo.arguments.contains("--automation")
    }
}

struct RootView: View {
    @ObservedObject var model: ChatModel
    @AppStorage("appearance") private var appearance = "system"
    @State private var automatedAppearance: String?
    @State private var automation: Automation?
    @Environment(\.colorScheme) private var colorScheme
    private let isAutomation = ProcessInfo.processInfo.arguments.contains("--automation")

    private var theme: Theme { Theme(dark: colorScheme == .dark) }
    private var preferredScheme: ColorScheme? {
        switch automatedAppearance ?? appearance {
        case "light": return .light
        case "dark": return .dark
        default: return nil
        }
    }
    private var appearanceBinding: Binding<String> {
        guard isAutomation else { return $appearance }
        return Binding(
            get: { automatedAppearance ?? appearance },
            set: { automatedAppearance = $0 }
        )
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            settings
            Divider().overlay(theme.line)
            transcript
            if let error = model.error { errorBanner(error) }
            Divider().overlay(theme.line)
            composer
        }
        .frame(minWidth: 660, minHeight: 500)
        .background(theme.background)
        .foregroundStyle(theme.text)
        .tint(theme.accent)
        .preferredColorScheme(preferredScheme)
        .onAppear {
            if automation == nil {
                automation = Automation.requested(model: model, appearance: appearanceBinding)
            }
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            Text("ummmlx")
                .font(.system(size: theme.titleSize, weight: .semibold))
            Circle().fill(model.isRunning ? theme.accent : theme.tertiary)
                .frame(width: 7, height: 7)
                .accessibilityHidden(true)
            Text(model.status)
                .font(.system(size: theme.captionSize))
                .foregroundStyle(theme.secondary)
                .accessibilityIdentifier("status")
            Spacer()
            Picker("Theme", selection: appearanceBinding) {
                Text("System").tag("system")
                Text("Light").tag("light")
                Text("Dark").tag("dark")
            }
            .pickerStyle(.menu)
            .fixedSize()
            Button("New chat") { model.newChat() }
                .accessibilityIdentifier("new-chat")
        }
        .padding(16)
        .background(theme.surface)
    }

    private var settings: some View {
        HStack(alignment: .bottom, spacing: 12) {
            VStack(alignment: .leading, spacing: 5) {
                Text("Base URL")
                    .font(.system(size: theme.captionSize))
                    .foregroundStyle(theme.secondary)
                TextField("http://127.0.0.1:30000/v1", text: $model.endpoint)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                    .accessibilityLabel("Base URL")
                    .accessibilityIdentifier("endpoint")
            }
            .frame(maxWidth: .infinity)
            VStack(alignment: .leading, spacing: 5) {
                Text("Model")
                    .font(.system(size: theme.captionSize))
                    .foregroundStyle(theme.secondary)
                TextField("Model served by ummInfer", text: $model.model)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                    .accessibilityLabel("Model")
                    .accessibilityIdentifier("model")
            }
            .frame(width: 285)
        }
        .disabled(model.isRunning)
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(theme.surface)
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                if model.turns.isEmpty {
                    VStack(spacing: 10) {
                        Image(systemName: "text.bubble")
                            .font(.system(size: 30, weight: .light))
                            .foregroundStyle(theme.accent)
                        Text("Chat with your local ummInfer MLX stage")
                            .font(.system(size: theme.sectionSize, weight: .medium))
                        Text("Text only · loopback HTTP · streaming SSE")
                            .font(.system(size: theme.bodySize))
                            .foregroundStyle(theme.secondary)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.top, 110)
                } else {
                    LazyVStack(spacing: 16) {
                        ForEach(model.turns) { turn in
                            messageRow(turn).id(turn.id)
                        }
                    }
                    .padding(20)
                }
            }
            .onChange(of: model.turns.count) { _, _ in scrollToEnd(proxy) }
            .onChange(of: model.turns.last?.text) { _, _ in scrollToEnd(proxy) }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func messageRow(_ turn: ChatTurn) -> some View {
        HStack(alignment: .top) {
            if turn.role == .user { Spacer(minLength: 90) }
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 7) {
                    Text(turn.role == .user ? "You" : "ummInfer")
                        .font(.system(size: theme.captionSize, weight: .medium))
                        .foregroundStyle(theme.secondary)
                    if turn.state == .streaming && turn.text.isEmpty {
                        ProgressView().controlSize(.small)
                    }
                    if turn.state == .stopped {
                        Text("Stopped").font(.system(size: theme.captionSize))
                            .foregroundStyle(theme.secondary)
                    }
                    if turn.state == .failed {
                        Text("Incomplete").font(.system(size: theme.captionSize))
                            .foregroundStyle(theme.danger)
                    }
                }
                Text(turn.text)
                    .font(.system(size: theme.bodySize))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 13)
                    .padding(.vertical, 10)
                    .background(turn.role == .user ? theme.ink : theme.surface)
                    .foregroundStyle(turn.role == .user ? theme.onInk : theme.text)
                    .clipShape(RoundedRectangle(cornerRadius: theme.cardRadius))
            }
            .frame(maxWidth: 620, alignment: .leading)
            .accessibilityElement(children: .combine)
            .accessibilityIdentifier(turn.role == .user ? "user-message" : "assistant-message")
            if turn.role == .assistant { Spacer(minLength: 40) }
        }
    }

    private func errorBanner(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(theme.danger)
                .accessibilityHidden(true)
            Text(text)
                .font(.system(size: theme.bodySize))
                .foregroundStyle(theme.danger)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(12)
        .background(theme.danger.opacity(0.08))
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("error")
    }

    private var composer: some View {
        VStack(spacing: 8) {
            HStack(alignment: .bottom, spacing: 10) {
                TextField("Write a prompt", text: $model.prompt, axis: .vertical)
                    .lineLimit(1...5)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: theme.bodySize))
                    .onSubmit { model.send() }
                    .accessibilityLabel("Prompt")
                    .accessibilityIdentifier("prompt")
                if model.isRunning {
                    Button("Stop") { model.stop() }
                        .buttonStyle(.borderedProminent)
                        .tint(theme.danger)
                        .keyboardShortcut(".", modifiers: .command)
                        .accessibilityIdentifier("stop")
                } else {
                    Button("Send") { model.send() }
                        .buttonStyle(.borderedProminent)
                        .tint(theme.ink)
                        .keyboardShortcut(.return, modifiers: .command)
                        .disabled(model.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        .accessibilityIdentifier("send")
                }
            }
            Text("⌘Return to send · streaming replies are selectable · max 64 tokens")
                .font(.system(size: theme.captionSize))
                .foregroundStyle(theme.tertiary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(16)
        .background(theme.surface)
    }

    private func scrollToEnd(_ proxy: ScrollViewProxy) {
        guard let id = model.turns.last?.id else { return }
        withAnimation(.easeOut(duration: 0.12)) { proxy.scrollTo(id, anchor: .bottom) }
    }
}
