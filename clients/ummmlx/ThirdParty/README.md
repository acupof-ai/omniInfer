# Voxt attribution

Source: https://github.com/sgl-project/sglang-omni/tree/main/Voxt
Commit: `5645daf0480e9bf572b45a2bba8b8e2f1e642daa`
License: Apache-2.0. See `Voxt/LICENSE` and `Voxt/NOTICE`.

The Go files under `Voxt/` are unmodified reference copies.

Modified ports in `Sources/ummmlx/`:

- `ChatClient.swift`: ports the SSE data-line framing and OpenAI error extraction
  from `internal/llm/openai_http.go` and `internal/llm/openai.go`. Changes include
  Swift concurrency, incremental byte parsing, size limits, strict `[DONE]`
  handling, and the ummInfer request contract.
- `Theme.swift`: ports dimensions and the neutral/teal palette from
  `ui_tokens.go`. Changes replace MyGo types with SwiftUI colors and remove
  unrelated recording, selection, and provider states.

Voxt voice capture, ASR runtimes, model downloads, account systems, cloud
presets, MyGo, and the Swift helper are not used.
