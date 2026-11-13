// SPDX-License-Identifier: Apache-2.0
// Unit / contract tests for the unified multimodal engine v1.
//
// These tests drive the engine through the real Coordinator over ipc sockets,
// but the ASR and VLM "stages" are bound PULL sockets and a PUSH socket: no
// model worker runs. They verify the ASR -> VLM orchestration contract: event
// order, deterministic aggregation, cancellation, stage failures,
// backpressure, and duplicate/out-of-order handling. They are not real-model
// E2E tests.
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "umminfer/comm/comm.hpp"
#include "umminfer/pipeline/pipeline.hpp"
#include "umminfer/pipeline/unified.hpp"

using namespace umminfer;

namespace {

int g_failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++g_failures;
  }
}

void wait_until(const std::function<bool()>& condition, const char* message) {
  for (int i = 0; i < 200; ++i) {
    if (condition()) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  check(false, message);
}

// Callbacks run on the coordinator pump thread; the lock covers the handoff.
struct EventLog {
  std::mutex mu;
  std::vector<UnifiedEvent> events;
  std::map<std::string, UnifiedEvent> terminals;
  // Every on_terminal invocation, counted separately from the first-only
  // terminal map. Without this, emplace makes terminal_count() always <= 1 and
  // a duplicate-terminal bug could not be observed.
  std::map<std::string, int> terminal_callbacks;

  void add(const UnifiedEvent& event) {
    std::lock_guard<std::mutex> lock(mu);
    events.push_back(event);
  }
  void finish(const UnifiedEvent& event) {
    std::lock_guard<std::mutex> lock(mu);
    // Keep the first terminal for state checks, but count every callback.
    terminals.emplace(event.request_id, event);
    ++terminal_callbacks[event.request_id];
  }
  bool has_terminal(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu);
    return terminals.count(id) != 0;
  }
  std::optional<UnifiedEvent> terminal_for(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu);
    auto it = terminals.find(id);
    return it == terminals.end() ? std::nullopt : std::optional(it->second);
  }
  size_t terminal_count(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu);
    auto it = terminal_callbacks.find(id);
    return it == terminal_callbacks.end() ? 0 : static_cast<size_t>(it->second);
  }
  // Backwards-compatible single-request view used by the other tests.
  bool has_terminal() {
    std::lock_guard<std::mutex> lock(mu);
    return !terminals.empty();
  }
  std::optional<UnifiedEvent> terminal_event() {
    std::lock_guard<std::mutex> lock(mu);
    return terminals.empty() ? std::nullopt
                             : std::optional(terminals.begin()->second);
  }
  size_t size() {
    std::lock_guard<std::mutex> lock(mu);
    return events.size();
  }
  std::vector<UnifiedEvent> snapshot() {
    std::lock_guard<std::mutex> lock(mu);
    return events;
  }
};

// One mkdtemp(3) directory per Harness: a unique 0700 directory this run
// owns. Socket files keep their basenames inside it, and teardown removes only
// those paths plus the owned directory. No predictable global path is ever
// pre-deleted, so a foreign process cannot lose its sockets.
struct IpcDir {
  std::string path;
  std::vector<std::string> files;

  static IpcDir create(const char* tag) {
    std::string templ =
        std::string("/tmp/omni-unified-") + tag + "-XXXXXX";
    std::vector<char> buffer(templ.begin(), templ.end());
    buffer.push_back('\0');
    char* made = mkdtemp(buffer.data());
    if (made == nullptr) {
      throw std::runtime_error(std::string("mkdtemp failed: ") +
                               std::strerror(errno));
    }
    IpcDir dir;
    dir.path = made;  // mkdtemp itself applies 0700
    for (const char* name : {"asr", "vlm", "done", "abort"}) {
      dir.files.push_back(dir.path + "/" + name);
    }
    return dir;
  }

  ~IpcDir() {
    for (const auto& file : files) std::remove(file.c_str());
    // rmdir only succeeds when the directory holds nothing this run did not
    // create; it never recurses into a foreign path.
    ::rmdir(path.c_str());
  }

  std::string endpoint(const char* name) const {
    return "ipc://" + path + "/" + name;
  }
};

Value::Map part(const std::string& type) {
  Value::Map out;
  out["type"] = Value(type);
  return out;
}

StagePayload audio_request(const std::string& modality = "audio") {
  StagePayload payload;
  Value::Map message;
  message["role"] = Value("user");
  Value::Array content;
  content.push_back(Value(part("input_audio")));
  message["content"] = Value(std::move(content));
  Value::Array messages;
  messages.push_back(Value(std::move(message)));
  Value::Array modalities;
  modalities.push_back(Value(modality));
  Value::Map inputs;
  inputs["messages"] = Value(std::move(messages));
  inputs["modalities"] = Value(std::move(modalities));
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}

StagePayload image_part_request() {
  StagePayload payload;
  Value::Map message;
  message["role"] = Value("user");
  Value::Array content;
  content.push_back(Value(part("image_url")));
  message["content"] = Value(std::move(content));
  Value::Array messages;
  messages.push_back(Value(std::move(message)));
  Value::Map inputs;
  inputs["messages"] = Value(std::move(messages));
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}

Value::Map text_chunk(const std::string& text, bool is_final) {
  Value::Map chunk;
  chunk["text"] = Value(text);
  if (is_final) chunk["is_final"] = Value(true);
  return chunk;
}

Value::Map text_result(const std::string& text) {
  Value::Map result;
  result["text"] = Value(text);
  return result;
}

// Real Coordinator with two fake stage inboxes and one event pusher. No
// compute runs anywhere.
struct Harness {
  IpcDir dir;
  PullSocket asr_recv;
  PullSocket vlm_recv;
  PushSocket event_push;
  Coordinator coordinator;
  std::thread pump;
  // broadcast_abort pushes AbortMessage on the same stage PULL sockets, so the
  // fake stages observe cancels here. The PUSH path is the reliable ordered
  // one; the test deliberately does not depend on PUB slow-joiner timing.
  std::mutex abort_mu;
  std::vector<std::string> aborted_legs;

  explicit Harness(const char* tag, bool start_pump_thread = true)
      : dir(IpcDir::create(tag)),
        asr_recv(dir.endpoint("asr"), true),
        vlm_recv(dir.endpoint("vlm"), true),
        event_push(dir.endpoint("done")),
        coordinator(dir.endpoint("done"), dir.endpoint("abort")) {
    asr_recv.start();
    vlm_recv.start();
    coordinator.register_stage("asr", dir.endpoint("asr"));
    coordinator.register_stage("vlm", dir.endpoint("vlm"));
    coordinator.start();
    if (start_pump_thread) start_pump();
    event_push.connect();
  }

  void start_pump() {
    pump = std::thread([this] { coordinator.run(); });
  }

  ~Harness() {
    coordinator.stop();
    if (pump.joinable()) pump.join();
    event_push.close();
    asr_recv.close();
    vlm_recv.close();
  }

  std::optional<SubmitMessage> next_submit(PullSocket& socket) {
    // Abort frames share the stage PULL socket: broadcast_abort pushes to every
    // stage socket, and the real Stage drains them in handle_message. The fake
    // stage records them and skips to the next submit frame.
    while (true) {
      auto msg = socket.recv_timeout(2000);
      if (!msg) return std::nullopt;
      if (auto* abort = std::get_if<AbortMessage>(&*msg)) {
        std::lock_guard<std::mutex> lock(abort_mu);
        aborted_legs.push_back(abort->request_id);
        continue;
      }
      if (auto* submit = std::get_if<SubmitMessage>(&*msg)) return *submit;
    }
  }

  bool saw_abort(const std::string& leg_id) {
    std::lock_guard<std::mutex> lock(abort_mu);
    for (const auto& id : aborted_legs) {
      if (id == leg_id) return true;
    }
    return false;
  }

  void record_abort(const std::string& leg_id) {
    std::lock_guard<std::mutex> lock(abort_mu);
    aborted_legs.push_back(leg_id);
  }

  // Negative assertion: only the PUSH abort path can arrive, never a submit.
  bool no_submit(PullSocket& socket) {
    while (true) {
      auto msg = socket.recv_timeout(200);
      if (!msg) return true;
      if (auto* abort = std::get_if<AbortMessage>(&*msg)) {
        std::lock_guard<std::mutex> lock(abort_mu);
        aborted_legs.push_back(abort->request_id);
        continue;
      }
      if (auto* submit = std::get_if<SubmitMessage>(&*msg)) {
        (void)submit;
        return false;
      }
    }
  }

  // Drains both stage inboxes until the abort for leg_id is observed.
  bool wait_abort(const std::string& leg_id) {
    for (int i = 0; i < 40 && !saw_abort(leg_id); ++i) {
      for (PullSocket* socket : {&asr_recv, &vlm_recv}) {
        auto msg = socket->recv_timeout(50);
        if (!msg) continue;
        if (auto* abort = std::get_if<AbortMessage>(&*msg)) {
          std::lock_guard<std::mutex> lock(abort_mu);
          aborted_legs.push_back(abort->request_id);
        }
      }
    }
    return saw_abort(leg_id);
  }

  void stream(const std::string& id, const std::string& stage, Value chunk) {
    StreamMessage msg;
    msg.request_id = id;
    msg.from_stage = stage;
    msg.chunk = std::move(chunk);
    event_push.send(Message{std::move(msg)});
  }

  void complete(const std::string& id, const std::string& stage, bool success,
                Value result,
                std::optional<std::string> error = std::nullopt) {
    CompleteMessage msg;
    msg.request_id = id;
    msg.from_stage = stage;
    msg.success = success;
    msg.result = std::move(result);
    msg.error = std::move(error);
    event_push.send(Message{std::move(msg)});
  }
};

// Streaming ASR partials/final -> VLM: ordering, gap-free seq, aggregation.
void test_streaming_asr_to_vlm() {
  Harness h("stream");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");

  h.stream(id, "asr", text_chunk("hel", false));
  h.stream(id, "asr", text_chunk("lo", false));
  h.stream(id, "asr", text_chunk("hello world", true));
  wait_until([&] { return log.size() == 3; },
             "ASR partial/final events were not delivered");

  // The VLM leg starts when the ASR completion arrives, not on the final chunk.
  h.complete(id, "asr", true, text_result("hello world"));
  // The final transcript replaces the concatenated partial deltas, and the VLM
  // leg starts with exactly that text as its prompt.
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "VLM stage received no submit");
  if (vlm_submit) {
    check(vlm_submit->request_id == id + ":vlm", "VLM leg id is not a child id");
    StagePayload vlm_payload = StagePayload::from_value(vlm_submit->data);
    const Value& messages = *vlm_payload.request.inputs.find("messages");
    const Value& first = messages.as_array().at(0);
    check(first.find("content")->as_str() == "hello world",
          "VLM prompt is not the final ASR transcript");
  }

  const auto asr_leg_info = h.coordinator.info(id);
  check(asr_leg_info && asr_leg_info->state == "completed",
        "the ASR parent leg did not complete before the VLM barrier");
  const auto vlm_phase_info = engine.info(id);
  check(vlm_phase_info && vlm_phase_info->request_id == id &&
            vlm_phase_info->state == "running" &&
            vlm_phase_info->current_stage == kVlmStageId,
        "info() exposed the completed ASR parent leg instead of unified VLM state");

  h.stream(id + ":vlm", "vlm", text_chunk("hi", false));
  h.stream(id + ":vlm", "vlm", text_chunk(" there", false));
  h.complete(id + ":vlm", "vlm", true, text_result("hi there"));
  check(engine.wait(id, 5.0), "unified request did not complete");

  const auto events = log.snapshot();
  if (events.size() != 6) {
    check(false, "wrong event count for the full ASR -> VLM flow");
    engine.release(id);
    return;
  }
  const UnifiedEvent::Kind expected_kinds[] = {
      UnifiedEvent::Kind::kAsrPartial, UnifiedEvent::Kind::kAsrPartial,
      UnifiedEvent::Kind::kAsrFinal,   UnifiedEvent::Kind::kVlmDelta,
      UnifiedEvent::Kind::kVlmDelta,   UnifiedEvent::Kind::kComplete};
  for (size_t i = 0; i < events.size(); ++i) {
    check(events[i].seq == static_cast<int64_t>(i),
          "unified event sequence is not gap-free from zero");
    check(events[i].kind == expected_kinds[i], "unified event kind out of order");
    check(events[i].request_id == id, "an event leaked the internal child id");
  }
  check(events[2].is_final && events[2].text == "hello world",
        "ASR final event has the wrong text");
  check(events[5].text == "hi there", "complete event lost the VLM aggregation");
  const auto final_info = engine.info(id);
  check(final_info && final_info->state == "completed" &&
            final_info->current_stage == kVlmStageId &&
            final_info->result.find("text") != nullptr &&
            final_info->result.find("text")->as_str() == "hi there",
        "info() did not retain the unified VLM terminal state");
  engine.release(id);
}

// A non-streaming ASR stage emits only a completion; the engine synthesizes
// the ASR final from its result text.
void test_nonstream_asr_synthesizes_final() {
  Harness h("nonstream");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.complete(id, "asr", true, text_result("spoken text"));
  check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
  h.complete(id + ":vlm", "vlm", true, text_result("answer"));
  check(engine.wait(id, 5.0), "non-streaming request did not complete");

  bool saw_final = false;
  for (const auto& event : log.snapshot()) {
    if (event.kind == UnifiedEvent::Kind::kAsrFinal) {
      saw_final = true;
      check(event.text == "spoken text", "synthesized final has the wrong text");
    }
  }
  check(saw_final, "no ASR final was synthesized from a non-streaming result");
  engine.release(id);
}

// The ASR contract accepts a plain string result, matching
// serve_detail::asr_result_text. The engine must extract it and use it as the
// VLM prompt.
void test_asr_plain_string_result() {
  Harness h("plainstring");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.complete(id, "asr", true, Value("spoken text"));  // plain string, no map

  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "plain-string result did not reach the VLM leg");
  if (vlm_submit) {
    StagePayload vlm_payload = StagePayload::from_value(vlm_submit->data);
    const Value& messages = *vlm_payload.request.inputs.find("messages");
    check(messages.as_array().at(0).find("content")->as_str() == "spoken text",
          "the VLM prompt is not the plain-string ASR transcript");
  }
  h.complete(id + ":vlm", "vlm", true, text_result("answer"));
  check(engine.wait(id, 5.0), "plain-string request did not complete");
  bool saw_final = false;
  for (const auto& event : log.snapshot()) {
    if (event.kind == UnifiedEvent::Kind::kAsrFinal) {
      saw_final = true;
      check(event.text == "spoken text", "plain-string final has the wrong text");
    }
  }
  check(saw_final, "no ASR final was synthesized from a plain-string result");
  engine.release(id);
}

// A result with no usable text is a contract failure: the engine must emit
// kError on the ASR stage and must never start the VLM leg, rather than
// forwarding an empty prompt.
void test_asr_malformed_result_fails() {
  std::vector<Value> bad_results;
  bad_results.emplace_back(Value::Map{});
  bad_results.emplace_back(Value(42));
  for (const Value& bad : bad_results) {
    Harness h("malformed");
    UnifiedEngine engine(h.coordinator, UnifiedConfig{});
    EventLog log;
    engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

    const std::string id = engine.submit(audio_request());
    check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
    h.complete(id, "asr", true, bad);
    check(engine.wait(id, 5.0), "a malformed ASR result did not settle");

    const auto terminal = log.terminal_for(id);
    check(terminal.has_value() &&
              terminal->kind == UnifiedEvent::Kind::kError &&
              terminal->stage_id == kAsrStageId,
          "a malformed ASR result did not produce an ASR-stage kError");
    check(terminal && !terminal->error.empty(),
          "the malformed-result error carried no message");
    check(h.no_submit(h.vlm_recv),
          "a malformed ASR result still started the VLM leg");
    engine.release(id);
  }
}

// A malformed streaming final must fail before the completion can start VLM.
void test_streaming_asr_malformed_final_fails() {
  std::vector<Value> bad_finals;
  Value::Map missing_text;
  missing_text["is_final"] = Value(true);
  bad_finals.emplace_back(std::move(missing_text));
  Value::Map numeric_text;
  numeric_text["is_final"] = Value(true);
  numeric_text["text"] = Value(42);
  bad_finals.emplace_back(std::move(numeric_text));

  for (const Value& bad : bad_finals) {
    Harness h("streammalformed");
    UnifiedEngine engine(h.coordinator, UnifiedConfig{});
    EventLog log;
    engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };
    engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

    const std::string id = engine.submit(audio_request());
    check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
    h.stream(id, "asr", bad);
    // The same PUSH -> PULL queue makes this completion a FIFO barrier: the old
    // code treats the malformed final as valid and starts VLM here.
    h.complete(id, "asr", true, text_result("must not run"));
    check(engine.wait(id, 5.0), "a malformed streaming final did not settle");

    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kError &&
              terminal->stage_id == kAsrStageId && !terminal->error.empty(),
          "a malformed streaming final did not produce one ASR kError");
    check(log.terminal_count(id) == 1,
          "a malformed streaming final produced duplicate terminals");
    check(log.size() == 1 && log.snapshot()[0].kind == UnifiedEvent::Kind::kError,
          "a malformed streaming final emitted an ASR final or success event");
    check(h.no_submit(h.vlm_recv),
          "a malformed streaming final still started the VLM leg");
    engine.release(id);
  }
}

void test_unknown_id_wait_fails() {
  Harness h("unknownwait");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  check(!engine.wait("unknown", 0.0), "wait() reported success for an unknown id");
  check(!engine.info("unknown"), "info() returned state for an unknown id");
}

// cancel() broadcasts an abort to both stages and settles the request.
void test_cancel_propagates_to_both_stages() {
  Harness h("cancel");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  check(engine.cancel(id), "cancel rejected an active request");
  check(h.wait_abort(id), "abort did not reach the ASR stage");
  wait_until([&] { return log.has_terminal(); },
             "cancellation did not settle the request");
  check(log.terminal_event() && log.terminal_event()->kind == UnifiedEvent::Kind::kCancelled,
        "cancel produced the wrong terminal event");
  check(engine.info(id)->state == "aborted", "coordinator state is not aborted");
  check(!engine.cancel(id), "a second cancel was not idempotent");
  engine.release(id);

  // Second request: cancel during the VLM leg aborts the VLM child id.
  const std::string id2 = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.complete(id2, "asr", true, text_result("transcript"));
  check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
  check(engine.cancel(id2), "cancel rejected the VLM-leg request");
  check(h.wait_abort(id2 + ":vlm"),
        "VLM-leg cancel did not abort the active stage leg");
  engine.release(id2);
}

// An ASR failure stops the flow before any VLM submit.
void test_asr_failure() {
  Harness h("asrfail");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.complete(id, "asr", false, Value{}, std::string("asr broke"));
  check(engine.wait(id, 5.0), "ASR failure did not settle the request");
  check(log.terminal_event() && log.terminal_event()->kind == UnifiedEvent::Kind::kError &&
            log.terminal_event()->stage_id == kAsrStageId &&
            log.terminal_event()->error == "asr broke",
        "ASR failure produced the wrong terminal event");
  check(h.no_submit(h.vlm_recv),
        "a failed ASR leg still submitted to the VLM stage");
  engine.release(id);
}

// A VLM failure is observable with the VLM stage id.
void test_vlm_failure() {
  Harness h("vlmfail");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.complete(id, "asr", true, text_result("transcript"));
  check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
  h.complete(id + ":vlm", "vlm", false, Value{}, std::string("vlm broke"));
  check(engine.wait(id, 5.0), "VLM failure did not settle the request");
  check(log.terminal_event() && log.terminal_event()->kind == UnifiedEvent::Kind::kError &&
            log.terminal_event()->stage_id == kVlmStageId &&
            log.terminal_event()->error == "vlm broke",
        "VLM failure produced the wrong terminal event");
  engine.release(id);
}

// VLM results accept exactly the public bare-string/text/content shapes.
void test_vlm_result_text_contract() {
  std::vector<std::pair<Value, std::string>> valid_results;
  valid_results.emplace_back(Value("bare answer"), "bare answer");
  valid_results.emplace_back(Value(text_result("text answer")), "text answer");
  Value::Map content_result;
  content_result["content"] = Value("content answer");
  valid_results.emplace_back(Value(std::move(content_result)), "content answer");

  for (const auto& [result, expected] : valid_results) {
    Harness h("vlmresult");
    UnifiedEngine engine(h.coordinator, UnifiedConfig{});
    EventLog log;
    engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

    const std::string id = engine.submit(audio_request());
    check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
    h.complete(id, "asr", true, text_result("transcript"));
    check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
    h.complete(id + ":vlm", "vlm", true, result);
    check(engine.wait(id, 5.0), "a valid VLM result did not settle");

    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kComplete &&
              terminal->text == expected,
          "a valid VLM result shape lost its text");
    check(log.terminal_count(id) == 1,
          "a valid VLM result produced duplicate terminals");
    engine.release(id);
  }

  std::vector<Value> bad_results;
  bad_results.emplace_back(Value(42));
  bad_results.emplace_back(Value::Map{});
  Value::Map numeric_text;
  numeric_text["text"] = Value(42);
  bad_results.emplace_back(std::move(numeric_text));
  for (const Value& bad : bad_results) {
    Harness h("vlmmalformed");
    UnifiedEngine engine(h.coordinator, UnifiedConfig{});
    EventLog log;
    engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

    const std::string id = engine.submit(audio_request());
    check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
    h.complete(id, "asr", true, text_result("transcript"));
    check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
    h.stream(id + ":vlm", "vlm", text_chunk("delta", false));
    h.complete(id + ":vlm", "vlm", true, bad);
    check(engine.wait(id, 5.0), "a malformed VLM result did not settle");

    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kError &&
              terminal->stage_id == kVlmStageId && !terminal->error.empty(),
          "a malformed VLM result did not produce exactly one VLM kError");
    check(log.terminal_count(id) == 1,
          "a malformed VLM result produced duplicate terminals");
    engine.release(id);
  }
}

// The admission cap is a hard bound: overflow throws, a finished slot reopens.
void test_queue_cap_backpressure() {
  Harness h("queue");
  UnifiedConfig config;
  config.max_in_flight = 2;
  UnifiedEngine engine(h.coordinator, config);

  const std::string a = engine.submit(audio_request());
  const std::string b = engine.submit(audio_request());
  check(engine.in_flight() == 2, "admission count is wrong");
  bool full = false;
  try {
    engine.submit(audio_request());
  } catch (const QueueFullError&) {
    full = true;
  }
  check(full, "the queue cap did not reject an overflow request");

  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.complete(a, "asr", true, text_result("t"));
  check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
  h.complete(a + ":vlm", "vlm", true, text_result("r"));
  check(engine.wait(a, 5.0), "request did not complete");
  engine.release(a);
  wait_until([&] { return engine.in_flight() == 1; },
             "a finished request did not free an admission slot");
  const std::string c = engine.submit(audio_request());
  check(engine.in_flight() == 2, "a freed slot did not admit a new request");
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  engine.release(b);
  engine.release(c);
}

// A repeated ASR final is idempotent; an out-of-order chunk is dropped.
void test_duplicate_final_and_out_of_order() {
  Harness h("dup");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.stream(id, "asr", text_chunk("first", true));
  h.stream(id, "asr", text_chunk("second", true));  // duplicate final
  wait_until([&] { return log.size() == 1; },
             "the duplicate ASR final was not idempotent");
  check(log.snapshot()[0].text == "first",
        "the duplicate final overwrote the transcript");

  // A VLM-stage chunk before the VLM leg exists is out of order and dropped.
  h.stream(id, "vlm", text_chunk("early", false));
  h.complete(id, "asr", true, text_result("first"));
  check(h.next_submit(h.vlm_recv).has_value(), "VLM stage received no submit");
  // An ASR chunk after the VLM leg started is late and dropped.
  h.stream(id, "asr", text_chunk("late", false));
  h.stream(id + ":vlm", "vlm", text_chunk("answer", false));
  h.complete(id + ":vlm", "vlm", true, text_result("answer"));
  check(engine.wait(id, 5.0), "request did not complete");

  size_t finals = 0, deltas = 0;
  for (const auto& event : log.snapshot()) {
    if (event.kind == UnifiedEvent::Kind::kAsrFinal) ++finals;
    if (event.kind == UnifiedEvent::Kind::kVlmDelta) ++deltas;
  }
  check(finals == 1, "more than one ASR final event was emitted");
  check(deltas == 1, "an out-of-order chunk leaked into the event stream");
  engine.release(id);
}

// Image and video stay declarations: any request requiring them is rejected.
void test_image_video_rejected() {
  Harness h("media");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});

  bool image_modality_rejected = false;
  try {
    engine.submit(audio_request("image"));
  } catch (const ProtocolError&) {
    image_modality_rejected = true;
  }
  check(image_modality_rejected, "an image modality was accepted without a worker");

  bool video_modality_rejected = false;
  try {
    engine.submit(audio_request("video"));
  } catch (const ProtocolError&) {
    video_modality_rejected = true;
  }
  check(video_modality_rejected, "a video modality was accepted without a worker");

  bool image_part_rejected = false;
  try {
    engine.submit(image_part_request());
  } catch (const ProtocolError&) {
    image_part_rejected = true;
  }
  check(image_part_rejected, "an image content part was accepted without a worker");

  bool bad_contract_rejected = false;
  try {
    StagePayload empty;
    engine.submit(empty);
  } catch (const ProtocolError&) {
    bad_contract_rejected = true;
  }
  check(bad_contract_rejected, "a request without messages was accepted");
}

// Cancel before the queued ASR completion is pumped. With no pump thread the
// Coordinator still marks the request aborted synchronously inside cancel(),
// then the stale success completion must be ignored: one kCancelled, no VLM.
void test_cancel_before_pumped_completion() {
  Harness h("stale", /*start_pump_thread=*/false);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(),
        "ASR stage received no submit");

  // A sentinel request whose terminal we can observe. Both completions go on
  // the one PUSH -> PULL completion socket, which is FIFO, so seeing the
  // sentinel terminal proves the stale completion ahead of it was consumed.
  const std::string sentinel = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(),
        "sentinel ASR stage received no submit");

  // Queue the ASR success for the cancelled request, then the sentinel's
  // failure (a failure terminates without starting a VLM leg). Nobody drains
  // either message until the pump starts.
  h.complete(id, "asr", true, text_result("transcript"));
  h.complete(sentinel, "asr", false, Value{}, std::string("sentinel failure"));
  check(engine.cancel(id), "cancel rejected an active request");
  check(log.terminal_for(id).has_value() &&
            log.terminal_for(id)->kind == UnifiedEvent::Kind::kCancelled,
        "cancel before pumping did not settle as kCancelled");

  // Drain the queue. The sentinel terminal is a deterministic processing
  // barrier: FIFO delivery means the stale completion was already dequeued.
  h.start_pump();
  wait_until([&] { return log.has_terminal(sentinel); },
             "the sentinel completion was not processed");
  check(log.terminal_for(sentinel) &&
            log.terminal_for(sentinel)->kind == UnifiedEvent::Kind::kError,
        "the sentinel did not terminate as expected");

  // The stale success reached a terminal record, so it neither resurrected the
  // request nor started the VLM leg, and produced no second terminal.
  check(log.terminal_count(id) == 1, "the stale completion added a terminal");
  check(h.no_submit(h.vlm_recv), "the VLM leg ran after cancellation");
  engine.release(id);
  engine.release(sentinel);
}

// Both interleavings of the ASR->VLM handoff, repeated. Every request must end
// as exactly one kCancelled; any VLM leg that did start must be aborted.
void test_cancel_handoff_race_stress() {
  Harness h("racestress");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  constexpr int kIterations = 40;
  for (int i = 0; i < kIterations; ++i) {
    const std::string id = engine.submit(audio_request());
    check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
    if ((i % 2) == 0) {
      // Completion queued first; cancel may win or lose the pump race.
      h.complete(id, "asr", true, text_result("t"));
      engine.cancel(id);
    } else {
      engine.cancel(id);
      h.complete(id, "asr", true, text_result("t"));
    }
    check(engine.wait(id, 10.0), "a stress iteration did not settle");
    const auto terminal = log.terminal_for(id);
    check(terminal && terminal->kind == UnifiedEvent::Kind::kCancelled,
          "a handoff race did not settle as kCancelled");
    check(log.terminal_count(id) == 1,
          "a handoff race produced a duplicate terminal");

    // Every VLM submit that escaped must have a matching abort.
    std::set<std::string> vlm_submits;
    while (true) {
      auto msg = h.vlm_recv.recv_timeout(20);
      if (!msg) break;
      if (auto* submit = std::get_if<SubmitMessage>(&*msg)) {
        vlm_submits.insert(submit->request_id);
      } else if (auto* abort = std::get_if<AbortMessage>(&*msg)) {
        h.record_abort(abort->request_id);
      }
    }
    for (const auto& child : vlm_submits) {
      check(h.saw_abort(child), "a started VLM leg was not aborted");
    }
    engine.release(id);
  }
}

// Deterministic reproduction of the ASR->VLM handoff race:
//   1. the Coordinator's ASR record is already "completed", but the engine has
//      not observed that completion (its record is still in phase kAsr);
//   2. cancel() then aborts an already-terminal ASR id, which is a no-op;
//   3. the success completion is handed to the engine afterwards.
// The engine must settle as exactly one kCancelled and must not start the VLM
// leg. This fails on the pre-fix engine, which started the VLM leg instead.
void test_cancel_after_asr_completed_swallows_vlm() {
  Harness h("swallow", /*start_pump_thread=*/false);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");

  // Save the engine sink, then install a dropping sink that resolves a promise
  // when the ASR completion is consumed. The pump only ever reads this dropping
  // sink; it is installed before start_pump and never reassigned, so there is
  // no concurrent read/write of coordinator.on_complete. The saved engine sink
  // is a separate copy invoked only on this test thread.
  auto engine_sink = h.coordinator.on_complete;
  std::promise<void> dropped;
  std::future<void> dropped_future = dropped.get_future();
  h.coordinator.on_complete = [&dropped, id](const RequestInfo& info) {
    if (info.request_id == id) dropped.set_value();
  };
  h.start_pump();
  h.complete(id, "asr", true, text_result("transcript"));

  // The future synchronizes with the pump thread: once wait() returns, the
  // dropping callback has run past set_value, the Coordinator ASR record is
  // "completed", and the engine has provably never seen that completion.
  dropped_future.wait();
  {
    auto info = h.coordinator.info(id);
    check(info && info->state == "completed",
          "the ASR coordinator record did not settle as completed");
  }

  // ASR is terminal in the Coordinator: abort(id) is now a no-op, exactly the
  // reviewer's window. cancel must still record the intent.
  check(engine.cancel(id), "cancel rejected the handoff request");

  // Deliver the success the engine has not seen yet, on this thread.
  RequestInfo asr_done;
  asr_done.request_id = id;
  asr_done.state = "completed";
  asr_done.current_stage = "asr";
  asr_done.result = text_result("transcript");
  engine_sink(asr_done);

  const auto terminal = log.terminal_for(id);
  check(terminal.has_value() &&
            terminal->kind == UnifiedEvent::Kind::kCancelled,
        "a cancel whose abort lost to ASR completion did not yield kCancelled");
  check(log.terminal_count(id) == 1, "the handoff produced a duplicate terminal");
  check(h.no_submit(h.vlm_recv), "the VLM leg ran after a handoff cancellation");
  engine.release(id);
}

// The destructor cannot return while a pump callback still uses the engine.
// Promises, not sleeps, fix the interleaving. A sink copied before destruction
// must also remain safe when invoked after destruction.
void test_destructor_synchronizes_inflight_callback() {
  Harness h("destruct");
  std::atomic<int> forwarded_calls{0};
  std::promise<void> entered;
  std::future<void> entered_future = entered.get_future();
  std::promise<void> release;
  std::shared_future<void> release_future = release.get_future().share();
  h.coordinator.on_stream = [&](const StreamMessage&) {
    if (forwarded_calls.fetch_add(1) == 0) {
      entered.set_value();
      release_future.wait();
    }
  };

  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  EventLog log;
  engine->on_event = [&](const UnifiedEvent& event) { log.add(event); };
  const std::string id = engine->submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.stream(id, "asr", text_chunk("partial", false));
  entered_future.wait();
  const auto copied_sink = h.coordinator.on_stream;

  std::promise<void> destroy_started;
  std::future<void> destroy_started_future = destroy_started.get_future();
  std::promise<void> destroyed;
  std::future<void> destroyed_future = destroyed.get_future();
  std::thread destroyer([&] {
    destroy_started.set_value();
    engine.reset();
    destroyed.set_value();
  });
  destroy_started_future.wait();
  const std::future_status early = destroyed_future.wait_for(std::chrono::seconds(1));
  check(early == std::future_status::timeout,
        "destructor returned while an inflight callback still used the engine");
  release.set_value();
  destroyed_future.wait();
  destroyer.join();
  check(log.size() == 1, "the inflight callback did not finish before destruction");

  StreamMessage after;
  after.request_id = "unknown";
  after.from_stage = "asr";
  after.chunk = text_chunk("after", false);
  copied_sink(after);
  check(forwarded_calls.load() == 2,
        "a copied sink did not forward safely after engine destruction");
}

}  // namespace

int main() {
  alarm(60);
  test_streaming_asr_to_vlm();
  test_nonstream_asr_synthesizes_final();
  test_asr_plain_string_result();
  test_asr_malformed_result_fails();
  test_streaming_asr_malformed_final_fails();
  test_unknown_id_wait_fails();
  test_cancel_propagates_to_both_stages();
  test_asr_failure();
  test_vlm_failure();
  test_vlm_result_text_contract();
  test_queue_cap_backpressure();
  test_duplicate_final_and_out_of_order();
  test_image_video_rejected();
  test_cancel_before_pumped_completion();
  test_cancel_after_asr_completed_swallows_vlm();
  test_cancel_handoff_race_stress();
  test_destructor_synchronizes_inflight_callback();
  return g_failures == 0 ? 0 : 1;
}
