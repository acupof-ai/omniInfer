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
#include "umminfer/audio/audio.hpp"
#include "umminfer/pipeline/pipeline.hpp"
#include "umminfer/pipeline/unified.hpp"
#include "umminfer/serve/openai_api.hpp"
#include "client/http.hpp"
#include <nlohmann/json.hpp>

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

Value image_part(const std::string& url = "data:image/png;base64,AA") {
  Value::Map image_url;
  image_url["url"] = Value(url);
  Value::Map out = part("image_url");
  out["image_url"] = Value(std::move(image_url));
  return Value(std::move(out));
}

StagePayload image_part_request(const std::string& url = "data:image/png;base64,AA") {
  StagePayload payload;
  Value::Map message;
  message["role"] = Value("user");
  Value::Array content;
  content.push_back(image_part(url));
  message["content"] = Value(std::move(content));
  Value::Array messages;
  messages.push_back(Value(std::move(message)));
  Value::Map inputs;
  inputs["messages"] = Value(std::move(messages));
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}

// One request carrying real PCM for the ASR leg and one image part for the VLM
// leg. The image rides on the original messages; the ASR leg never sees it.
StagePayload joint_request(const std::string& session_id = "") {
  StagePayload payload;
  Value::Map message;
  message["role"] = Value("user");
  Value::Array content;
  content.push_back(image_part());
  content.push_back(Value(part("input_audio")));
  message["content"] = Value(std::move(content));
  Value::Array messages;
  messages.push_back(Value(std::move(message)));
  Value::Array modalities;
  modalities.push_back(Value("audio"));
  modalities.push_back(Value("image"));
  Value::Map inputs;
  inputs["messages"] = Value(std::move(messages));
  inputs["modalities"] = Value(std::move(modalities));
  inputs["audio"] = Value::bin(std::string(800, '\0'));
  if (!session_id.empty()) inputs["session_id"] = Value(session_id);
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}

// Real PCM selects the ASR leg. The optional flag is the documented opt-in for
// incremental ASR events through the unified chat route.
StagePayload audio_leg_request(bool asr_partials) {
  StagePayload payload = audio_request();
  Value::Map inputs = payload.request.inputs.as_map();
  inputs["audio"] = Value::bin(std::string(800, '\0'));
  if (asr_partials) inputs["asr_partials"] = Value(true);
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

  explicit Harness(const char* tag, bool start_pump_thread = false)
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
  h.start_pump();
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

// Opt-in ASR partials: the ASR leg request streams, partial deltas precede one
// final, and the final replaces the concatenated partials for the VLM prompt.
void test_asr_partials_opt_in() {
  Harness h("partials");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(audio_leg_request(true));
  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "opt-in ASR partial request was not submitted");
  if (asr_submit) {
    StagePayload sent = StagePayload::from_value(asr_submit->data);
    check(sent.request.params.at("stream").as_bool(),
          "opt-in request did not ask the ASR stage to stream");
    check(sent.request.inputs.find("asr_partials") == nullptr,
          "the ASR stage received the opt-in hint instead of audio only");
  }

  h.stream(id, "asr", text_chunk("hel", false));
  h.stream(id, "asr", text_chunk("lo", false));
  h.complete(id, "asr", true, text_result("hello"));
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "opt-in ASR leg did not reach the VLM leg");
  if (vlm_submit) {
    StagePayload sent = StagePayload::from_value(vlm_submit->data);
    const Value& messages = *sent.request.inputs.find("messages");
    check(messages.as_array().at(0).find("content")->as_str() == "hello",
          "partial deltas were not replaced by the final transcript");
  }
  h.complete(id + ":vlm", "vlm", true, text_result("answer"));
  check(engine.wait(id, 5.0), "opt-in ASR partial request did not complete");

  const auto events = log.snapshot();
  if (events.size() != 4) {
    check(false, "wrong event count for the ASR partial flow");
    engine.release(id);
    return;
  }
  const UnifiedEvent::Kind expected[] = {
      UnifiedEvent::Kind::kAsrPartial, UnifiedEvent::Kind::kAsrPartial,
      UnifiedEvent::Kind::kAsrFinal, UnifiedEvent::Kind::kComplete};
  for (size_t i = 0; i < events.size(); ++i) {
    check(events[i].seq == static_cast<int64_t>(i),
          "ASR partial sequence is not gap-free");
    check(events[i].kind == expected[i], "ASR partial event out of order");
    check(events[i].request_id == id, "ASR partial leaked the child id");
  }
  check(events[0].text == "hel" && events[1].text == "lo" &&
            !events[0].is_final && !events[1].is_final,
        "ASR partial events do not carry the delta text");
  check(events[2].is_final && events[2].text == "hello",
        "ASR final does not replace the partial text");
  engine.release(id);
}

// Default request: the ASR leg is not asked to stream and no partial is emitted.
void test_asr_partials_default_off() {
  Harness h("partialsoff");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(audio_leg_request(false));
  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "default ASR request was not submitted");
  if (asr_submit) {
    StagePayload sent = StagePayload::from_value(asr_submit->data);
    check(!sent.request.params.at("stream").as_bool(),
          "a default request asked the ASR stage to stream");
  }
  h.complete(id, "asr", true, text_result("hello"));
  check(h.next_submit(h.vlm_recv).has_value(),
        "default ASR leg did not reach the VLM leg");
  h.complete(id + ":vlm", "vlm", true, text_result("answer"));
  check(engine.wait(id, 5.0), "default request did not complete");
  for (const auto& event : log.snapshot()) {
    check(event.kind != UnifiedEvent::Kind::kAsrPartial,
          "a default request emitted an ASR partial");
  }
  engine.release(id);
}

// Cancel after a partial: exactly one cancelled terminal, no VLM submit.
void test_asr_partials_cancel() {
  Harness h("partialscancel");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };
  engine.on_terminal = [&](const UnifiedEvent& event) {
    log.finish(event);
  };

  const std::string id = engine.submit(audio_leg_request(true));
  check(h.next_submit(h.asr_recv).has_value(), "ASR leg was not submitted");
  h.stream(id, "asr", text_chunk("par", false));
  wait_until([&] { return log.size() == 1; },
             "the ASR partial event was not delivered before the cancel");
  check(engine.cancel(id), "cancel was rejected before the ASR leg finished");
  check(h.wait_abort(id), "the ASR stage did not observe the abort");
  h.complete(id, "asr", false, Value{}, std::string("request aborted"));
  check(engine.wait(id, 5.0), "cancelled request did not settle");
  h.no_submit(h.vlm_recv);

  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kCancelled,
        "cancel after a partial did not produce a cancelled terminal");
  check(log.terminal_count(id) == 1,
        "cancel after a partial produced more than one terminal");
  for (const auto& event : log.snapshot()) {
    check(event.kind != UnifiedEvent::Kind::kVlmDelta,
          "a cancelled partial request emitted a VLM delta");
  }
  check(engine.in_flight() == 0, "cancelled request stayed in flight");
  engine.release(id);
}

// An ASR stream failure after a partial: exactly one error terminal.
void test_asr_partials_error() {
  Harness h("partialserror");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  const std::string id = engine.submit(audio_leg_request(true));
  check(h.next_submit(h.asr_recv).has_value(), "ASR leg was not submitted");
  h.stream(id, "asr", text_chunk("par", false));
  h.complete(id, "asr", false, Value{}, std::string("backend reset"));
  check(engine.wait(id, 5.0), "failed partial request did not settle");

  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kError &&
            terminal->error == "backend reset",
        "ASR stream failure did not produce the error terminal");
  check(log.terminal_count(id) == 1,
        "ASR stream failure produced more than one terminal");
  check(h.no_submit(h.vlm_recv), "a failed ASR leg started the VLM leg");
  engine.release(id);
}

// A non-streaming ASR stage emits only a completion; the engine synthesizes
// the ASR final from its result text.
void test_nonstream_asr_synthesizes_final() {
  Harness h("nonstream");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
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
  h.start_pump();
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
    h.start_pump();
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
    h.start_pump();
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
  h.start_pump();
  check(!engine.wait("unknown", 0.0), "wait() reported success for an unknown id");
  check(!engine.info("unknown"), "info() returned state for an unknown id");
}

// cancel() broadcasts an abort to both stages and settles the request.
void test_cancel_propagates_to_both_stages() {
  Harness h("cancel");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
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
  h.start_pump();
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
  h.start_pump();
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
    h.start_pump();
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
    h.start_pump();
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
  h.start_pump();

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
  h.start_pump();
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

// Image + audio: the ASR leg receives PCM only, the VLM leg receives the
// original image part plus the final transcript as a text part. Video stays
// rejected.
void test_joint_image_audio() {
  Harness h("joint");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string id = engine.submit(joint_request());
  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "ASR stage received no submit");
  if (asr_submit) {
    StagePayload payload = StagePayload::from_value(asr_submit->data);
    check(payload.request.inputs.find("audio") != nullptr &&
              payload.request.inputs.find("messages") == nullptr,
          "the ASR leg did not receive PCM only");
  }
  h.complete(id, "asr", true, text_result("what is on screen"));
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "VLM stage received no submit");
  if (vlm_submit) {
    StagePayload payload = StagePayload::from_value(vlm_submit->data);
    const Value& content = *payload.request.inputs.find("messages")
                                ->as_array()
                                .at(0)
                                .find("content");
    check(content.is_array() && content.as_array().size() == 2,
          "the VLM prompt is not image part + transcript text part");
    if (content.is_array() && content.as_array().size() == 2) {
      const Value& image = content.as_array()[0];
      const Value& text = content.as_array()[1];
      check(image.find("type")->as_str() == "image_url" &&
                image.find("image_url")->find("url")->as_str() ==
                    "data:image/png;base64,AA",
            "the VLM lost the original image part");
      check(text.find("type")->as_str() == "text" &&
                text.find("text")->as_str() == "what is on screen",
            "the VLM lost the ASR transcript");
    }
    const Value& modalities = *payload.request.inputs.find("modalities");
    check(modalities.as_array().size() == 2 &&
              modalities.as_array()[0].as_str() == "text" &&
              modalities.as_array()[1].as_str() == "image",
          "the VLM did not declare both text and image modalities");
  }
  h.complete(id + ":vlm", "vlm", true, text_result("a settings screen"));
  check(engine.wait(id, 5.0), "joint request did not complete");
  const auto events = log.snapshot();
  check(!events.empty() &&
            events.front().kind == UnifiedEvent::Kind::kAsrFinal &&
            events.back().kind == UnifiedEvent::Kind::kComplete,
        "joint event order is not asr_final .. complete");
  engine.release(id);
}

// Malformed media contracts are rejected before any stage sees the work.
void test_media_contract_errors() {
  Harness h("mediaerr");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});

  auto rejects = [&](StagePayload payload, const char* message) {
    bool rejected = false;
    try {
      engine.submit(std::move(payload));
    } catch (const ProtocolError&) {
      rejected = true;
    }
    check(rejected, message);
  };

  rejects(audio_request("video"), "a video modality was accepted");
  rejects(image_part_request(""), "an empty image URL was accepted");

  StagePayload no_url = image_part_request("data:image/png;base64,AA");
  no_url.request.inputs =
      Value(Value::Map{{"messages", Value::Array{Value::Map{
                          {"role", Value("user")},
                          {"content", Value::Array{part("image_url")}}}}}});
  rejects(std::move(no_url), "an image part without image_url.url was accepted");

  StagePayload two_images;
  Value::Map message;
  message["role"] = Value("user");
  message["content"] = Value::Array{image_part(), image_part(), Value(part("input_audio"))};
  two_images.request.inputs = Value::Map{{"messages", Value::Array{Value(std::move(message))}}};
  rejects(std::move(two_images), "a request with two image parts was accepted");

  StagePayload empty;
  rejects(std::move(empty), "a request without messages was accepted");
}

// Two sequential requests in one session. session_id is a grouping label only:
// ids, sequences and terminals stay per request, and neither request inherits
// the other's transcript or VLM answer.
void test_session_reuse() {
  Harness h("session");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  auto run = [&](const std::string& transcript, const std::string& answer) {
    const std::string id = engine.submit(joint_request("session-1"));
    auto asr = h.next_submit(h.asr_recv);
    check(asr && asr->request_id == id, "ASR leg id does not match the request");
    h.complete(id, "asr", true, text_result(transcript));
    auto vlm = h.next_submit(h.vlm_recv);
    check(vlm && vlm->request_id == id + ":vlm", "VLM leg id is not a child id");
    if (vlm) {
      // Keep the payload alive: content points into it.
      const StagePayload payload = StagePayload::from_value(vlm->data);
      const Value& content = *payload.request.inputs.find("messages")
                                  ->as_array()
                                  .at(0)
                                  .find("content");
      check(content.is_array() && content.as_array().size() == 2 &&
                content.as_array()[1].find("text")->as_str() == transcript,
            "the VLM leg did not receive this request's transcript");
    }
    h.stream(id + ":vlm", "vlm", text_chunk(answer, false));
    h.complete(id + ":vlm", "vlm", true, text_result(answer));
    check(engine.wait(id, 5.0), "session request did not complete");
    return id;
  };

  const std::string first = run("first transcript", "first answer");
  const std::string second = run("second transcript", "second answer");
  check(first != second, "two session requests shared a request id");

  for (const std::string& id : {first, second}) {
    int64_t next_seq = 0;
    int terminals = 0;
    for (const auto& event : log.snapshot()) {
      if (event.request_id != id) continue;
      check(event.session_id == "session-1", "event lost the session label");
      check(event.seq == next_seq++, "event sequence is not gap-free per request");
      if (event.is_final) ++terminals;
    }
    check(next_seq == 3, "unexpected event count for one session request");
    check(terminals == 2, "a session request did not emit asr_final and complete");
  }
  engine.release(first);
  engine.release(second);
}

// Concurrent requests in the same session interleave. Each event keeps its own
// request id and sequence; no delta or terminal crosses over.
void test_session_reuse_concurrent() {
  Harness h("sessionconc");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  engine.on_event = [&](const UnifiedEvent& event) { log.add(event); };

  const std::string a = engine.submit(joint_request("session-2"));
  const std::string b = engine.submit(joint_request("session-2"));
  auto asr_a = h.next_submit(h.asr_recv);
  auto asr_b = h.next_submit(h.asr_recv);
  check(asr_a && asr_b && asr_a->request_id != asr_b->request_id,
        "concurrent session requests did not reach the ASR stage separately");

  h.complete(a, "asr", true, text_result("alpha"));
  h.complete(b, "asr", true, text_result("beta"));
  auto vlm_a = h.next_submit(h.vlm_recv);
  auto vlm_b = h.next_submit(h.vlm_recv);
  check(vlm_a && vlm_b, "both concurrent VLM legs did not start");

  h.stream(a + ":vlm", "vlm", text_chunk("A", false));
  h.stream(b + ":vlm", "vlm", text_chunk("B", false));
  h.complete(b + ":vlm", "vlm", true, text_result("B"));
  h.complete(a + ":vlm", "vlm", true, text_result("A"));
  check(engine.wait(a, 5.0) && engine.wait(b, 5.0),
        "concurrent session requests did not settle");

  for (const std::string& id : {a, b}) {
    const std::string expected = id == a ? "A" : "B";
    int64_t next_seq = 0;
    std::string delta, terminal;
    for (const auto& event : log.snapshot()) {
      if (event.request_id != id) continue;
      check(event.session_id == "session-2", "event lost the session label");
      check(event.seq == next_seq++, "concurrent event sequence is not per request");
      if (event.kind == UnifiedEvent::Kind::kVlmDelta) delta = event.text;
      if (event.kind == UnifiedEvent::Kind::kComplete) terminal = event.text;
    }
    check(delta == expected, "a VLM delta crossed between session requests");
    check(terminal == expected, "a terminal crossed between session requests");
  }
  engine.release(a);
  engine.release(b);
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
  h.start_pump();
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

// An out-of-order VLM chunk takes the forwarding path. The previous sink
// reenters info() on the pump thread, which deadlocks if forwarding holds mu_.
void test_previous_sink_reentry_does_not_deadlock() {
  Harness h("sinkreentry");
  UnifiedEngine* engine_ptr = nullptr;
  std::promise<std::optional<RequestInfo>> reentered;
  auto reentered_future = reentered.get_future();
  h.coordinator.on_stream = [&](const StreamMessage& msg) {
    reentered.set_value(engine_ptr->info(msg.request_id));
  };
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  engine_ptr = &engine;
  h.start_pump();

  const std::string id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.stream(id, "vlm", text_chunk("early", false));
  const auto status = reentered_future.wait_for(std::chrono::seconds(1));
  check(status == std::future_status::ready,
        "previous sink reentry deadlocked against engine mu_");
  if (status != std::future_status::ready) std::abort();
  const auto info = reentered_future.get();
  check(info && info->request_id == id && info->state == "running" &&
            info->current_stage == kAsrStageId,
        "previous sink reentry returned the wrong unified state");
  engine.release(id);
}

// Hold the partial callback while cancel() runs on another thread. The terminal
// must queue behind the already-sequenced partial, including callback reentry.
void test_cancel_cannot_overtake_partial_dispatch() {
  Harness h("partialcancel");
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();
  EventLog log;
  std::promise<void> partial_entered;
  auto partial_entered_future = partial_entered.get_future();
  std::promise<void> release_partial;
  auto release_partial_future = release_partial.get_future().share();
  std::promise<void> partial_done;
  auto partial_done_future = partial_done.get_future();

  const std::string id = engine.submit(audio_request());
  engine.on_event = [&](const UnifiedEvent& event) {
    if (event.kind == UnifiedEvent::Kind::kAsrPartial) {
      partial_entered.set_value();
      release_partial_future.wait();
    }
    log.add(event);
    if (event.kind == UnifiedEvent::Kind::kAsrPartial) partial_done.set_value();
  };
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.stream(id, "asr", text_chunk("partial", false));
  partial_entered_future.wait();
  check(engine.cancel(id), "cancel rejected the partial-dispatch request");
  check(log.size() == 0, "cancel terminal overtook the blocked partial callback");
  release_partial.set_value();
  partial_done_future.wait();
  check(engine.wait(id, 5.0), "cancel during partial dispatch did not settle");
  wait_until([&] { return log.terminal_count(id) == 1; },
             "cancel during partial dispatch did not emit its terminal");

  const auto events = log.snapshot();
  check(events.size() == 2 &&
            events[0].kind == UnifiedEvent::Kind::kAsrPartial &&
            events[0].seq == 0 &&
            events[1].kind == UnifiedEvent::Kind::kCancelled &&
            events[1].seq == 1,
        "single-drainer dispatch did not preserve partial then cancelled order");
  engine.release(id);

  // Same-thread reentry: cancel from inside on_event must only enqueue the
  // terminal. The outer drainer emits it after the partial callback returns.
  EventLog reentrant_log;
  std::string reentrant_id;
  std::promise<void> reentrant_partial_done;
  auto reentrant_partial_future = reentrant_partial_done.get_future();
  engine.on_event = [&](const UnifiedEvent& event) {
    if (event.kind == UnifiedEvent::Kind::kAsrPartial) {
      check(engine.cancel(reentrant_id), "reentrant cancel rejected the request");
    }
    reentrant_log.add(event);
    if (event.kind == UnifiedEvent::Kind::kAsrPartial) {
      reentrant_partial_done.set_value();
    }
  };
  engine.on_terminal = [&](const UnifiedEvent& event) { reentrant_log.finish(event); };
  reentrant_id = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no reentrant submit");
  h.stream(reentrant_id, "asr", text_chunk("reentrant", false));
  reentrant_partial_future.wait();
  check(engine.wait(reentrant_id, 5.0), "reentrant cancel did not settle");
  wait_until([&] { return reentrant_log.terminal_count(reentrant_id) == 1; },
             "reentrant cancel did not emit its terminal");
  const auto reentrant_events = reentrant_log.snapshot();
  check(reentrant_events.size() == 2 &&
            reentrant_events[0].kind == UnifiedEvent::Kind::kAsrPartial &&
            reentrant_events[1].kind == UnifiedEvent::Kind::kCancelled,
        "reentrant cancel recursively overtook the partial callback");
  engine.release(reentrant_id);
}

// Forwarded callbacks share the unified event order. Cancel either from another
// thread or from inside on_event; neither may complete the old sink before the
// earlier stream callback has returned.
void test_forwarded_sink_cancel_order(bool reentrant) {
  Harness h(reentrant ? "fwdreenter" : "fwdthread");
  std::mutex forwarded_mu;
  std::vector<std::string> forwarded;
  bool stream_active = false;
  bool complete_overlapped_stream = false;
  std::promise<void> stream_entered;
  auto stream_entered_future = stream_entered.get_future();
  std::promise<void> release_stream;
  auto release_stream_future = release_stream.get_future().share();
  h.coordinator.on_stream = [&](const StreamMessage&) {
    {
      std::lock_guard<std::mutex> lock(forwarded_mu);
      forwarded.push_back("stream");
      stream_active = true;
    }
    stream_entered.set_value();
    release_stream_future.wait();
    std::lock_guard<std::mutex> lock(forwarded_mu);
    stream_active = false;
  };
  h.coordinator.on_complete = [&](const RequestInfo&) {
    std::lock_guard<std::mutex> lock(forwarded_mu);
    complete_overlapped_stream |= stream_active;
    forwarded.push_back("complete");
  };
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  h.start_pump();

  EventLog log;
  std::promise<void> partial_entered;
  auto partial_entered_future = partial_entered.get_future();
  std::promise<void> release_partial;
  auto release_partial_future = release_partial.get_future().share();
  const std::string id = engine.submit(audio_request());
  engine.on_event = [&](const UnifiedEvent& event) {
    if (event.kind == UnifiedEvent::Kind::kAsrPartial) {
      if (reentrant) {
        check(engine.cancel(id), "reentrant cancel rejected the forwarding request");
      } else {
        partial_entered.set_value();
        release_partial_future.wait();
      }
    }
    log.add(event);
  };
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no forwarding submit");
  h.stream(id, "asr", text_chunk("partial", false));
  if (!reentrant) {
    partial_entered_future.wait();
    check(engine.cancel(id), "cross-thread cancel rejected the forwarding request");
    release_partial.set_value();
  }
  check(engine.wait(id, 5.0), "cancel during forwarding did not settle");
  stream_entered_future.wait();
  release_stream.set_value();
  wait_until([&] {
    std::lock_guard<std::mutex> lock(forwarded_mu);
    return forwarded.size() == 2 && !stream_active &&
           log.terminal_count(id) == 1;
  }, "forwarded stream/completion or unified terminal did not arrive");

  const auto events = log.snapshot();
  check(events.size() == 2 &&
            events[0].kind == UnifiedEvent::Kind::kAsrPartial &&
            events[0].seq == 0 &&
            events[1].kind == UnifiedEvent::Kind::kCancelled &&
            events[1].seq == 1,
        "forwarding changed unified partial/cancelled order");
  check(log.terminal_count(id) == 1, "forwarding produced a duplicate terminal");

  std::vector<std::string> order;
  bool overlapped = false;
  {
    std::lock_guard<std::mutex> lock(forwarded_mu);
    order = forwarded;
    overlapped = complete_overlapped_stream;
  }
  check(!overlapped, "forwarded completion reentered the active stream sink");
  std::cerr << (reentrant ? "FORWARD_ORDER_REENTRANT=" : "FORWARD_ORDER_THREAD=");
  for (const auto& call : order) std::cerr << call << ",";
  std::cerr << "\n";
  check(order == std::vector<std::string>{"stream", "complete"},
        "forwarded completion overtook the earlier stream callback");
  engine.release(id);
}

// A previous sink blocked on the pump proves run() is active. Construction must
// reject the exchange without modifying that sink instead of racing it.
void test_construction_after_pump_start_rejected() {
  Harness h("lateconstruct");
  std::promise<void> entered;
  auto entered_future = entered.get_future();
  std::promise<void> release;
  auto release_future = release.get_future().share();
  std::promise<void> previous_done;
  auto previous_done_future = previous_done.get_future();
  h.coordinator.on_stream = [&](const StreamMessage&) {
    entered.set_value();
    release_future.wait();
    previous_done.set_value();
  };
  h.start_pump();

  const std::string id = h.coordinator.submit(audio_request(), "asr");
  check(h.next_submit(h.asr_recv).has_value(), "ASR stage received no submit");
  h.stream(id, "asr", text_chunk("partial", false));
  entered_future.wait();

  bool rejected = false;
  try {
    UnifiedEngine late(h.coordinator, UnifiedConfig{});
  } catch (const ProtocolError&) {
    rejected = true;
  }
  check(rejected, "construction exchanged sinks after the pump had started");
  release.set_value();
  previous_done_future.wait();
}

// Promises, not sleeps, fix the interleaving. A sink copied before destruction
// must also remain safe when invoked after destruction.
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
  h.start_pump();
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

struct RetiredSinkProbe {
  std::mutex mu;
  std::vector<std::string> calls;
  bool stream_active = false;
  bool overlapped = false;
  bool nested = false;
  std::thread::id stream_thread;
  bool throw_after_stream = false;
  bool throw_after_complete = false;
  int stream_calls = 0;
  std::function<void()> stream_action;
  std::promise<void> stream_entered;
  std::future<void> stream_entered_future = stream_entered.get_future();
  std::promise<void> release_stream;
  std::shared_future<void> release_stream_future = release_stream.get_future().share();

  void stream(const StreamMessage&) {
    bool first = false;
    {
      std::lock_guard<std::mutex> lock(mu);
      calls.push_back("stream");
      stream_active = true;
      stream_thread = std::this_thread::get_id();
      first = ++stream_calls == 1;
    }
    if (first) stream_entered.set_value();
    if (stream_action) stream_action();
    release_stream_future.wait();
    {
      std::lock_guard<std::mutex> lock(mu);
      stream_active = false;
    }
    if (throw_after_stream) throw std::runtime_error("previous stream failed");
  }

  void complete(const RequestInfo&) {
    {
      std::lock_guard<std::mutex> lock(mu);
      if (stream_active) {
        overlapped = true;
        if (std::this_thread::get_id() == stream_thread) nested = true;
      }
      calls.push_back("complete");
    }
    if (throw_after_complete) throw std::runtime_error("previous complete failed");
  }

  size_t call_count() {
    std::lock_guard<std::mutex> lock(mu);
    return calls.size();
  }

  void release() { release_stream.set_value(); }

  void check_order(const char* tag) {
    check_order_with_exceptions(tag, {"stream", "complete"});
  }

  void check_order_with_exceptions(
      const char* tag, const std::vector<std::string>& expected) {
    std::vector<std::string> order;
    bool saw_overlap = false;
    bool saw_nested = false;
    {
      std::lock_guard<std::mutex> lock(mu);
      order = calls;
      saw_overlap = overlapped;
      saw_nested = nested;
    }
    std::cerr << tag << "=";
    for (const auto& call : order) std::cerr << call << ",";
    std::cerr << " OVERLAP=" << saw_overlap << " NESTED=" << saw_nested << "\n";
    check(order == expected, "forwarded callback order or drain is wrong");
    check(!saw_overlap, "forwarded completion overlapped the active stream callback");
    check(!saw_nested, "forwarded completion nested inside the stream callback");
  }
};

StreamMessage retired_stream_message() {
  StreamMessage msg;
  msg.request_id = "retired-stream";
  msg.from_stage = "asr";
  msg.chunk = text_chunk("after", false);
  return msg;
}

RequestInfo retired_complete_info() {
  RequestInfo info;
  info.request_id = "retired-complete";
  info.state = "aborted";
  return info;
}

// The copied completion arrives after retire() has nulled the engine but while
// the active previous_stream callback is still blocked.
void test_retire_wait_fallback_order() {
  RetiredSinkProbe probe;
  Harness h("retirewait");
  h.coordinator.on_stream = [&](const StreamMessage& msg) { probe.stream(msg); };
  h.coordinator.on_complete = [&](const RequestInfo& info) { probe.complete(info); };
  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  h.start_pump();
  const std::string id = engine->submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(), "retire-wait ASR submit missing");
  h.stream(id, "asr", text_chunk("partial", false));
  probe.stream_entered_future.wait();
  const auto copied_complete = h.coordinator.on_complete;
  UnifiedEngine* engine_ptr = engine.get();

  std::promise<void> destroyed;
  auto destroyed_future = destroyed.get_future();
  std::thread destroyer([&] {
    engine.reset();
    destroyed.set_value();
  });
  bool retiring = false;
  wait_until([&] {
    retiring = unified_retirement_waiting_for_test(*engine_ptr);
    return retiring;
  }, "retire did not wait for the active forwarded stream");
  if (!retiring) {
    probe.release();
    destroyed_future.wait();
    destroyer.join();
    return;
  }

  std::promise<void> complete_returned;
  auto complete_returned_future = complete_returned.get_future();
  std::thread fallback([&] {
    copied_complete(retired_complete_info());
    complete_returned.set_value();
  });
  complete_returned_future.wait();
  check(probe.call_count() == 1,
        "fallback completion ran before the active stream returned");
  probe.release();
  destroyed_future.wait();
  fallback.join();
  destroyer.join();
  probe.check_order("RETIRE_WAIT_ORDER");
}

// Two copied sinks race after destruction. The completion caller may return
// after enqueueing, but its old sink must not run until stream is released.
void test_retired_cross_thread_order() {
  RetiredSinkProbe probe;
  Harness h("retiredthreads");
  h.coordinator.on_stream = [&](const StreamMessage& msg) { probe.stream(msg); };
  h.coordinator.on_complete = [&](const RequestInfo& info) { probe.complete(info); };
  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  const auto copied_stream = h.coordinator.on_stream;
  const auto copied_complete = h.coordinator.on_complete;
  engine.reset();

  std::thread stream_thread([&] { copied_stream(retired_stream_message()); });
  probe.stream_entered_future.wait();
  std::promise<void> complete_returned;
  auto complete_returned_future = complete_returned.get_future();
  std::thread complete_thread([&] {
    copied_complete(retired_complete_info());
    complete_returned.set_value();
  });
  complete_returned_future.wait();
  check(probe.call_count() == 1,
        "post-retirement completion overlapped the blocked stream");
  probe.release();
  stream_thread.join();
  complete_thread.join();
  probe.check_order("RETIRED_THREAD_ORDER");
}

// A retired stream sink reenters completion on the same thread. Reentry must
// enqueue behind the active callback rather than recursively invoke the old sink.
void test_retired_same_thread_reentry() {
  RetiredSinkProbe probe;
  Harness h("retiredreenter");
  h.coordinator.on_stream = [&](const StreamMessage& msg) { probe.stream(msg); };
  h.coordinator.on_complete = [&](const RequestInfo& info) { probe.complete(info); };
  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  const auto copied_stream = h.coordinator.on_stream;
  const auto copied_complete = h.coordinator.on_complete;
  engine.reset();

  probe.release();
  probe.stream_action = [&] {
    copied_complete(retired_complete_info());
    check(probe.call_count() == 1,
          "retired completion recursively entered the old sink");
  };
  copied_stream(retired_stream_message());
  probe.check_order("RETIRED_REENTRANT_ORDER");
}

// A copied sink reports an exception only after every callback queued ahead of
// the error has drained. A later copied sink must still be usable.
void test_retired_reentry_exception_drains() {
  RetiredSinkProbe probe;
  Harness h("retiredthrow");
  h.coordinator.on_stream = [&](const StreamMessage& msg) { probe.stream(msg); };
  h.coordinator.on_complete = [&](const RequestInfo& info) { probe.complete(info); };
  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  const auto copied_stream = h.coordinator.on_stream;
  const auto copied_complete = h.coordinator.on_complete;
  engine.reset();

  probe.release();
  probe.throw_after_stream = true;
  probe.stream_action = [&] { copied_complete(retired_complete_info()); };
  bool caught = false;
  try {
    copied_stream(retired_stream_message());
  } catch (const std::runtime_error&) {
    caught = true;
  }
  check(caught, "retired copied sink did not report the previous sink exception");
  probe.check_order_with_exceptions("RETIRED_EXCEPTION_FIRST_DRAIN",
                                    {"stream", "complete"});

  probe.stream_action = {};
  probe.throw_after_stream = false;
  copied_stream(retired_stream_message());
  copied_complete(retired_complete_info());
  probe.check_order_with_exceptions("RETIRED_EXCEPTION_NEXT_SINK",
                                    {"stream", "complete", "stream", "complete"});
}

// The active engine drain has the same reentry contract before retirement.
void test_active_reentry_exception_drains() {
  RetiredSinkProbe probe;
  Harness h("activethrow");
  h.coordinator.on_stream = [&](const StreamMessage& msg) { probe.stream(msg); };
  h.coordinator.on_complete = [&](const RequestInfo& info) { probe.complete(info); };
  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  const auto copied_stream = h.coordinator.on_stream;
  const auto copied_complete = h.coordinator.on_complete;

  probe.release();
  probe.throw_after_stream = true;
  probe.stream_action = [&] { copied_complete(retired_complete_info()); };
  bool caught = false;
  try {
    copied_stream(retired_stream_message());
  } catch (const std::runtime_error&) {
    caught = true;
  }
  check(caught, "active copied sink did not report the previous sink exception");
  probe.check_order_with_exceptions("ACTIVE_EXCEPTION_FIRST_DRAIN",
                                    {"stream", "complete"});

  probe.stream_action = {};
  probe.throw_after_stream = false;
  copied_stream(retired_stream_message());
  copied_complete(retired_complete_info());
  probe.check_order_with_exceptions("ACTIVE_EXCEPTION_NEXT_SINK",
                                    {"stream", "complete", "stream", "complete"});
}

// retire() is a destructor boundary: old sink exceptions are suppressed there,
// but the queued completion still runs and later copied sinks remain callable.
void test_destructor_suppresses_previous_sink_exception() {
  RetiredSinkProbe probe;
  Harness h("retirethrow");
  h.coordinator.on_stream = [&](const StreamMessage& msg) { probe.stream(msg); };
  h.coordinator.on_complete = [&](const RequestInfo& info) { probe.complete(info); };
  auto engine = std::make_unique<UnifiedEngine>(h.coordinator, UnifiedConfig{});
  const auto copied_stream = h.coordinator.on_stream;
  const auto copied_complete = h.coordinator.on_complete;

  std::thread stream_thread([&] { copied_stream(retired_stream_message()); });
  probe.stream_entered_future.wait();
  UnifiedEngine* engine_ptr = engine.get();
  std::thread destroyer([&] { engine.reset(); });
  wait_until([&] { return unified_retirement_waiting_for_test(*engine_ptr); },
             "retire did not wait for the blocked previous stream");

  std::promise<void> complete_returned;
  auto complete_returned_future = complete_returned.get_future();
  std::thread complete_thread([&] {
    copied_complete(retired_complete_info());
    complete_returned.set_value();
  });
  // The stream is still blocked, so this return proves the completion is in the
  // fallback queue and cannot run until retire() drains it.
  complete_returned_future.wait();
  probe.throw_after_complete = true;
  probe.release();
  stream_thread.join();
  complete_thread.join();
  destroyer.join();
  probe.check_order("DESTRUCTOR_EXCEPTION_DRAIN");

  probe.throw_after_complete = false;
  copied_stream(retired_stream_message());
  copied_complete(retired_complete_info());
  probe.check_order_with_exceptions("DESTRUCTOR_EXCEPTION_NEXT_SINK",
                                    {"stream", "complete", "stream", "complete"});
}

// Real HTTP/SSE and Coordinator, with the same fake stage sockets as the other
// contract tests. This checks public wiring only; it is not a model result.
void test_public_http_unified_sse() {
  Harness h("publichttp");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "public unified server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  const std::string pcm(800, '\0');
  const std::string wav = audio::wav_from_pcm16(pcm);
  nlohmann::json audio_part = {
      {"type", "input_audio"},
      {"input_audio", {{"data", client::detail::base64_encode(wav)}, {"format", "wav"}}}};
  nlohmann::json message = {{"role", "user"},
                            {"content", nlohmann::json::array({audio_part})}};
  nlohmann::json body = {{"model", "qwen3-unified"},
                         {"messages", nlohmann::json::array({message})},
                         {"stream", true},
                         {"max_tokens", 37},
                         {"temperature", 0.25}};
  auto response_future = std::async(std::launch::async, [&] {
    client::detail::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{10000};
    return client::detail::perform(request);
  });

  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "public HTTP did not submit the ASR leg");
  if (asr_submit) {
    StagePayload payload = StagePayload::from_value(asr_submit->data);
    check(payload.request.inputs.as_map().size() == 1 &&
              payload.request.inputs.find("audio")->as_bin() == pcm,
          "public WAV was not converted to the ASR PCM-only contract");
    check(!payload.request.params.at("stream").as_bool() &&
              payload.request.params.at("temperature").as_double() == 0.0 &&
              payload.request.params.at("max_new_tokens").as_int() == 37,
          "ASR parameters were not fixed without losing max_new_tokens");
    h.complete(asr_submit->request_id, "asr", true, text_result("real transcript"));
  }

  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "public HTTP did not submit the VLM leg");
  if (vlm_submit) {
    StagePayload payload = StagePayload::from_value(vlm_submit->data);
    const Value& messages = *payload.request.inputs.find("messages");
    check(messages.as_array().at(0).find("content")->as_str() == "real transcript" &&
              payload.request.inputs.find("audio") == nullptr,
          "VLM did not receive exactly the ASR transcript");
    check(payload.request.params.at("stream").as_bool() &&
              payload.request.params.at("temperature").as_double() == 0.25 &&
              payload.request.params.at("max_new_tokens").as_int() == 37,
          "VLM did not preserve the public sampling parameters");
    h.stream(vlm_submit->request_id, "vlm", text_chunk("answer", false));
    auto result = text_result("answer");
    result["finish_reason"] = Value("stop");
    h.complete(vlm_submit->request_id, "vlm", true, Value(std::move(result)));
  }

  const auto response = response_future.get();
  api.stop();
  server_thread.join();
  check(response.status == 200, "public unified SSE returned an HTTP error");
  const size_t asr_pos = response.body.find("\"kind\":\"asr_final\"");
  const size_t vlm_pos = response.body.find("\"kind\":\"vlm_delta\"");
  const size_t complete_pos = response.body.find("\"kind\":\"complete\"");
  check(asr_pos != std::string::npos && vlm_pos != std::string::npos &&
            complete_pos != std::string::npos && asr_pos < vlm_pos &&
            vlm_pos < complete_pos,
        "public SSE order is not asr_final, vlm_delta, complete");
  check(response.body.find("\"text\":\"real transcript\"") != std::string::npos &&
            response.body.find("\"content\":\"answer\"") != std::string::npos,
        "public SSE lost the transcript or answer");
  check(response.body.find("\"kind\":\"complete\"", complete_pos + 1) ==
            std::string::npos,
        "public SSE emitted more than one successful terminal");
  wait_until([&] { return engine.in_flight() == 0; },
             "public unified request stayed in flight after completion");
}

// Public joint request: one instruction, one image, one WAV. The ASR leg sees
// PCM only; the VLM leg sees instruction, image, transcript in that order and
// every SSE unified event carries the public session label.
void test_public_http_joint_input() {
  Harness h("publicjoint");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "public joint server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  const std::string pcm(800, '\0');
  const std::string wav = audio::wav_from_pcm16(pcm);
  nlohmann::json audio_part = {
      {"type", "input_audio"},
      {"input_audio", {{"data", client::detail::base64_encode(wav)}, {"format", "wav"}}}};
  nlohmann::json instruction = {{"type", "text"}, {"text", "describe this image"}};
  nlohmann::json image = {{"type", "image_url"},
                          {"image_url", {{"url", "data:image/png;base64,AA"}}}};
  nlohmann::json body = {
      {"model", "qwen3-unified"},
      {"session_id", "public-session"},
      {"messages", nlohmann::json::array({nlohmann::json{
           {"role", "user"},
           {"content", nlohmann::json::array({instruction, image, audio_part})}}})},
      {"stream", true}};
  auto response_future = std::async(std::launch::async, [&] {
    client::detail::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{10000};
    return client::detail::perform(request);
  });

  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "public joint request did not submit the ASR leg");
  if (asr_submit) {
    StagePayload payload = StagePayload::from_value(asr_submit->data);
    check(payload.request.inputs.as_map().size() == 1 &&
              payload.request.inputs.find("audio")->as_bin() == pcm,
          "the ASR leg received more than PCM");
    h.complete(asr_submit->request_id, "asr", true, text_result("spoken"));
  }
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "public joint request did not submit the VLM leg");
  if (vlm_submit) {
    StagePayload payload = StagePayload::from_value(vlm_submit->data);
    const Value& content = *payload.request.inputs.find("messages")
                                ->as_array()
                                .at(0)
                                .find("content");
    check(content.is_array() && content.as_array().size() == 3,
          "the VLM prompt is not instruction + image + transcript");
    if (content.is_array() && content.as_array().size() == 3) {
      check(content.as_array()[0].find("text")->as_str() == "describe this image" &&
                content.as_array()[1].find("type")->as_str() == "image_url" &&
                content.as_array()[2].find("text")->as_str() == "spoken",
            "the VLM part order or content is wrong");
    }
    h.complete(vlm_submit->request_id, "vlm", true, text_result("answer"));
  }
  const auto response = response_future.get();
  api.stop();
  server_thread.join();
  check(response.status == 200, "public joint SSE returned an HTTP error");
  check(response.body.find("\"session_id\":\"public-session\"") !=
            std::string::npos,
        "public SSE lost the session label");
}

// Bad unified media fields are rejected before any stage sees the work.
void test_public_http_unified_errors() {
  Harness h("publicbad");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "public error-path server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  const std::string wav = audio::wav_from_pcm16(std::string(800, '\0'));
  const nlohmann::json audio_part = {
      {"type", "input_audio"},
      {"input_audio", {{"data", client::detail::base64_encode(wav)}, {"format", "wav"}}}};
  const nlohmann::json image = {{"type", "image_url"},
                                {"image_url", {{"url", "data:image/png;base64,AA"}}}};
  // The public client throws on a non-2xx response; a 400 is the expected result.
  auto rejected = [&](nlohmann::json body) {
    body["model"] = "qwen3-unified";
    body["stream"] = false;
    client::detail::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                  "/v1/chat/completions";
    request.body = body.dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{5000};
    try {
      client::detail::perform(request);
    } catch (const client::Error& exc) {
      return std::string(exc.what()).find("HTTP 400") != std::string::npos;
    }
    return false;
  };
  auto body_with = [](nlohmann::json content) {
    return nlohmann::json{{"messages", nlohmann::json::array({nlohmann::json{
                               {"role", "user"}, {"content", std::move(content)}}})}};
  };

  auto two_images = body_with(nlohmann::json::array({image, image, audio_part}));
  check(rejected(two_images), "two image parts were accepted");
  auto two_texts = body_with(nlohmann::json::array(
      {nlohmann::json{{"type", "text"}, {"text", "a"}},
       nlohmann::json{{"type", "text"}, {"text", "b"}}, image, audio_part}));
  check(rejected(two_texts), "two text parts were accepted");
  auto empty_text = body_with(nlohmann::json::array(
      {nlohmann::json{{"type", "text"}, {"text", ""}}, image, audio_part}));
  check(rejected(empty_text), "an empty text part was accepted");
  auto empty_session = body_with(nlohmann::json::array({image, audio_part}));
  empty_session["session_id"] = "";
  check(rejected(empty_session), "an empty session_id was accepted");
  check(h.no_submit(h.asr_recv), "an invalid unified request reached the ASR stage");
  api.stop();
  server_thread.join();
}

size_t occurrences(const std::string& text, const std::string& needle) {
  size_t count = 0;
  for (size_t pos = text.find(needle); pos != std::string::npos;
       pos = text.find(needle, pos + needle.size())) {
    ++count;
  }
  return count;
}

// Public chat body with one WAV input_audio part. The fake ASR stage receives
// the decoded PCM; no model runs in this file.
nlohmann::json public_audio_body(int max_tokens = 37) {
  const std::string pcm(800, '\0');
  const std::string wav = audio::wav_from_pcm16(pcm);
  nlohmann::json audio_part = {
      {"type", "input_audio"},
      {"input_audio", {{"data", client::detail::base64_encode(wav)}, {"format", "wav"}}}};
  nlohmann::json message = {{"role", "user"},
                            {"content", nlohmann::json::array({audio_part})}};
  return {{"model", "qwen3-unified"},
          {"messages", nlohmann::json::array({message})},
          {"stream", true},
          {"max_tokens", max_tokens},
          {"temperature", 0.0}};
}

// Public HTTP abort while the VLM leg is already streaming. The deltas that
// were sequenced before the abort arrive, then exactly one cancelled terminal.
// A late delta and a late success completion are dropped, the VLM stage sees the
// abort, in_flight returns to zero, and the next request completes. A
// cancellation is not a generation failure, so the stream carries no error
// frame.
void test_public_http_cancel_during_vlm() {
  Harness h("httpcancelvlm");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "cancel server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  const std::string base =
      "http://127.0.0.1:" + std::to_string(api.bound_port());

  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  std::mutex body_mu;
  std::string body;
  std::string stream_request_id;
  std::promise<void> saw_two_deltas;
  auto saw_two_deltas_future = saw_two_deltas.get_future();
  bool signaled = false;
  auto stream_future = std::async(std::launch::async, [&] {
    client::detail::HttpRequest request;
    request.url = base + "/v1/chat/completions";
    request.body = public_audio_body().dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{10000};
    request.on_body = [&](std::string_view bytes) {
      std::lock_guard<std::mutex> lock(body_mu);
      body.append(bytes);
      if (stream_request_id.empty()) {
        const size_t id_pos = body.find("\"id\":\"");
        if (id_pos != std::string::npos) {
          const size_t begin = id_pos + 6;
          const size_t end = body.find('"', begin);
          if (end != std::string::npos) {
            stream_request_id = body.substr(begin, end - begin);
          }
        }
      }
      if (!signaled && occurrences(body, "\"kind\":\"vlm_delta\"") >= 2) {
        signaled = true;
        saw_two_deltas.set_value();
      }
      return true;
    };
    return client::detail::perform(request);
  });

  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "cancel request did not reach the ASR stage");
  const std::string id = asr_submit ? asr_submit->request_id : std::string();
  h.complete(id, "asr", true, text_result("transcript"));
  auto vlm_submit = h.next_submit(h.vlm_recv);
  check(vlm_submit.has_value(), "cancel request did not reach the VLM stage");
  const std::string child = vlm_submit ? vlm_submit->request_id : id + ":vlm";
  h.stream(child, "vlm", text_chunk("first ", false));
  h.stream(child, "vlm", text_chunk("second", false));

  check(saw_two_deltas_future.wait_for(std::chrono::seconds(5)) ==
            std::future_status::ready,
        "two VLM deltas never reached the public stream");
  check(!stream_request_id.empty() && stream_request_id == id,
        "the public stream id did not match the submitted request");

  client::detail::HttpRequest abort_request;
  abort_request.url = base + "/abort";
  abort_request.body =
      nlohmann::json{{"request_id", stream_request_id.empty() ? id
                                                              : stream_request_id}}
          .dump();
  abort_request.headers.emplace("Content-Type", "application/json");
  abort_request.timeout = client::Milliseconds{5000};
  const auto abort_response = client::detail::perform(abort_request);
  check(abort_response.status == 200,
        "public abort did not accept the active request");

  // Tokens already in flight at the real worker can still arrive, and the
  // worker can still report success. Both are stale after the abort.
  h.stream(child, "vlm", text_chunk("late", false));
  h.complete(child, "vlm", true, text_result("late answer"));

  const auto response = stream_future.get();
  api.stop();
  server_thread.join();

  check(response.status == 200, "cancelled SSE returned an HTTP error");
  check(occurrences(body, "\"kind\":\"cancelled\"") == 1,
        "cancel during VLM did not produce exactly one cancelled terminal");
  check(occurrences(body, "\"kind\":\"complete\"") == 0 &&
            occurrences(body, "\"kind\":\"error\"") == 0,
        "cancel during VLM produced a second terminal");
  check(occurrences(body, "\"kind\":\"vlm_delta\"") == 2,
        "a VLM delta delivered after the abort reached the public stream");
  check(body.find("late") == std::string::npos,
        "late VLM text reached the public stream");
  check(occurrences(body, "[DONE]") == 1,
        "the cancelled stream did not end with exactly one [DONE]");
  const size_t cancelled_pos = body.find("\"kind\":\"cancelled\"");
  const size_t last_delta = body.rfind("\"kind\":\"vlm_delta\"");
  check(cancelled_pos != std::string::npos &&
            (last_delta == std::string::npos || last_delta < cancelled_pos),
        "a vlm_delta was written after the cancelled terminal");
  check(occurrences(body, "\"type\":\"generation_error\"") == 0,
        "a client-requested cancel was reported as a generation error");
  check(log.terminal_count(id) == 1,
        "the engine reported more than one terminal for one cancellation");
  wait_until([&] { return engine.in_flight() == 0; },
             "in_flight stayed above zero after a VLM cancellation");
  check(h.wait_abort(child), "the VLM stage never received the abort");

  const std::string next = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(),
        "the request after a cancellation did not reach the ASR stage");
  h.complete(next, "asr", true, text_result("second transcript"));
  check(h.next_submit(h.vlm_recv).has_value(),
        "the request after a cancellation did not reach the VLM stage");
  h.complete(next + ":vlm", "vlm", true, text_result("second answer"));
  check(engine.wait(next, 5.0),
        "the request after a cancellation did not settle");
  const auto follow_up = log.terminal_for(next);
  check(follow_up && follow_up->kind == UnifiedEvent::Kind::kComplete,
        "the request after a cancellation did not complete");
  check(engine.in_flight() == 0,
        "in_flight stayed above zero after the follow-up request");
  engine.release(next);
}

// Public HTTP request whose ASR leg fails in the stage, not at request
// validation. The stream carries one error attributed to the ASR stage, no ASR
// final, no VLM delta, no VLM submit, in_flight returns to zero, and the next
// request completes.
void test_public_http_asr_model_failure() {
  Harness h("httpasrfail");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  UnifiedEngine engine(h.coordinator, UnifiedConfig{});
  api.set_unified_engine(engine, "vlm");
  check(api.bind(), "ASR failure server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });
  const std::string base =
      "http://127.0.0.1:" + std::to_string(api.bound_port());

  EventLog log;
  engine.on_terminal = [&](const UnifiedEvent& event) { log.finish(event); };

  auto response_future = std::async(std::launch::async, [&] {
    client::detail::HttpRequest request;
    request.url = base + "/v1/chat/completions";
    request.body = public_audio_body().dump();
    request.headers.emplace("Content-Type", "application/json");
    request.timeout = client::Milliseconds{10000};
    return client::detail::perform(request);
  });

  auto asr_submit = h.next_submit(h.asr_recv);
  check(asr_submit.has_value(), "ASR failure request did not reach the stage");
  const std::string id = asr_submit ? asr_submit->request_id : std::string();
  h.complete(id, "asr", false, Value{},
             std::string("ASR model backend unavailable"));

  const auto response = response_future.get();
  api.stop();
  server_thread.join();

  check(response.status == 200, "ASR failure SSE returned an HTTP error");
  check(occurrences(response.body, "\"kind\":\"error\"") == 1,
        "the ASR stage failure did not produce exactly one error terminal");
  check(occurrences(response.body, "\"kind\":\"asr_final\"") == 0 &&
            occurrences(response.body, "\"kind\":\"vlm_delta\"") == 0 &&
            occurrences(response.body, "\"kind\":\"complete\"") == 0,
        "the ASR stage failure emitted success or VLM events");
  check(occurrences(response.body, "\"type\":\"generation_error\"") == 1,
        "the ASR stage failure did not produce exactly one error frame");
  check(response.body.find("\"stage\":\"asr\"") != std::string::npos,
        "the error was not attributed to the ASR stage");
  check(response.body.find("ASR model backend unavailable") !=
            std::string::npos,
        "the stage failure reason was not propagated");
  check(occurrences(response.body, "[DONE]") == 1,
        "the failed stream did not end with exactly one [DONE]");
  check(log.terminal_count(id) == 1, "the ASR failure produced duplicate terminals");
  const auto terminal = log.terminal_for(id);
  check(terminal && terminal->kind == UnifiedEvent::Kind::kError &&
            terminal->stage_id == kAsrStageId,
        "the ASR failure terminal did not name the ASR stage");
  check(h.no_submit(h.vlm_recv),
        "the VLM leg started after an ASR stage failure");
  wait_until([&] { return engine.in_flight() == 0; },
             "in_flight stayed above zero after an ASR stage failure");
  engine.release(id);

  const std::string next = engine.submit(audio_request());
  check(h.next_submit(h.asr_recv).has_value(),
        "the request after an ASR failure did not reach the ASR stage");
  h.complete(next, "asr", true, text_result("recovered transcript"));
  check(h.next_submit(h.vlm_recv).has_value(),
        "the request after an ASR failure did not reach the VLM stage");
  h.stream(next + ":vlm", "vlm", text_chunk("recovered", false));
  h.complete(next + ":vlm", "vlm", true, text_result("recovered answer"));
  check(engine.wait(next, 5.0),
        "the request after an ASR failure did not settle");
  const auto follow_up = log.terminal_for(next);
  check(follow_up && follow_up->kind == UnifiedEvent::Kind::kComplete,
        "the request after an ASR failure did not complete");
  check(engine.in_flight() == 0,
        "in_flight stayed above zero after the follow-up request");
  engine.release(next);
}

// Without a unified engine, session_id must be rejected, not accepted and
// dropped. This keeps the server-wide unknown-field contract intact.
void test_public_http_session_id_requires_unified() {
  Harness h("plainsession");
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "qwen3-unified";
  config.request_timeout_s = 5.0;
  OpenAIApi api(h.coordinator, config);
  check(api.bind(), "plain server did not bind");
  h.start_pump();
  std::thread server_thread([&] { api.listen(); });

  nlohmann::json body = {
      {"model", "qwen3-unified"},
      {"session_id", "plain-session"},
      {"messages", nlohmann::json::array({nlohmann::json{
           {"role", "user"}, {"content", "hello"}}})}};
  client::detail::HttpRequest request;
  request.url = "http://127.0.0.1:" + std::to_string(api.bound_port()) +
                "/v1/chat/completions";
  request.body = body.dump();
  request.headers.emplace("Content-Type", "application/json");
  request.timeout = client::Milliseconds{5000};
  bool rejected_with_400 = false;
  try {
    client::detail::perform(request);
  } catch (const client::Error& exc) {
    rejected_with_400 =
        std::string(exc.what()).find("HTTP 400") != std::string::npos &&
        std::string(exc.what()).find("requires unified mode") != std::string::npos;
  }
  check(rejected_with_400,
        "a non-unified request accepted session_id instead of rejecting it");
  check(h.no_submit(h.asr_recv),
        "a non-unified request with session_id reached the ASR stage");
  api.stop();
  server_thread.join();
}

}  // namespace

int main() {
  alarm(120);
  test_streaming_asr_to_vlm();
  test_asr_partials_opt_in();
  test_asr_partials_default_off();
  test_asr_partials_cancel();
  test_asr_partials_error();
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
  test_joint_image_audio();
  test_media_contract_errors();
  test_session_reuse();
  test_session_reuse_concurrent();
  test_cancel_before_pumped_completion();
  test_cancel_after_asr_completed_swallows_vlm();
  test_cancel_handoff_race_stress();
  test_previous_sink_reentry_does_not_deadlock();
  test_cancel_cannot_overtake_partial_dispatch();
  test_forwarded_sink_cancel_order(false);
  test_forwarded_sink_cancel_order(true);
  test_construction_after_pump_start_rejected();
  test_destructor_synchronizes_inflight_callback();
  test_retire_wait_fallback_order();
  test_retired_cross_thread_order();
  test_retired_same_thread_reentry();
  test_active_reentry_exception_drains();
  test_retired_reentry_exception_drains();
  test_destructor_suppresses_previous_sink_exception();
  test_public_http_unified_sse();
  test_public_http_cancel_during_vlm();
  test_public_http_asr_model_failure();
  test_public_http_joint_input();
  test_public_http_unified_errors();
  test_public_http_session_id_requires_unified();
  return g_failures == 0 ? 0 : 1;
}
