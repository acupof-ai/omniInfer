// SPDX-License-Identifier: Apache-2.0
// Unit / contract tests for the unified multimodal engine v1.
//
// These tests drive the engine through the real Coordinator over ipc sockets,
// but the ASR and VLM "stages" are bound PULL sockets and a PUSH socket: no
// model worker runs. They verify the ASR -> VLM orchestration contract: event
// order, deterministic aggregation, cancellation, stage failures,
// backpressure, and duplicate/out-of-order handling. They are not real-model
// E2E tests.
#include <chrono>
#include <functional>
#include <iostream>
#include <mutex>
#include <optional>
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
  std::optional<UnifiedEvent> terminal;

  void add(const UnifiedEvent& event) {
    std::lock_guard<std::mutex> lock(mu);
    events.push_back(event);
  }
  void finish(const UnifiedEvent& event) {
    std::lock_guard<std::mutex> lock(mu);
    terminal = event;
  }
  bool has_terminal() {
    std::lock_guard<std::mutex> lock(mu);
    return terminal.has_value();
  }
  std::optional<UnifiedEvent> terminal_event() {
    std::lock_guard<std::mutex> lock(mu);
    return terminal;
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

struct IpcCleanup {
  std::vector<std::string> paths;
  ~IpcCleanup() {
    for (const auto& path : paths) std::remove(path.c_str());
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
  std::string base;
  IpcCleanup cleanup;
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

  explicit Harness(const char* tag)
      : base("/tmp/omni-unified-" + std::string(tag) + "-" +
             std::to_string(getpid()) + "-"),
        cleanup{{base + "asr", base + "vlm", base + "done", base + "abort"}},
        asr_recv("ipc://" + base + "asr", true),
        vlm_recv("ipc://" + base + "vlm", true),
        event_push("ipc://" + base + "done"),
        coordinator("ipc://" + base + "done", "ipc://" + base + "abort") {
    for (const auto& path : cleanup.paths) std::remove(path.c_str());
    asr_recv.start();
    vlm_recv.start();
    coordinator.register_stage("asr", "ipc://" + base + "asr");
    coordinator.register_stage("vlm", "ipc://" + base + "vlm");
    coordinator.start();
    pump = std::thread([this] { coordinator.run(); });
    event_push.connect();
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

  h.stream(id + ":vlm", "vlm", text_chunk("hi", false));
  h.stream(id + ":vlm", "vlm", text_chunk(" there", false));
  h.complete(id + ":vlm", "vlm", true, text_result("hi there"));
  check(engine.wait(id, 5.0), "unified request did not complete");

  const auto events = log.snapshot();
  check(events.size() == 6, "wrong event count for the full ASR -> VLM flow");
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
  check(engine.info(id)->state == "completed", "coordinator state is not completed");
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

}  // namespace

int main() {
  alarm(60);
  test_streaming_asr_to_vlm();
  test_nonstream_asr_synthesizes_final();
  test_cancel_propagates_to_both_stages();
  test_asr_failure();
  test_vlm_failure();
  test_queue_cap_backpressure();
  test_duplicate_final_and_out_of_order();
  test_image_video_rejected();
  return g_failures == 0 ? 0 : 1;
}
