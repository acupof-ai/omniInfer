// SPDX-License-Identifier: Apache-2.0
// MessagePack control-plane messages and payload schemas.
//
// The wire contract is msgpack maps with a "type" discriminator. Field names,
// optional-field rules and validation order follow the Python source
// (proto/messages.py, proto/request.py, proto/data_ref.py, proto/admin.py).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "omniinfer/proto/wire.hpp"

namespace omniinfer {

// ---- data_ref.py ---------------------------------------------------------

enum class TransportKind { LocalObject, CudaIpc, Shm, Mooncake };
enum class DataKind {
  StagePayload,
  StreamChunk,
  StreamMetadataTensor,
  KvPages,
  WeightBucket,
  MoeExpertPayload,
};
enum class DataLayout { PackedTensors, RawTensor, Paged, Bucketed, Scatter };

const char* to_string(TransportKind k);
const char* to_string(DataKind k);
const char* to_string(DataLayout k);
TransportKind transport_from_string(const std::string& s);
DataKind data_kind_from_string(const std::string& s);
DataLayout data_layout_from_string(const std::string& s);

struct TensorMeta {
  std::string path;
  std::vector<int64_t> shape;
  std::string dtype;
  std::string device;
  int64_t offset = 0;
  int64_t size = 0;

  Value to_value() const;
  static TensorMeta from_value(const Value& v);
};

struct BackendRef {
  TransportKind transport = TransportKind::Shm;
  Value info;  // relay-specific handle map
  int64_t length = 0;

  Value to_value() const;
  static BackendRef from_value(const Value& v);
};

struct DataRef;
struct MetadataTensorRef {
  std::string path;
  std::shared_ptr<DataRef> ref_obj;
  Value to_value() const;
};

struct DataRef {
  int64_t version = 1;
  DataKind kind = DataKind::StagePayload;
  std::string object_id;
  TransportKind transport = TransportKind::Shm;
  DataLayout layout = DataLayout::PackedTensors;
  BackendRef buffer;
  std::optional<std::string> header;
  std::vector<TensorMeta> tensors;
  std::optional<std::vector<int64_t>> shape;
  std::optional<std::string> dtype;
  std::optional<std::string> device;
  std::optional<int64_t> offset;
  std::optional<Value> metadata;
  std::vector<MetadataTensorRef> metadata_tensors;

  Value to_value() const;
  static DataRef from_value(const Value& v);
};

// ---- request.py ----------------------------------------------------------

struct OmniRequest {
  Value inputs;
  Value::Map params;
  Value::Map metadata;

  Value to_value() const;
  static OmniRequest from_value(const Value& v);
};

struct StagePayload {
  std::string request_id;
  OmniRequest request;
  Value data;

  Value to_value() const;
  static StagePayload from_value(const Value& v);
};

struct RoutedPayloads {
  std::map<std::string, StagePayload> payloads;
  bool keep_request_state = false;
};

// ---- admin.py ------------------------------------------------------------

inline constexpr const char* kAdminModelInfo = "model_info";
inline constexpr const char* kAdminPauseGeneration = "pause_generation";
inline constexpr const char* kAdminContinueGeneration = "continue_generation";
inline constexpr const char* kAdminUpdateWeightsFromDisk = "update_weights_from_disk";
inline constexpr const char* kAdminUpdateWeightsFromTensor = "update_weights_from_tensor";
inline constexpr const char* kAdminUpdateWeightsFromDistributed =
    "update_weights_from_distributed";
inline constexpr const char* kAdminInitWeightsUpdateGroup = "init_weights_update_group";
inline constexpr const char* kAdminDestroyWeightsUpdateGroup =
    "destroy_weights_update_group";
inline constexpr const char* kAdminWeightsChecker = "weights_checker";

bool is_update_action(const std::string& action);

struct AdminOperation {
  std::string op_id;
  std::string action;
  Value::Map payload;
  std::optional<std::vector<std::string>> target_stages;
  std::optional<double> timeout_s;

  Value to_value() const;
  static AdminOperation from_value(const Value& v);
};

struct AdminResult {
  std::string op_id;
  std::string stage;
  std::string action;
  bool success = false;
  std::string message;
  Value::Map data;
  std::optional<std::string> error;
  std::optional<int64_t> rank;
  std::optional<std::string> role;

  Value to_value() const;
  static AdminResult from_value(const Value& v);
};

// ---- kv_transfer.py ------------------------------------------------------

struct KVBufferSpec {
  std::string name;
  int64_t bytes_per_page = 0;
  Value to_value() const;
  static KVBufferSpec from_value(const Value& v);
};

struct KVPoolLayout {
  std::string layout_id;
  int64_t page_size = 0;
  std::vector<KVBufferSpec> buffers;
  bool compatible_with(const KVPoolLayout& other) const;
  Value to_value() const;
  static KVPoolLayout from_value(const Value& v);
};

struct KVTransferPrepareMessage {
  std::string request_id;
  std::string transfer_id;
  std::string from_stage;
  std::string to_stage;
  std::string source_pool_id;
  std::string target_pool_id;
  std::vector<int64_t> source_page_indices;
  KVPoolLayout source_layout;
  Value::Map metadata;
  Value to_value() const;
  static KVTransferPrepareMessage from_value(const Value& v);
};

struct KVTransferReadyMessage {
  std::string request_id;
  std::string transfer_id;
  std::string from_stage;
  std::string to_stage;
  bool success = false;
  std::optional<std::string> destination_pool_id;
  std::vector<int64_t> destination_page_indices;
  std::optional<Value> destination_ref;
  std::optional<std::string> error;
  Value to_value() const;
  static KVTransferReadyMessage from_value(const Value& v);
};

// ---- messages.py ---------------------------------------------------------

struct DataReadyMessage {
  std::string request_id;
  std::string from_stage;
  std::string to_stage;
  std::optional<Value> data_ref;
  std::optional<int64_t> chunk_id;
  bool is_done = false;
  std::optional<std::string> error;
  std::optional<std::map<std::string, int64_t>> replica_bindings;

  Value to_value() const;
  static DataReadyMessage from_value(const Value& v);
};

struct StreamChunkBatchMessage {
  std::string from_stage;
  std::string to_stage;
  std::vector<std::string> request_ids;
  std::vector<int64_t> chunk_ids;
  std::vector<std::string> dtypes;
  std::vector<std::vector<int64_t>> shapes;
  std::vector<std::string> tensor_bytes;
  std::string pickled_metadata;
  std::map<std::string, std::map<std::string, int64_t>> replica_bindings;

  Value to_value() const;
  static StreamChunkBatchMessage from_value(const Value& v);
};

struct DataAckMessage {
  std::string request_id;
  std::string from_stage;
  std::string to_stage;
  std::string object_id;
  bool success = true;
  std::optional<std::string> error;

  Value to_value() const;
  static DataAckMessage from_value(const Value& v);
};

struct AbortMessage {
  std::string request_id;
  Value to_value() const;
  static AbortMessage from_value(const Value& v);
};

struct CompleteMessage {
  std::string request_id;
  std::string from_stage;
  bool success = false;
  Value result;
  std::optional<std::string> error;

  Value to_value() const;
  static CompleteMessage from_value(const Value& v);
};

struct StreamMessage {
  std::string request_id;
  std::string from_stage;
  Value chunk;
  std::optional<int64_t> stage_id;
  std::optional<std::string> stage_name;
  std::optional<std::string> modality;
  std::optional<int64_t> chunk_id;

  Value to_value() const;
  static StreamMessage from_value(const Value& v);
};

struct SubmitMessage {
  std::string request_id;
  Value data;  // a StagePayload value, or any payload object
  std::optional<std::map<std::string, int64_t>> replica_bindings;

  Value to_value() const;
  static SubmitMessage from_value(const Value& v);
};

struct ShutdownMessage {
  Value to_value() const;
  static ShutdownMessage from_value(const Value& v);
};

struct FreezeGCMessage {
  Value to_value() const;
  static FreezeGCMessage from_value(const Value& v);
};

struct ProfilerStartMessage {
  std::string run_id;
  std::string trace_path_template;
  std::optional<std::string> event_dir;
  bool enable_torch = true;
  std::optional<int64_t> event_buffer_capacity;

  Value to_value() const;
  static ProfilerStartMessage from_value(const Value& v);
};

struct ProfilerStopMessage {
  std::optional<std::string> run_id;
  Value to_value() const;
  static ProfilerStopMessage from_value(const Value& v);
};

struct AdminMessage {
  AdminOperation operation;
  Value to_value() const;
  static AdminMessage from_value(const Value& v);
};

struct AdminResultMessage {
  AdminResult result;
  Value to_value() const;
  static AdminResultMessage from_value(const Value& v);
};

// Every control-plane message the stage and coordinator exchange.
using Message = std::variant<DataReadyMessage, StreamChunkBatchMessage, DataAckMessage,
                             KVTransferPrepareMessage, KVTransferReadyMessage,
                             AbortMessage, CompleteMessage, StreamMessage,
                             SubmitMessage, ShutdownMessage, FreezeGCMessage,
                             ProfilerStartMessage, ProfilerStopMessage, AdminMessage,
                             AdminResultMessage>;

// Dispatch on the "type" discriminator, as parse_message() does.
Message parse_message(const Value& v);
const char* message_type_name(const Message& msg);

std::string serialize_message(const Message& msg);
Message deserialize_message(const std::string& data);

}  // namespace omniinfer
