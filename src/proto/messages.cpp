// SPDX-License-Identifier: Apache-2.0
#include "omniinfer/proto/messages.hpp"

#include <memory>
#include <utility>

namespace omniinfer {
namespace {

Value opt_str(const std::optional<std::string>& v) {
  return v ? Value(*v) : Value();
}

Value opt_int(const std::optional<int64_t>& v) {
  return v ? Value(*v) : Value();
}

std::optional<std::string> read_opt_str(const Value& map, const std::string& key) {
  const Value* v = map.find(key);
  if (v == nullptr || v->is_nil()) return std::nullopt;
  if (!v->is_str()) throw ProtocolError(key + " must be str or none");
  return v->as_str();
}

std::optional<int64_t> read_opt_int(const Value& map, const std::string& key) {
  const Value* v = map.find(key);
  if (v == nullptr || v->is_nil()) return std::nullopt;
  if (!v->is_int()) throw ProtocolError(key + " must be int or none");
  return v->as_int();
}

// An inline buffer has no file path, so this field accepts the empty string
// without weakening the global require_str rule.
std::string require_str_allow_empty(const Value& map, const std::string& key) {
  const Value& v = require_field(map, key);
  if (!v.is_str()) throw ProtocolError(key + " must be a str");
  return v.as_str();
}

int base64_value(unsigned char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  if (c == '=') return -2;  // padding ends the data
  return -1;
}

std::string base64_decode(const std::string& text) {
  std::string clean;
  clean.reserve(text.size());
  for (unsigned char c : text) {
    if (c != '\n' && c != '\r' && c != ' ') clean.push_back(static_cast<char>(c));
  }

  const size_t data_end = clean.find('=');
  const size_t data_size = data_end == std::string::npos ? clean.size() : data_end;
  if (data_size % 4 == 1) throw ProtocolError("header is not valid base64");
  if (data_end != std::string::npos) {
    const size_t padding = clean.size() - data_end;
    if (padding > 2 || clean.size() % 4 != 0 ||
        clean.find_first_not_of('=', data_end) != std::string::npos) {
      throw ProtocolError("header is not valid base64");
    }
  }

  std::string out;
  uint32_t acc = 0;
  int bits = 0;
  for (size_t i = 0; i < data_size; ++i) {
    const int value = base64_value(static_cast<unsigned char>(clean[i]));
    if (value < 0) throw ProtocolError("header is not valid base64");
    acc = (acc << 6) | value;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<char>((acc >> bits) & 0xff));
    }
  }
  return out;
}

std::vector<int64_t> read_int_list(const Value& v, const std::string& name) {
  if (!v.is_array()) throw ProtocolError(name + " must be list[int]");
  std::vector<int64_t> out;
  for (const auto& item : v.as_array()) {
    if (!item.is_int()) throw ProtocolError(name + " must be list[int]");
    out.push_back(item.as_int());
  }
  return out;
}

Value int_list_value(const std::vector<int64_t>& items) {
  Value::Array arr;
  arr.reserve(items.size());
  for (int64_t i : items) arr.emplace_back(i);
  return Value(std::move(arr));
}

Value string_list_value(const std::vector<std::string>& items) {
  Value::Array arr;
  arr.reserve(items.size());
  for (const auto& s : items) arr.emplace_back(s);
  return Value(std::move(arr));
}

// Optional map field of str -> int, used by replica_bindings.
Value bindings_value(const std::optional<std::map<std::string, int64_t>>& b) {
  if (!b) return Value();
  Value::Map m;
  for (const auto& [k, v] : *b) m[k] = Value(v);
  return Value(std::move(m));
}

std::optional<std::map<std::string, int64_t>> read_bindings(const Value& map,
                                                            const std::string& key) {
  const Value* v = map.find(key);
  if (v == nullptr || v->is_nil()) return std::nullopt;
  std::map<std::string, int64_t> out;
  for (const auto& [k, item] : v->as_map()) out[k] = item.as_int();
  return out;
}

}  // namespace

// ---- enum spellings ------------------------------------------------------

const char* to_string(TransportKind k) {
  switch (k) {
    case TransportKind::LocalObject: return "local_object";
    case TransportKind::CudaIpc: return "cuda_ipc";
    case TransportKind::Shm: return "shm";
    case TransportKind::Mooncake: return "mooncake";
  }
  return "shm";
}

const char* to_string(DataKind k) {
  switch (k) {
    case DataKind::StagePayload: return "stage_payload";
    case DataKind::StreamChunk: return "stream_chunk";
    case DataKind::StreamMetadataTensor: return "stream_metadata_tensor";
    case DataKind::KvPages: return "kv_pages";
    case DataKind::WeightBucket: return "weight_bucket";
    case DataKind::MoeExpertPayload: return "moe_expert_payload";
  }
  return "stage_payload";
}

const char* to_string(DataLayout k) {
  switch (k) {
    case DataLayout::PackedTensors: return "packed_tensors";
    case DataLayout::RawTensor: return "raw_tensor";
    case DataLayout::Paged: return "paged";
    case DataLayout::Bucketed: return "bucketed";
    case DataLayout::Scatter: return "scatter";
  }
  return "packed_tensors";
}

TransportKind transport_from_string(const std::string& s) {
  if (s == "local_object") return TransportKind::LocalObject;
  if (s == "cuda_ipc") return TransportKind::CudaIpc;
  if (s == "shm") return TransportKind::Shm;
  if (s == "mooncake") return TransportKind::Mooncake;
  throw ProtocolError("unknown transport: " + s);
}

DataKind data_kind_from_string(const std::string& s) {
  if (s == "stage_payload") return DataKind::StagePayload;
  if (s == "stream_chunk") return DataKind::StreamChunk;
  if (s == "stream_metadata_tensor") return DataKind::StreamMetadataTensor;
  if (s == "kv_pages") return DataKind::KvPages;
  if (s == "weight_bucket") return DataKind::WeightBucket;
  if (s == "moe_expert_payload") return DataKind::MoeExpertPayload;
  throw ProtocolError("unknown data kind: " + s);
}

DataLayout data_layout_from_string(const std::string& s) {
  if (s == "packed_tensors") return DataLayout::PackedTensors;
  if (s == "raw_tensor") return DataLayout::RawTensor;
  if (s == "paged") return DataLayout::Paged;
  if (s == "bucketed") return DataLayout::Bucketed;
  if (s == "scatter") return DataLayout::Scatter;
  throw ProtocolError("unknown data layout: " + s);
}

// ---- data_ref ------------------------------------------------------------

Value TensorMeta::to_value() const {
  Value::Map m;
  m["path"] = Value(path);
  m["shape"] = int_list_value(shape);
  m["dtype"] = Value(dtype);
  m["device"] = Value(device);
  m["offset"] = Value(offset);
  m["size"] = Value(size);
  return Value(std::move(m));
}

TensorMeta TensorMeta::from_value(const Value& v) {
  TensorMeta t;
  t.path = require_str_allow_empty(v, "path");
  t.shape = read_int_list(require_field(v, "shape"), "shape");
  t.dtype = require_str(v, "dtype");
  t.device = require_str(v, "device");
  t.offset = require_field(v, "offset").as_int();
  t.size = require_field(v, "size").as_int();
  return t;
}

Value BackendRef::to_value() const {
  Value::Map m;
  m["transport"] = Value(to_string(transport));
  m["info"] = info;
  m["length"] = Value(length);
  return Value(std::move(m));
}

BackendRef BackendRef::from_value(const Value& v) {
  BackendRef b;
  b.transport = transport_from_string(require_str(v, "transport"));
  b.info = require_field(v, "info");
  b.length = require_field(v, "length").as_int();
  return b;
}

Value MetadataTensorRef::to_value() const {
  Value::Map m;
  m["path"] = Value(path);
  m["ref"] = ref_obj ? ref_obj->to_value() : Value();
  return Value(std::move(m));
}

Value DataRef::to_value() const {
  Value::Map m;
  m["_type"] = Value("DataRef");
  m["version"] = Value(version);
  m["kind"] = Value(to_string(kind));
  m["object_id"] = Value(object_id);
  m["transport"] = Value(to_string(transport));
  m["layout"] = Value(to_string(layout));
  m["buffer"] = buffer.to_value();
  Value::Array tensors_out;
  for (const auto& t : tensors) tensors_out.push_back(t.to_value());
  m["tensors"] = Value(std::move(tensors_out));
  Value::Array metadata_tensors_out;
  for (const auto& t : metadata_tensors) metadata_tensors_out.push_back(t.to_value());
  m["metadata_tensors"] = Value(std::move(metadata_tensors_out));
  if (header) m["header"] = Value::bin(*header);
  if (shape) m["shape"] = int_list_value(*shape);
  if (dtype) m["dtype"] = Value(*dtype);
  if (device) m["device"] = Value(*device);
  if (offset) m["offset"] = Value(*offset);
  if (metadata) m["metadata"] = *metadata;
  return Value(std::move(m));
}

DataRef DataRef::from_value(const Value& v) {
  if (require_str(v, "_type") != "DataRef") {
    throw ProtocolError("data_ref must have _type='DataRef'");
  }
  DataRef d;
  d.version = require_field(v, "version").as_int();
  if (d.version != 1) {
    throw ProtocolError("unsupported DataRef version " + std::to_string(d.version));
  }
  d.kind = data_kind_from_string(require_str(v, "kind"));
  d.object_id = require_str(v, "object_id");
  d.transport = transport_from_string(require_str(v, "transport"));
  d.layout = data_layout_from_string(require_str(v, "layout"));
  d.buffer = BackendRef::from_value(require_field(v, "buffer"));
  if (const Value* h = v.find("header"); h != nullptr && !h->is_nil()) {
    // Python sends bytes as bin; older senders base64-encoded them as text.
    if (h->is_bin()) {
      d.header = h->as_bin();
    } else if (h->is_str()) {
      d.header = base64_decode(h->as_str());
    } else {
      throw ProtocolError("header must be bytes or base64 str");
    }
  }
  for (const auto& item : require_field(v, "tensors").as_array()) {
    d.tensors.push_back(TensorMeta::from_value(item));
  }
  for (const auto& item : require_field(v, "metadata_tensors").as_array()) {
    MetadataTensorRef ref;
    ref.path = require_str_allow_empty(item, "path");
    ref.ref_obj = std::make_shared<DataRef>(
        DataRef::from_value(require_field(item, "ref")));
    d.metadata_tensors.push_back(std::move(ref));
  }
  if (const Value* s = v.find("shape"); s != nullptr && !s->is_nil()) {
    d.shape = read_int_list(*s, "shape");
  }
  d.dtype = read_opt_str(v, "dtype");
  d.device = read_opt_str(v, "device");
  d.offset = read_opt_int(v, "offset");
  if (const Value* m = v.find("metadata"); m != nullptr && !m->is_nil()) {
    d.metadata = *m;
  }
  return d;
}

// ---- request -------------------------------------------------------------

Value OmniRequest::to_value() const {
  Value::Map m;
  m["_type"] = Value("OmniRequest");
  m["inputs"] = inputs;
  m["params"] = Value(params);
  m["metadata"] = Value(metadata);
  return Value(std::move(m));
}

OmniRequest OmniRequest::from_value(const Value& v) {
  OmniRequest r;
  if (const Value* i = v.find("inputs"); i != nullptr) r.inputs = *i;
  if (const Value* p = v.find("params"); p != nullptr && !p->is_nil()) {
    r.params = p->as_map();
  }
  if (const Value* m = v.find("metadata"); m != nullptr && !m->is_nil()) {
    r.metadata = m->as_map();
  }
  return r;
}

Value StagePayload::to_value() const {
  Value::Map m;
  m["_type"] = Value("StagePayload");
  m["request_id"] = Value(request_id);
  m["request"] = request.to_value();
  m["data"] = data;
  return Value(std::move(m));
}

StagePayload StagePayload::from_value(const Value& v) {
  StagePayload p;
  p.request_id = require_str(v, "request_id");
  if (const Value* r = v.find("request"); r != nullptr && !r->is_nil()) {
    p.request = OmniRequest::from_value(*r);
  }
  if (const Value* d = v.find("data"); d != nullptr) p.data = *d;
  return p;
}

// ---- admin ---------------------------------------------------------------

bool is_update_action(const std::string& action) {
  return action == kAdminInitWeightsUpdateGroup ||
         action == kAdminDestroyWeightsUpdateGroup ||
         action == kAdminUpdateWeightsFromDisk ||
         action == kAdminUpdateWeightsFromTensor ||
         action == kAdminUpdateWeightsFromDistributed;
}

Value AdminOperation::to_value() const {
  Value::Map m;
  m["op_id"] = Value(op_id);
  m["action"] = Value(action);
  m["payload"] = Value(payload);
  if (target_stages) {
    Value::Array arr;
    for (const auto& s : *target_stages) arr.emplace_back(s);
    m["target_stages"] = Value(std::move(arr));
  } else {
    m["target_stages"] = Value();
  }
  m["timeout_s"] = timeout_s ? Value(*timeout_s) : Value();
  return Value(std::move(m));
}

AdminOperation AdminOperation::from_value(const Value& v) {
  AdminOperation op;
  op.op_id = require_str(v, "op_id");
  op.action = require_str(v, "action");
  if (const Value* p = v.find("payload"); p != nullptr && !p->is_nil()) {
    op.payload = p->as_map();
  }
  if (const Value* t = v.find("target_stages"); t != nullptr && !t->is_nil()) {
    std::vector<std::string> stages;
    for (const auto& s : t->as_array()) stages.push_back(s.as_str());
    op.target_stages = std::move(stages);
  }
  if (const Value* t = v.find("timeout_s"); t != nullptr && !t->is_nil()) {
    op.timeout_s = t->as_double();
  }
  return op;
}

Value AdminResult::to_value() const {
  Value::Map m;
  m["op_id"] = Value(op_id);
  m["stage"] = Value(stage);
  m["action"] = Value(action);
  m["success"] = Value(success);
  m["message"] = Value(message);
  m["data"] = Value(data);
  m["error"] = opt_str(error);
  m["rank"] = opt_int(rank);
  m["role"] = opt_str(role);
  return Value(std::move(m));
}

AdminResult AdminResult::from_value(const Value& v) {
  AdminResult r;
  r.op_id = require_str(v, "op_id");
  r.stage = require_str(v, "stage");
  r.action = require_str(v, "action");
  r.success = require_bool(v, "success");
  if (const Value* m = v.find("message"); m != nullptr && !m->is_nil()) {
    r.message = m->as_str();
  }
  if (const Value* d = v.find("data"); d != nullptr && !d->is_nil()) {
    r.data = d->as_map();
  }
  r.error = read_opt_str(v, "error");
  r.rank = read_opt_int(v, "rank");
  r.role = read_opt_str(v, "role");
  return r;
}

// ---- kv_transfer ---------------------------------------------------------

Value KVBufferSpec::to_value() const {
  Value::Map m;
  m["name"] = Value(name);
  m["bytes_per_page"] = Value(bytes_per_page);
  return Value(std::move(m));
}

KVBufferSpec KVBufferSpec::from_value(const Value& v) {
  KVBufferSpec s;
  s.name = require_str(v, "name");
  s.bytes_per_page = require_field(v, "bytes_per_page").as_int();
  return s;
}

bool KVPoolLayout::compatible_with(const KVPoolLayout& other) const {
  if (layout_id != other.layout_id || page_size != other.page_size) return false;
  if (buffers.size() != other.buffers.size()) return false;
  for (size_t i = 0; i < buffers.size(); ++i) {
    if (buffers[i].name != other.buffers[i].name) return false;
    if (buffers[i].bytes_per_page != other.buffers[i].bytes_per_page) return false;
  }
  return true;
}

Value KVPoolLayout::to_value() const {
  Value::Map m;
  m["layout_id"] = Value(layout_id);
  m["page_size"] = Value(page_size);
  Value::Array arr;
  for (const auto& b : buffers) arr.push_back(b.to_value());
  m["buffers"] = Value(std::move(arr));
  return Value(std::move(m));
}

KVPoolLayout KVPoolLayout::from_value(const Value& v) {
  KVPoolLayout l;
  l.layout_id = require_str(v, "layout_id");
  l.page_size = require_field(v, "page_size").as_int();
  for (const auto& item : require_field(v, "buffers").as_array()) {
    l.buffers.push_back(KVBufferSpec::from_value(item));
  }
  return l;
}

Value KVTransferPrepareMessage::to_value() const {
  Value::Map m;
  m["type"] = Value("kv_transfer_prepare");
  m["request_id"] = Value(request_id);
  m["transfer_id"] = Value(transfer_id);
  m["from_stage"] = Value(from_stage);
  m["to_stage"] = Value(to_stage);
  m["source_pool_id"] = Value(source_pool_id);
  m["target_pool_id"] = Value(target_pool_id);
  m["source_page_indices"] = int_list_value(source_page_indices);
  m["source_layout"] = source_layout.to_value();
  m["metadata"] = Value(metadata);
  return Value(std::move(m));
}

KVTransferPrepareMessage KVTransferPrepareMessage::from_value(const Value& v) {
  KVTransferPrepareMessage msg;
  msg.request_id = require_str(v, "request_id");
  msg.transfer_id = require_str(v, "transfer_id");
  msg.from_stage = require_str(v, "from_stage");
  msg.to_stage = require_str(v, "to_stage");
  msg.source_pool_id = require_str(v, "source_pool_id");
  msg.target_pool_id = require_str(v, "target_pool_id");
  msg.source_page_indices =
      read_int_list(require_field(v, "source_page_indices"), "source_page_indices");
  msg.source_layout = KVPoolLayout::from_value(require_field(v, "source_layout"));
  if (const Value* m = v.find("metadata"); m != nullptr && !m->is_nil()) {
    msg.metadata = m->as_map();
  }
  return msg;
}

Value KVTransferReadyMessage::to_value() const {
  Value::Map m;
  m["type"] = Value("kv_transfer_ready");
  m["request_id"] = Value(request_id);
  m["transfer_id"] = Value(transfer_id);
  m["from_stage"] = Value(from_stage);
  m["to_stage"] = Value(to_stage);
  m["success"] = Value(success);
  m["destination_pool_id"] = opt_str(destination_pool_id);
  m["destination_page_indices"] = int_list_value(destination_page_indices);
  m["destination_ref"] = destination_ref ? *destination_ref : Value();
  m["error"] = opt_str(error);
  return Value(std::move(m));
}

KVTransferReadyMessage KVTransferReadyMessage::from_value(const Value& v) {
  KVTransferReadyMessage msg;
  msg.request_id = require_str(v, "request_id");
  msg.transfer_id = require_str(v, "transfer_id");
  msg.from_stage = require_str(v, "from_stage");
  msg.to_stage = require_str(v, "to_stage");
  msg.success = require_bool(v, "success");
  msg.destination_pool_id = read_opt_str(v, "destination_pool_id");
  if (const Value* p = v.find("destination_page_indices");
      p != nullptr && !p->is_nil()) {
    msg.destination_page_indices = read_int_list(*p, "destination_page_indices");
  }
  if (const Value* r = v.find("destination_ref"); r != nullptr && !r->is_nil()) {
    msg.destination_ref = *r;
  }
  msg.error = read_opt_str(v, "error");
  return msg;
}

// ---- messages ------------------------------------------------------------

Value DataReadyMessage::to_value() const {
  if (request_id.empty()) throw ProtocolError("request_id must be a non-empty str");
  if (from_stage.empty()) throw ProtocolError("from_stage must be a non-empty str");
  if (to_stage.empty()) throw ProtocolError("to_stage must be a non-empty str");
  if (is_done && error) {
    throw ProtocolError("stream signal cannot be both done and error");
  }
  if (is_done || error) {
    if (data_ref) throw ProtocolError("stream signal must not carry data_ref");
    if (chunk_id) throw ProtocolError("stream signal must not carry chunk_id");
  } else if (!data_ref || !data_ref->is_map()) {
    throw ProtocolError("DataReadyMessage.data_ref must be dict for data messages");
  }
  Value::Map m;
  m["type"] = Value("data_ready");
  m["request_id"] = Value(request_id);
  m["from_stage"] = Value(from_stage);
  m["to_stage"] = Value(to_stage);
  if (data_ref) m["data_ref"] = *data_ref;
  if (chunk_id) {
    if (*chunk_id < 0) throw ProtocolError("chunk_id must be a non-negative int");
    m["chunk_id"] = Value(*chunk_id);
  }
  if (is_done) m["is_done"] = Value(true);
  if (error) m["error"] = Value(*error);
  if (replica_bindings && !replica_bindings->empty()) {
    m["replica_bindings"] = bindings_value(replica_bindings);
  }
  return Value(std::move(m));
}

DataReadyMessage DataReadyMessage::from_value(const Value& v) {
  DataReadyMessage msg;
  msg.request_id = require_str(v, "request_id");
  msg.from_stage = require_str(v, "from_stage");
  msg.to_stage = require_str(v, "to_stage");
  if (const Value* d = v.find("data_ref"); d != nullptr && !d->is_nil()) {
    msg.data_ref = *d;
  }
  // Python omits is_done for ordinary data messages, so a missing field means
  // false; a present field must still be a bool.
  const Value* done = v.find("is_done");
  msg.is_done = done == nullptr ? false : require_bool(v, "is_done");
  msg.error = read_opt_str(v, "error");
  if (msg.is_done && msg.error) {
    throw ProtocolError("stream signal cannot be both done and error");
  }
  if (msg.is_done || msg.error) {
    if (msg.data_ref) throw ProtocolError("stream signal must not carry data_ref");
    if (v.contains("chunk_id")) {
      throw ProtocolError("stream signal must not carry chunk_id");
    }
  } else if (!msg.data_ref || !msg.data_ref->is_map()) {
    throw ProtocolError("data_ready data_ref must be dict for data messages");
  }
  msg.chunk_id = read_opt_int(v, "chunk_id");
  if (msg.chunk_id && *msg.chunk_id < 0) {
    throw ProtocolError("chunk_id must be a non-negative int");
  }
  msg.replica_bindings = read_bindings(v, "replica_bindings");
  return msg;
}

Value StreamChunkBatchMessage::to_value() const {
  Value::Map m;
  m["type"] = Value("stream_chunk_batch");
  m["from_stage"] = Value(from_stage);
  m["to_stage"] = Value(to_stage);
  m["request_ids"] = string_list_value(request_ids);
  m["chunk_ids"] = int_list_value(chunk_ids);
  m["dtypes"] = string_list_value(dtypes);
  Value::Array shapes_out;
  for (const auto& s : shapes) shapes_out.push_back(int_list_value(s));
  m["shapes"] = Value(std::move(shapes_out));
  Value::Array bytes_out;
  for (const auto& b : tensor_bytes) bytes_out.push_back(Value::bin(b));
  m["tensor_bytes"] = Value(std::move(bytes_out));
  m["pickled_metadata"] = Value::bin(pickled_metadata);
  Value::Map bindings;
  for (const auto& [stage, inner] : replica_bindings) {
    Value::Map im;
    for (const auto& [k, val] : inner) im[k] = Value(val);
    bindings[stage] = Value(std::move(im));
  }
  m["replica_bindings"] = Value(std::move(bindings));
  return Value(std::move(m));
}

StreamChunkBatchMessage StreamChunkBatchMessage::from_value(const Value& v) {
  StreamChunkBatchMessage msg;
  const char* columns[] = {"request_ids", "chunk_ids", "dtypes", "shapes",
                           "tensor_bytes"};
  size_t length = 0;
  bool first = true;
  for (const char* name : columns) {
    const Value& col = require_field(v, name);
    if (!col.is_array()) {
      throw ProtocolError("stream chunk batch columns must be lists");
    }
    if (first || col.as_array().size() > length) length = col.as_array().size();
    first = false;
  }
  for (const char* name : columns) {
    if (require_field(v, name).as_array().size() != length) {
      throw ProtocolError("stream chunk batch columns must have one length");
    }
  }
  for (const auto& item : require_field(v, "request_ids").as_array()) {
    if (!item.is_str() || item.as_str().empty()) {
      throw ProtocolError("request_id must be a non-empty str");
    }
    msg.request_ids.push_back(item.as_str());
  }
  for (const auto& item : require_field(v, "chunk_ids").as_array()) {
    if (!item.is_int() || item.as_int() < 0) {
      throw ProtocolError("chunk_id must be a non-negative int");
    }
    msg.chunk_ids.push_back(item.as_int());
  }
  for (const auto& item : require_field(v, "dtypes").as_array()) {
    msg.dtypes.push_back(item.as_str());
  }
  for (const auto& item : require_field(v, "shapes").as_array()) {
    msg.shapes.push_back(read_int_list(item, "shapes"));
  }
  for (const auto& item : require_field(v, "tensor_bytes").as_array()) {
    msg.tensor_bytes.push_back(item.as_bin());
  }
  const Value& meta = require_field(v, "pickled_metadata");
  if (!meta.is_bin()) {
    throw ProtocolError("stream chunk batch pickled_metadata must be bytes");
  }
  msg.pickled_metadata = meta.as_bin();
  const Value& bindings = require_field(v, "replica_bindings");
  if (!bindings.is_map()) {
    throw ProtocolError("stream chunk batch replica_bindings must be dict");
  }
  for (const auto& [stage, inner] : bindings.as_map()) {
    std::map<std::string, int64_t> im;
    for (const auto& [k, val] : inner.as_map()) im[k] = val.as_int();
    msg.replica_bindings[stage] = std::move(im);
  }
  msg.from_stage = require_str(v, "from_stage");
  msg.to_stage = require_str(v, "to_stage");
  return msg;
}

Value DataAckMessage::to_value() const {
  if (request_id.empty()) throw ProtocolError("request_id must be a non-empty str");
  if (from_stage.empty()) throw ProtocolError("from_stage must be a non-empty str");
  if (to_stage.empty()) throw ProtocolError("to_stage must be a non-empty str");
  if (object_id.empty()) throw ProtocolError("object_id must be a non-empty str");
  if (success && error) {
    throw ProtocolError("successful data ack must not carry error");
  }
  if (!success && (!error || error->empty())) {
    throw ProtocolError("error must be a non-empty str");
  }
  Value::Map m;
  m["type"] = Value("data_ack");
  m["request_id"] = Value(request_id);
  m["from_stage"] = Value(from_stage);
  m["to_stage"] = Value(to_stage);
  m["object_id"] = Value(object_id);
  m["success"] = Value(success);
  if (error) m["error"] = Value(*error);
  return Value(std::move(m));
}

DataAckMessage DataAckMessage::from_value(const Value& v) {
  DataAckMessage msg;
  msg.success = require_bool(v, "success");
  const Value* error = v.find("error");
  if (msg.success) {
    if (error != nullptr && !error->is_nil()) {
      throw ProtocolError("successful data ack must not carry error");
    }
  } else {
    msg.error = require_str(v, "error");
  }
  msg.request_id = require_str(v, "request_id");
  msg.from_stage = require_str(v, "from_stage");
  msg.to_stage = require_str(v, "to_stage");
  msg.object_id = require_str(v, "object_id");
  return msg;
}

Value AbortMessage::to_value() const {
  Value::Map m;
  m["type"] = Value("abort");
  m["request_id"] = Value(request_id);
  return Value(std::move(m));
}

AbortMessage AbortMessage::from_value(const Value& v) {
  return AbortMessage{require_str(v, "request_id")};
}

Value CompleteMessage::to_value() const {
  Value::Map m;
  m["type"] = Value("complete");
  m["request_id"] = Value(request_id);
  m["from_stage"] = Value(from_stage);
  m["success"] = Value(success);
  m["result"] = result;
  m["error"] = opt_str(error);
  return Value(std::move(m));
}

CompleteMessage CompleteMessage::from_value(const Value& v) {
  CompleteMessage msg;
  msg.request_id = require_str(v, "request_id");
  msg.from_stage = require_str(v, "from_stage");
  msg.success = require_bool(v, "success");
  if (const Value* r = v.find("result"); r != nullptr) msg.result = *r;
  msg.error = read_opt_str(v, "error");
  return msg;
}

Value StreamMessage::to_value() const {
  Value::Map m;
  m["type"] = Value("stream");
  m["request_id"] = Value(request_id);
  m["from_stage"] = Value(from_stage);
  m["chunk"] = chunk;
  m["stage_id"] = opt_int(stage_id);
  m["stage_name"] = opt_str(stage_name);
  m["modality"] = opt_str(modality);
  if (chunk_id) m["chunk_id"] = Value(*chunk_id);
  return Value(std::move(m));
}

StreamMessage StreamMessage::from_value(const Value& v) {
  StreamMessage msg;
  msg.request_id = require_str(v, "request_id");
  msg.from_stage = require_str(v, "from_stage");
  if (const Value* c = v.find("chunk"); c != nullptr) msg.chunk = *c;
  msg.stage_id = read_opt_int(v, "stage_id");
  msg.stage_name = read_opt_str(v, "stage_name");
  msg.modality = read_opt_str(v, "modality");
  msg.chunk_id = read_opt_int(v, "chunk_id");
  return msg;
}

Value SubmitMessage::to_value() const {
  Value::Map m;
  m["type"] = Value("submit");
  m["request_id"] = Value(request_id);
  m["data"] = data;
  if (replica_bindings && !replica_bindings->empty()) {
    m["replica_bindings"] = bindings_value(replica_bindings);
  }
  return Value(std::move(m));
}

SubmitMessage SubmitMessage::from_value(const Value& v) {
  SubmitMessage msg;
  msg.request_id = require_str(v, "request_id");
  const Value& data = require_field(v, "data");
  // A StagePayload map arrives as a plain map; it stays a map on this side and
  // the stage decodes it with StagePayload::from_value when it needs the fields.
  msg.data = data;
  msg.replica_bindings = read_bindings(v, "replica_bindings");
  return msg;
}

Value ShutdownMessage::to_value() const {
  Value::Map m;
  m["type"] = Value("shutdown");
  return Value(std::move(m));
}

ShutdownMessage ShutdownMessage::from_value(const Value&) { return ShutdownMessage{}; }

Value FreezeGCMessage::to_value() const {
  Value::Map m;
  m["type"] = Value("freeze_gc");
  return Value(std::move(m));
}

FreezeGCMessage FreezeGCMessage::from_value(const Value&) { return FreezeGCMessage{}; }

Value ProfilerStartMessage::to_value() const {
  Value::Map m;
  m["type"] = Value("profiler_start");
  m["run_id"] = Value(run_id);
  m["trace_path_template"] = Value(trace_path_template);
  m["event_dir"] = opt_str(event_dir);
  m["enable_torch"] = Value(enable_torch);
  if (event_buffer_capacity) m["event_buffer_capacity"] = Value(*event_buffer_capacity);
  return Value(std::move(m));
}

ProfilerStartMessage ProfilerStartMessage::from_value(const Value& v) {
  ProfilerStartMessage msg;
  msg.run_id = require_str(v, "run_id");
  msg.trace_path_template = require_str(v, "trace_path_template");
  msg.event_dir = read_opt_str(v, "event_dir");
  if (const Value* e = v.find("enable_torch"); e != nullptr && !e->is_nil()) {
    msg.enable_torch = e->as_bool();
  }
  msg.event_buffer_capacity = read_opt_int(v, "event_buffer_capacity");
  if (msg.event_buffer_capacity && *msg.event_buffer_capacity <= 0) {
    throw ProtocolError("event_buffer_capacity must be a positive integer");
  }
  return msg;
}

Value ProfilerStopMessage::to_value() const {
  Value::Map m;
  m["type"] = Value("profiler_stop");
  m["run_id"] = opt_str(run_id);
  return Value(std::move(m));
}

ProfilerStopMessage ProfilerStopMessage::from_value(const Value& v) {
  return ProfilerStopMessage{read_opt_str(v, "run_id")};
}

Value AdminMessage::to_value() const {
  Value::Map m;
  m["type"] = Value("admin");
  m["operation"] = operation.to_value();
  return Value(std::move(m));
}

AdminMessage AdminMessage::from_value(const Value& v) {
  return AdminMessage{AdminOperation::from_value(require_field(v, "operation"))};
}

Value AdminResultMessage::to_value() const {
  Value::Map m;
  m["type"] = Value("admin_result");
  m["result"] = result.to_value();
  return Value(std::move(m));
}

AdminResultMessage AdminResultMessage::from_value(const Value& v) {
  return AdminResultMessage{AdminResult::from_value(require_field(v, "result"))};
}

// ---- dispatch ------------------------------------------------------------

Message parse_message(const Value& v) {
  const std::string type = require_str(v, "type");
  if (type == "data_ready") return DataReadyMessage::from_value(v);
  if (type == "stream_chunk_batch") return StreamChunkBatchMessage::from_value(v);
  if (type == "data_ack") return DataAckMessage::from_value(v);
  if (type == "kv_transfer_prepare") return KVTransferPrepareMessage::from_value(v);
  if (type == "kv_transfer_ready") return KVTransferReadyMessage::from_value(v);
  if (type == "abort") return AbortMessage::from_value(v);
  if (type == "complete") return CompleteMessage::from_value(v);
  if (type == "stream") return StreamMessage::from_value(v);
  if (type == "submit") return SubmitMessage::from_value(v);
  if (type == "shutdown") return ShutdownMessage::from_value(v);
  if (type == "freeze_gc") return FreezeGCMessage::from_value(v);
  if (type == "profiler_start") return ProfilerStartMessage::from_value(v);
  if (type == "profiler_stop") return ProfilerStopMessage::from_value(v);
  if (type == "admin") return AdminMessage::from_value(v);
  if (type == "admin_result") return AdminResultMessage::from_value(v);
  throw ProtocolError("Unknown message type: " + type);
}

const char* message_type_name(const Message& msg) {
  return std::visit(
      [](const auto& m) -> const char* {
        using T = std::decay_t<decltype(m)>;
        if constexpr (std::is_same_v<T, DataReadyMessage>) return "data_ready";
        else if constexpr (std::is_same_v<T, StreamChunkBatchMessage>)
          return "stream_chunk_batch";
        else if constexpr (std::is_same_v<T, DataAckMessage>) return "data_ack";
        else if constexpr (std::is_same_v<T, KVTransferPrepareMessage>)
          return "kv_transfer_prepare";
        else if constexpr (std::is_same_v<T, KVTransferReadyMessage>)
          return "kv_transfer_ready";
        else if constexpr (std::is_same_v<T, AbortMessage>) return "abort";
        else if constexpr (std::is_same_v<T, CompleteMessage>) return "complete";
        else if constexpr (std::is_same_v<T, StreamMessage>) return "stream";
        else if constexpr (std::is_same_v<T, SubmitMessage>) return "submit";
        else if constexpr (std::is_same_v<T, ShutdownMessage>) return "shutdown";
        else if constexpr (std::is_same_v<T, FreezeGCMessage>) return "freeze_gc";
        else if constexpr (std::is_same_v<T, ProfilerStartMessage>)
          return "profiler_start";
        else if constexpr (std::is_same_v<T, ProfilerStopMessage>)
          return "profiler_stop";
        else if constexpr (std::is_same_v<T, AdminMessage>) return "admin";
        else return "admin_result";
      },
      msg);
}

std::string serialize_message(const Message& msg) {
  return pack(std::visit([](const auto& m) { return m.to_value(); }, msg));
}

Message deserialize_message(const std::string& data) {
  return parse_message(unpack(data));
}

}  // namespace omniinfer
