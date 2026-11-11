// SPDX-License-Identifier: Apache-2.0
#include "umminfer/proto/wire.hpp"

#include <cstdint>

namespace umminfer {
namespace {

// msgpack C API packs strings and binary in two calls: length, then body.
void pack_str(msgpack_packer* pk, const std::string& s) {
  msgpack_pack_str(pk, s.size());
  msgpack_pack_str_body(pk, s.data(), s.size());
}

void pack_bin(msgpack_packer* pk, const std::string& s) {
  msgpack_pack_bin(pk, s.size());
  msgpack_pack_bin_body(pk, s.data(), s.size());
}

std::string read_str(const msgpack_object& o) {
  return std::string(o.via.str.ptr, o.via.str.size);
}

std::string read_bin(const msgpack_object& o) {
  return std::string(o.via.bin.ptr, o.via.bin.size);
}

// Tensor map keys, kept next to the encoder that writes them.
constexpr const char* kTensorMark = "__tensor__";

// A map is a tensor only when it is exactly the five fields the encoder writes.
// Any other map keeps every key, including one that happens to carry the mark.
bool is_complete_tensor_map(const Value::Map& m) {
  if (m.size() != 5) return false;
  auto mark = m.find(kTensorMark);
  if (mark == m.end() || !mark->second.is_bool() || !mark->second.as_bool()) {
    return false;
  }
  auto device = m.find("device");
  auto dtype = m.find("dtype");
  auto shape = m.find("shape");
  auto data = m.find("data");
  if (device == m.end() || dtype == m.end() || shape == m.end() ||
      data == m.end()) {
    return false;
  }
  if (!device->second.is_str() || !dtype->second.is_str()) return false;
  if (!data->second.is_bin() || !shape->second.is_array()) return false;
  for (const auto& dim : shape->second.as_array()) {
    if (!dim.is_int()) return false;
  }
  return true;
}

}  // namespace

void Value::encode(msgpack_packer* pk) const {
  switch (type()) {
    case Type::Nil:
      msgpack_pack_nil(pk);
      return;
    case Type::Bool:
      if (std::get<bool>(v_)) {
        msgpack_pack_true(pk);
      } else {
        msgpack_pack_false(pk);
      }
      return;
    case Type::Int:
      msgpack_pack_int64(pk, std::get<int64_t>(v_));
      return;
    case Type::Float:
      msgpack_pack_double(pk, std::get<double>(v_));
      return;
    case Type::Str:
      pack_str(pk, std::get<std::string>(v_));
      return;
    case Type::Bin:
      pack_bin(pk, std::get<Bin>(v_).data);
      return;
    case Type::Array: {
      const auto& items = std::get<Array>(v_);
      msgpack_pack_array(pk, items.size());
      for (const auto& item : items) item.encode(pk);
      return;
    }
    case Type::Map: {
      const auto& m = std::get<Map>(v_);
      msgpack_pack_map(pk, m.size());
      for (const auto& [k, item] : m) {
        pack_str(pk, k);
        item.encode(pk);
      }
      return;
    }
    case Type::Tensor_: {
      const auto& t = std::get<Tensor>(v_);
      msgpack_pack_map(pk, 5);
      pack_str(pk, kTensorMark);
      msgpack_pack_true(pk);
      pack_str(pk, "device");
      pack_str(pk, t.device);
      pack_str(pk, "dtype");
      pack_str(pk, t.dtype);
      pack_str(pk, "shape");
      msgpack_pack_array(pk, t.shape.size());
      for (int64_t d : t.shape) msgpack_pack_int64(pk, d);
      pack_str(pk, "data");
      pack_bin(pk, t.bytes);
      return;
    }
  }
  throw ProtocolError("unencodable value");
}

Value Value::decode(const msgpack_object& obj) {
  switch (obj.type) {
    case MSGPACK_OBJECT_NIL:
      return Value();
    case MSGPACK_OBJECT_BOOLEAN:
      return Value(static_cast<bool>(obj.via.boolean));
    case MSGPACK_OBJECT_POSITIVE_INTEGER: {
      // via.i64 is a union alias; read the unsigned field so a value above
      // INT64_MAX fails loudly instead of wrapping negative.
      const uint64_t value = obj.via.u64;
      if (value > static_cast<uint64_t>(INT64_MAX)) {
        throw ProtocolError("msgpack integer exceeds int64 range");
      }
      return Value(static_cast<int64_t>(value));
    }
    case MSGPACK_OBJECT_NEGATIVE_INTEGER:
      return Value(static_cast<int64_t>(obj.via.i64));
    case MSGPACK_OBJECT_FLOAT32:
    case MSGPACK_OBJECT_FLOAT64:
      return Value(obj.via.f64);
    case MSGPACK_OBJECT_STR:
      return Value(read_str(obj));
    case MSGPACK_OBJECT_BIN:
      return Value::bin(read_bin(obj));
    case MSGPACK_OBJECT_ARRAY: {
      Array items;
      items.reserve(obj.via.array.size);
      for (uint32_t i = 0; i < obj.via.array.size; ++i) {
        items.push_back(decode(obj.via.array.ptr[i]));
      }
      return Value(std::move(items));
    }
    case MSGPACK_OBJECT_MAP: {
      Map m;
      for (uint32_t i = 0; i < obj.via.map.size; ++i) {
        const auto& kv = obj.via.map.ptr[i];
        if (kv.key.type != MSGPACK_OBJECT_STR) {
          throw ProtocolError("map key must be str");
        }
        m[read_str(kv.key)] = decode(kv.val);
      }
      if (is_complete_tensor_map(m)) {
        Tensor t;
        if (auto it = m.find("device"); it != m.end()) {
          t.device = it->second.as_str();
        }
        if (auto it = m.find("dtype"); it != m.end()) {
          t.dtype = it->second.as_str();
        }
        if (auto it = m.find("shape"); it != m.end()) {
          for (const auto& dim : it->second.as_array()) {
            t.shape.push_back(dim.as_int());
          }
        }
        if (auto it = m.find("data"); it != m.end()) {
          t.bytes = it->second.as_bin();
        }
        return Value(std::move(t));
      }
      return Value(std::move(m));
    }
    default:
      throw ProtocolError("unsupported msgpack object type");
  }
}

std::string pack(const Value& value) {
  msgpack_sbuffer sbuf;
  msgpack_sbuffer_init(&sbuf);
  msgpack_packer pk;
  msgpack_packer_init(&pk, &sbuf, msgpack_sbuffer_write);
  try {
    value.encode(&pk);
  } catch (...) {
    msgpack_sbuffer_destroy(&sbuf);
    throw;
  }
  std::string out(sbuf.data, sbuf.size);
  msgpack_sbuffer_destroy(&sbuf);
  return out;
}

Value unpack(const std::string& data) {
  msgpack_unpacked result;
  msgpack_unpacked_init(&result);
  size_t offset = 0;
  if (!msgpack_unpack_next(&result, data.data(), data.size(), &offset)) {
    msgpack_unpacked_destroy(&result);
    throw ProtocolError("malformed msgpack payload");
  }
  Value value;
  try {
    value = Value::decode(result.data);
  } catch (...) {
    msgpack_unpacked_destroy(&result);
    throw;
  }
  msgpack_unpacked_destroy(&result);
  if (offset != data.size()) {
    throw ProtocolError("trailing bytes after msgpack value");
  }
  return value;
}

const Value& require_field(const Value& map, const std::string& key) {
  const Value* v = map.find(key);
  if (v == nullptr) throw ProtocolError("missing field: " + key);
  return *v;
}

std::string require_str(const Value& map, const std::string& key) {
  const Value& v = require_field(map, key);
  if (!v.is_str()) throw ProtocolError(key + " must be a non-empty str");
  const std::string& s = v.as_str();
  if (s.empty()) throw ProtocolError(key + " must be a non-empty str");
  return s;
}

bool require_bool(const Value& map, const std::string& key) {
  const Value& v = require_field(map, key);
  if (!v.is_bool()) throw ProtocolError(key + " must be bool");
  return v.as_bool();
}

int64_t require_non_negative_int(const Value& map, const std::string& key) {
  const Value& v = require_field(map, key);
  if (!v.is_int()) throw ProtocolError(key + " must be a non-negative int");
  const int64_t n = v.as_int();
  if (n < 0) throw ProtocolError(key + " must be a non-negative int");
  return n;
}

}  // namespace umminfer
