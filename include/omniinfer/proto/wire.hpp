// SPDX-License-Identifier: Apache-2.0
// MessagePack value used by every control-plane and payload field.
//
// MessagePack values hold primitive types and a Tensor leaf.
#pragma once

#include <msgpack.h>

#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace omniinfer {

// Invalid wire types and schema fields raise ProtocolError.
class ProtocolError : public std::runtime_error {
 public:
  explicit ProtocolError(const std::string& what) : std::runtime_error(what) {}
};

// A dense tensor leaf. The wire form is a map with the key "__tensor__".
struct Tensor {
  std::string device;   // "cpu", "cuda:0", "mps", ...
  std::string dtype;    // "float32", "int64", ...
  std::vector<int64_t> shape;
  std::string bytes;    // raw little-endian payload

  size_t nbytes() const { return bytes.size(); }
  bool operator==(const Tensor& o) const {
    return device == o.device && dtype == o.dtype && shape == o.shape &&
           bytes == o.bytes;
  }
};

class Value {
 public:
  using Array = std::vector<Value>;
  using Map = std::map<std::string, Value>;

  enum class Type { Nil, Bool, Int, Float, Str, Bin, Array, Map, Tensor_ };

  Value() : v_(std::monostate{}) {}
  Value(std::nullptr_t) : v_(std::monostate{}) {}
  Value(bool b) : v_(b) {}
  Value(int i) : v_(static_cast<int64_t>(i)) {}
  Value(int64_t i) : v_(i) {}
  Value(uint64_t i) {
    if (i > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      throw ProtocolError("uint64 value exceeds signed int64");
    }
    v_ = static_cast<int64_t>(i);
  }
  Value(double d) : v_(d) {}
  Value(std::string s) : v_(std::move(s)) {}
  Value(const char* s) : v_(std::string(s)) {}
  Value(Array a) : v_(std::move(a)) {}
  Value(Map m) : v_(std::move(m)) {}
  Value(Tensor t) : v_(std::move(t)) {}

  static Value bin(std::string b) {
    Value v;
    v.v_ = Bin{std::move(b)};
    return v;
  }
  static Value array(Array a) { return Value(std::move(a)); }
  static Value map(Map m) { return Value(std::move(m)); }

  Type type() const {
    switch (v_.index()) {
      case 0: return Type::Nil;
      case 1: return Type::Bool;
      case 2: return Type::Int;
      case 3: return Type::Float;
      case 4: return Type::Str;
      case 5: return Type::Bin;
      case 6: return Type::Array;
      case 7: return Type::Map;
      default: return Type::Tensor_;
    }
  }

  bool is_nil() const { return type() == Type::Nil; }
  bool is_bool() const { return type() == Type::Bool; }
  bool is_int() const { return type() == Type::Int; }
  bool is_float() const { return type() == Type::Float; }
  bool is_str() const { return type() == Type::Str; }
  bool is_bin() const { return type() == Type::Bin; }
  bool is_array() const { return type() == Type::Array; }
  bool is_map() const { return type() == Type::Map; }
  bool is_tensor() const { return type() == Type::Tensor_; }

  bool as_bool() const {
    if (!is_bool()) throw ProtocolError("value must be bool");
    return std::get<bool>(v_);
  }
  int64_t as_int() const {
    if (!is_int()) throw ProtocolError("value must be int");
    return std::get<int64_t>(v_);
  }
  double as_double() const {
    if (is_int()) return static_cast<double>(std::get<int64_t>(v_));
    if (!is_float()) throw ProtocolError("value must be a number");
    return std::get<double>(v_);
  }
  // Text access accepts both str and bin, the two byte-like msgpack types.
  const std::string& as_str() const {
    if (is_str()) return std::get<std::string>(v_);
    if (is_bin()) return std::get<Bin>(v_).data;
    throw ProtocolError("value must be str");
  }
  const std::string& as_bin() const {
    if (is_bin()) return std::get<Bin>(v_).data;
    if (is_str()) return std::get<std::string>(v_);
    throw ProtocolError("value must be bytes");
  }
  const Array& as_array() const {
    if (!is_array()) throw ProtocolError("value must be a list");
    return std::get<Array>(v_);
  }
  const Map& as_map() const {
    if (!is_map()) throw ProtocolError("value must be a map");
    return std::get<Map>(v_);
  }
  Map& as_map_mut() {
    if (!is_map()) throw ProtocolError("value must be a map");
    return std::get<Map>(v_);
  }
  const Tensor& as_tensor() const {
    if (!is_tensor()) throw ProtocolError("value must be a tensor");
    return std::get<Tensor>(v_);
  }

  bool contains(const std::string& key) const {
    if (!is_map()) return false;
    return std::get<Map>(v_).count(key) != 0;
  }
  const Value* find(const std::string& key) const {
    if (!is_map()) return nullptr;
    const auto& m = std::get<Map>(v_);
    auto it = m.find(key);
    return it == m.end() ? nullptr : &it->second;
  }
  Value& operator[](const std::string& key) { return as_map_mut()[key]; }

  void encode(msgpack_packer* pk) const;
  static Value decode(const msgpack_object& obj);

 private:
  struct Bin {
    std::string data;
    bool operator==(const Bin& o) const { return data == o.data; }
  };
  std::variant<std::monostate, bool, int64_t, double, std::string, Bin, Array,
               Map, Tensor>
      v_;
};

// Pack a value into a fresh msgpack buffer (use_bin_type equivalent).
std::string pack(const Value& value);

// Unpack exactly one value. Trailing bytes are rejected.
Value unpack(const std::string& data);

// ---- schema field helpers --------------------------------------------------

const Value& require_field(const Value& map, const std::string& key);
std::string require_str(const Value& map, const std::string& key);
bool require_bool(const Value& map, const std::string& key);
int64_t require_non_negative_int(const Value& map, const std::string& key);

}  // namespace omniinfer
