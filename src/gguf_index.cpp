#include "stream_llm/gguf_index.hpp"

#include "ggml.h"
#include "gguf.h"

#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace stream_llm {

struct GgufIndex::Impl {
    ~Impl() {
        if (metadata) ggml_free(metadata);
        if (gguf) gguf_free(gguf);
    }

    gguf_context * gguf = nullptr;
    ggml_context * metadata = nullptr;
    std::size_t alignment = 0;
    std::uint64_t data_offset = 0;
    std::unordered_map<std::string, TensorInfo> tensors;

    std::int64_t require_key(std::string_view key, gguf_type type) const {
        const std::string owned_key{key};
        const auto id = gguf_find_key(gguf, owned_key.c_str());
        if (id < 0) throw std::runtime_error("missing GGUF key: " + owned_key);
        if (gguf_get_kv_type(gguf, id) != type) {
            throw std::runtime_error("unexpected type for GGUF key: " + owned_key);
        }
        return id;
    }
};

GgufIndex::GgufIndex(const std::filesystem::path & path) : impl_(std::make_unique<Impl>()) {
    const auto file_size = std::filesystem::file_size(path);
    impl_->gguf = gguf_init_from_file(path.c_str(), {true, &impl_->metadata});
    if (!impl_->gguf || !impl_->metadata) throw std::runtime_error("failed to parse GGUF metadata: " + path.string());

    impl_->alignment = gguf_get_alignment(impl_->gguf);
    impl_->data_offset = gguf_get_data_offset(impl_->gguf);
    const auto count = gguf_get_n_tensors(impl_->gguf);
    if (count < 0) throw std::runtime_error("invalid negative GGUF tensor count");
    impl_->tensors.reserve(static_cast<std::size_t>(count));

    for (std::int64_t id = 0; id < count; ++id) {
        const std::string name = gguf_get_tensor_name(impl_->gguf, id);
        const auto relative_offset = gguf_get_tensor_offset(impl_->gguf, id);
        const auto size = gguf_get_tensor_size(impl_->gguf, id);
        if (relative_offset > std::numeric_limits<std::uint64_t>::max() - impl_->data_offset) {
            throw std::runtime_error("GGUF tensor offset overflow: " + name);
        }
        const std::uint64_t file_offset = impl_->data_offset + relative_offset;
        if (file_offset > file_size || size > file_size - file_offset) {
            throw std::runtime_error("GGUF tensor exceeds file bounds: " + name);
        }

        const auto * dimensions = gguf_get_tensor_ne(impl_->gguf, id);
        const auto type = gguf_get_tensor_type(impl_->gguf, id);
        const auto * descriptor = ggml_get_tensor(impl_->metadata, name.c_str());
        if (!descriptor) throw std::runtime_error("missing GGUF tensor descriptor: " + name);
        TensorInfo info{
            name,
            static_cast<std::int32_t>(type),
            ggml_type_name(type),
            static_cast<std::uint32_t>(ggml_n_dims(descriptor)),
            {dimensions[0], dimensions[1], dimensions[2], dimensions[3]},
            file_offset,
            size,
        };
        if (!impl_->tensors.emplace(name, std::move(info)).second) {
            throw std::runtime_error("duplicate GGUF tensor: " + name);
        }
    }
}

GgufIndex::~GgufIndex() = default;
GgufIndex::GgufIndex(GgufIndex &&) noexcept = default;
GgufIndex & GgufIndex::operator=(GgufIndex &&) noexcept = default;

std::size_t GgufIndex::tensor_count() const noexcept { return impl_->tensors.size(); }
std::size_t GgufIndex::alignment() const noexcept { return impl_->alignment; }
std::uint64_t GgufIndex::data_offset() const noexcept { return impl_->data_offset; }

const TensorInfo & GgufIndex::tensor(std::string_view name) const {
    const auto found = impl_->tensors.find(std::string{name});
    if (found == impl_->tensors.end()) throw std::runtime_error("missing GGUF tensor: " + std::string{name});
    return found->second;
}

std::uint32_t GgufIndex::u32(std::string_view key) const {
    return gguf_get_val_u32(impl_->gguf, impl_->require_key(key, GGUF_TYPE_UINT32));
}

float GgufIndex::f32(std::string_view key) const {
    return gguf_get_val_f32(impl_->gguf, impl_->require_key(key, GGUF_TYPE_FLOAT32));
}

std::string GgufIndex::string(std::string_view key) const {
    return gguf_get_val_str(impl_->gguf, impl_->require_key(key, GGUF_TYPE_STRING));
}

} // namespace stream_llm
