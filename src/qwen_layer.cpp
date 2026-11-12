#include "stream_llm/qwen_layer.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace stream_llm {

namespace {

constexpr std::array<std::string_view, 12> tensor_roles{
    "attn_norm.weight",
    "attn_q.weight",
    "attn_q.bias",
    "attn_k.weight",
    "attn_k.bias",
    "attn_v.weight",
    "attn_v.bias",
    "attn_output.weight",
    "ffn_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
};

std::size_t align_up(std::size_t value, std::size_t alignment) {
    if (!std::has_single_bit(alignment)) throw std::invalid_argument("layer alignment must be a power of two");
    if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        throw std::overflow_error("layer buffer alignment overflow");
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

} // namespace

std::size_t QwenLayerLayout::layer() const noexcept { return layer_; }
std::size_t QwenLayerLayout::size() const noexcept { return size_; }
std::size_t QwenLayerLayout::alignment() const noexcept { return alignment_; }
const std::vector<LayerTensorRange> & QwenLayerLayout::tensors() const noexcept { return tensors_; }

struct BoundLayer::Impl {
    ~Impl() {
        if (backend) ggml_backend_synchronize(backend);
        if (context) ggml_free(context);
        if (wrapper) ggml_backend_buffer_free(wrapper);
    }

    std::size_t layer = 0;
    std::size_t weight_bytes = 0;
    ggml_backend_t backend = nullptr;
    BufferLease lease;
    ggml_backend_buffer_t wrapper = nullptr;
    ggml_context * context = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors;
};

BoundLayer::BoundLayer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
BoundLayer::~BoundLayer() = default;
BoundLayer::BoundLayer(BoundLayer &&) noexcept = default;
BoundLayer & BoundLayer::operator=(BoundLayer &&) noexcept = default;

std::size_t BoundLayer::layer() const noexcept { return impl_->layer; }
std::size_t BoundLayer::weight_bytes() const noexcept { return impl_->weight_bytes; }

ggml_tensor * BoundLayer::tensor(std::string_view role) const {
    const auto found = impl_->tensors.find(std::string{role});
    if (found == impl_->tensors.end()) throw std::runtime_error("unbound layer tensor: " + std::string{role});
    return found->second;
}

void BoundLayer::synchronize() const { ggml_backend_synchronize(impl_->backend); }

QwenLayerStreamer::QwenLayerStreamer(const GgufIndex & index, const TensorReader & reader,
                                     BufferPool & pool, ggml_backend_t backend)
    : index_(index), reader_(reader), pool_(pool), backend_(backend) {
    if (!backend_) throw std::invalid_argument("layer streamer requires a backend");
    // shortcut: external weight binding currently targets the CPU backend used by the initial macOS runtime.
    if (!ggml_backend_is_cpu(backend_)) throw std::invalid_argument("layer streamer currently requires a CPU backend");
    if (index_.string("general.architecture") != "qwen2") {
        throw std::invalid_argument("layer streamer requires a qwen2 GGUF model");
    }
    if (pool_.alignment() < ggml_backend_get_alignment(backend_) ||
        pool_.alignment() % ggml_backend_get_alignment(backend_) != 0) {
        throw std::invalid_argument("buffer pool alignment does not satisfy backend alignment");
    }
}

QwenLayerLayout QwenLayerStreamer::map(std::size_t layer) const {
    const auto layer_count = static_cast<std::size_t>(index_.u32("qwen2.block_count"));
    if (layer >= layer_count) throw std::out_of_range("Qwen layer index is out of range");

    QwenLayerLayout layout;
    layout.layer_ = layer;
    layout.alignment_ = ggml_backend_get_alignment(backend_);
    layout.index_ = &index_;
    layout.tensors_.reserve(tensor_roles.size());
    std::size_t next = 0;
    for (const auto role : tensor_roles) {
        next = align_up(next, layout.alignment_);
        const std::string name = "blk." + std::to_string(layer) + "." + std::string{role};
        const auto & source = index_.tensor(name);
        layout.tensors_.push_back({std::string{role}, source, next});
        if (next > std::numeric_limits<std::size_t>::max() - source.size) {
            throw std::overflow_error("Qwen layer byte size overflow");
        }
        next += source.size;
    }
    layout.size_ = next;
    return layout;
}

BoundLayer QwenLayerStreamer::load(const QwenLayerLayout & layout) const {
    if (layout.index_ != &index_) {
        throw std::invalid_argument("layer layout belongs to a different GGUF index");
    }
    if (layout.alignment() != ggml_backend_get_alignment(backend_)) {
        throw std::invalid_argument("layer layout alignment does not match backend");
    }

    auto impl = std::make_unique<BoundLayer::Impl>();
    impl->layer = layout.layer();
    impl->weight_bytes = layout.size();
    impl->backend = backend_;
    impl->lease = pool_.acquire(layout.size());

    for (const auto & range : layout.tensors()) {
        reader_.read_at(range.source.file_offset,
                        {impl->lease.data() + range.buffer_offset, range.source.size});
    }

    impl->wrapper = ggml_backend_cpu_buffer_from_ptr(impl->lease.data(), impl->lease.size());
    if (!impl->wrapper) throw std::runtime_error("failed to wrap layer weight buffer");
    ggml_backend_buffer_set_usage(impl->wrapper, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const std::size_t context_size = layout.tensors().size() * ggml_tensor_overhead() + 4096;
    impl->context = ggml_init({context_size, nullptr, true});
    if (!impl->context) throw std::runtime_error("failed to create layer tensor context");

    for (const auto & range : layout.tensors()) {
        auto * address = impl->lease.data() + range.buffer_offset;
        const auto type = static_cast<ggml_type>(range.source.type);
        if (!ggml_validate_row_data(type, address, range.source.size)) {
            throw std::runtime_error("invalid tensor bytes: " + range.source.name);
        }
        auto * tensor = ggml_new_tensor(impl->context, type, static_cast<int>(range.source.rank),
                                        range.source.dimensions.data());
        ggml_set_name(tensor, range.source.name.c_str());
        if (ggml_backend_tensor_alloc(impl->wrapper, tensor, address) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("failed to bind tensor: " + range.source.name);
        }
        impl->tensors.emplace(range.role, tensor);
    }

    return BoundLayer{std::move(impl)};
}

} // namespace stream_llm
