#pragma once

#include "stream_llm/buffer_pool.hpp"
#include "stream_llm/gguf_index.hpp"
#include "stream_llm/tensor_reader.hpp"

#include "ggml-backend.h"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace stream_llm {

struct LayerTensorRange {
    std::string role;
    TensorInfo source;
    std::size_t buffer_offset;
};

class QwenLayerLayout {
public:
    std::size_t layer() const noexcept;
    std::size_t size() const noexcept;
    std::size_t alignment() const noexcept;
    const std::vector<LayerTensorRange> & tensors() const noexcept;

private:
    friend class QwenLayerStreamer;

    std::size_t layer_ = 0;
    std::size_t size_ = 0;
    std::size_t alignment_ = 0;
    const GgufIndex * index_ = nullptr;
    std::vector<LayerTensorRange> tensors_;
};

class BoundLayer {
public:
    ~BoundLayer();

    BoundLayer(const BoundLayer &) = delete;
    BoundLayer & operator=(const BoundLayer &) = delete;
    BoundLayer(BoundLayer &&) noexcept;
    BoundLayer & operator=(BoundLayer &&) noexcept;

    std::size_t layer() const noexcept;
    std::size_t weight_bytes() const noexcept;
    ggml_tensor * tensor(std::string_view role) const;
    void synchronize() const;

private:
    friend class QwenLayerStreamer;
    struct Impl;

    explicit BoundLayer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class QwenLayerStreamer {
public:
    QwenLayerStreamer(const GgufIndex & index, const TensorReader & reader,
                      BufferPool & pool, ggml_backend_t backend);

    QwenLayerLayout map(std::size_t layer) const;
    BoundLayer load(const QwenLayerLayout & layout) const;

private:
    const GgufIndex & index_;
    const TensorReader & reader_;
    BufferPool & pool_;
    ggml_backend_t backend_;
};

} // namespace stream_llm
