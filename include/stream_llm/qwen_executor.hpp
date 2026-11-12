#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace stream_llm {

struct QwenConfig {
    std::size_t embedding_length;
    std::size_t feed_forward_length;
    std::size_t layer_count;
    std::size_t head_count;
    std::size_t kv_head_count;
    std::size_t head_dimension;
    std::size_t vocabulary_size;
    float rms_epsilon;
    float rope_base;
};

class QwenExecutor {
public:
    explicit QwenExecutor(const std::filesystem::path & model_path,
                          std::size_t weight_buffer_capacity = 64U * 1024U * 1024U);
    ~QwenExecutor();

    QwenExecutor(const QwenExecutor &) = delete;
    QwenExecutor & operator=(const QwenExecutor &) = delete;
    QwenExecutor(QwenExecutor &&) noexcept;
    QwenExecutor & operator=(QwenExecutor &&) noexcept;

    const QwenConfig & config() const noexcept;
    std::size_t position() const noexcept;
    std::vector<float> decode(std::int32_t token);
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace stream_llm
