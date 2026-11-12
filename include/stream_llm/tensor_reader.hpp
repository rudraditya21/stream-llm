#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

namespace stream_llm {

class TensorReader {
public:
    explicit TensorReader(const std::filesystem::path & path);
    ~TensorReader();

    TensorReader(const TensorReader &) = delete;
    TensorReader & operator=(const TensorReader &) = delete;

    std::uint64_t size() const noexcept;
    void read_at(std::uint64_t offset, std::span<std::byte> destination) const;

private:
    int file_ = -1;
    std::uint64_t size_ = 0;
};

} // namespace stream_llm
