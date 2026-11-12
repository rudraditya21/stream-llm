#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace stream_llm {

struct TensorInfo {
    std::string name;
    std::int32_t type;
    std::string type_name;
    std::uint32_t rank;
    std::array<std::int64_t, 4> dimensions;
    std::uint64_t file_offset;
    std::size_t size;
};

class GgufIndex {
public:
    explicit GgufIndex(const std::filesystem::path & path);
    ~GgufIndex();

    GgufIndex(const GgufIndex &) = delete;
    GgufIndex & operator=(const GgufIndex &) = delete;
    GgufIndex(GgufIndex &&) noexcept;
    GgufIndex & operator=(GgufIndex &&) noexcept;

    std::size_t tensor_count() const noexcept;
    std::size_t alignment() const noexcept;
    std::uint64_t data_offset() const noexcept;
    const TensorInfo * find_tensor(std::string_view name) const noexcept;
    const TensorInfo & tensor(std::string_view name) const;

    std::uint32_t u32(std::string_view key) const;
    float f32(std::string_view key) const;
    std::string string(std::string_view key) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace stream_llm
