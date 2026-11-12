#pragma once

#include <cstddef>
#include <memory>
#include <span>

namespace stream_llm {

class BufferPool;

class BufferLease {
public:
    BufferLease() = default;
    ~BufferLease();

    BufferLease(const BufferLease &) = delete;
    BufferLease & operator=(const BufferLease &) = delete;
    BufferLease(BufferLease && other) noexcept;
    BufferLease & operator=(BufferLease && other) noexcept;

    explicit operator bool() const noexcept;
    std::byte * data() const noexcept;
    std::size_t size() const noexcept;
    std::size_t capacity() const noexcept;
    std::span<std::byte> bytes() const noexcept;

private:
    struct State;
    friend class BufferPool;

    BufferLease(std::shared_ptr<State> state, std::size_t index, std::size_t size);
    void release() noexcept;

    std::shared_ptr<State> state_;
    std::size_t index_ = 0;
    std::size_t size_ = 0;
};

class BufferPool {
public:
    BufferPool(std::size_t budget, std::size_t buffer_capacity, std::size_t alignment);

    BufferLease acquire(std::size_t size);
    std::size_t budget() const noexcept;
    std::size_t buffer_capacity() const noexcept;
    std::size_t alignment() const noexcept;
    std::size_t buffer_count() const noexcept;
    std::size_t available() const;

private:
    std::shared_ptr<BufferLease::State> state_;
};

} // namespace stream_llm
