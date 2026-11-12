#include "stream_llm/buffer_pool.hpp"

#include <algorithm>
#include <bit>
#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace stream_llm {

struct BufferLease::State {
    State(std::size_t budget_value, std::size_t capacity_value, std::size_t alignment_value)
        : budget(budget_value), capacity(capacity_value), alignment(alignment_value) {
        const std::size_t count = budget / capacity;
        buffers.reserve(count);
        used.assign(count, false);
        try {
            for (std::size_t i = 0; i < count; ++i) {
                buffers.push_back(::operator new(capacity, std::align_val_t(alignment)));
            }
        } catch (...) {
            for (void * buffer : buffers) {
                ::operator delete(buffer, std::align_val_t(alignment));
            }
            throw;
        }
    }

    ~State() {
        for (void * buffer : buffers) {
            ::operator delete(buffer, std::align_val_t(alignment));
        }
    }

    std::size_t budget;
    std::size_t capacity;
    std::size_t alignment;
    std::vector<void *> buffers;
    std::vector<bool> used;
    mutable std::mutex mutex;
};

BufferLease::BufferLease(std::shared_ptr<State> state, std::size_t index, std::size_t size)
    : state_(std::move(state)), index_(index), size_(size) {}

BufferLease::~BufferLease() { release(); }

BufferLease::BufferLease(BufferLease && other) noexcept
    : state_(std::move(other.state_)), index_(other.index_), size_(other.size_) {
    other.size_ = 0;
}

BufferLease & BufferLease::operator=(BufferLease && other) noexcept {
    if (this != &other) {
        release();
        state_ = std::move(other.state_);
        index_ = other.index_;
        size_ = other.size_;
        other.size_ = 0;
    }
    return *this;
}

BufferLease::operator bool() const noexcept { return state_ != nullptr; }

std::byte * BufferLease::data() const noexcept {
    return state_ ? static_cast<std::byte *>(state_->buffers[index_]) : nullptr;
}

std::size_t BufferLease::size() const noexcept { return size_; }

std::size_t BufferLease::capacity() const noexcept { return state_ ? state_->capacity : 0; }

std::span<std::byte> BufferLease::bytes() const noexcept { return {data(), size_}; }

void BufferLease::release() noexcept {
    if (!state_) return;
    {
        std::lock_guard lock(state_->mutex);
        state_->used[index_] = false;
    }
    state_.reset();
    size_ = 0;
}

BufferPool::BufferPool(std::size_t budget, std::size_t buffer_capacity, std::size_t alignment) {
    if (budget == 0 || buffer_capacity == 0 || buffer_capacity > budget) {
        throw std::invalid_argument("buffer pool requires a non-zero budget and capacity within that budget");
    }
    if (alignment < alignof(void *) || !std::has_single_bit(alignment)) {
        throw std::invalid_argument("buffer alignment must be a power of two and at least pointer-aligned");
    }
    state_ = std::make_shared<BufferLease::State>(budget, buffer_capacity, alignment);
}

BufferLease BufferPool::acquire(std::size_t size) {
    if (size == 0 || size > state_->capacity) {
        throw std::invalid_argument("requested buffer size is zero or exceeds buffer capacity");
    }
    std::lock_guard lock(state_->mutex);
    const auto free = std::find(state_->used.begin(), state_->used.end(), false);
    if (free == state_->used.end()) throw std::runtime_error("buffer pool exhausted");
    const auto index = static_cast<std::size_t>(free - state_->used.begin());
    state_->used[index] = true;
    return BufferLease{state_, index, size};
}

std::size_t BufferPool::budget() const noexcept { return state_->budget; }
std::size_t BufferPool::buffer_capacity() const noexcept { return state_->capacity; }
std::size_t BufferPool::alignment() const noexcept { return state_->alignment; }
std::size_t BufferPool::buffer_count() const noexcept { return state_->buffers.size(); }

std::size_t BufferPool::available() const {
    std::lock_guard lock(state_->mutex);
    return static_cast<std::size_t>(std::count(state_->used.begin(), state_->used.end(), false));
}

} // namespace stream_llm
