#include "stream_llm/tensor_reader.hpp"

#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/stat.h>
#include <unistd.h>

namespace stream_llm {

// shortcut: positional I/O is POSIX-only, add a Windows ReadFile path when Windows support begins.

TensorReader::TensorReader(const std::filesystem::path & path) {
    file_ = ::open(path.c_str(), O_RDONLY);
    if (file_ < 0) throw std::system_error(errno, std::generic_category(), "open " + path.string());

    struct stat status {};
    if (::fstat(file_, &status) != 0) {
        const int error = errno;
        ::close(file_);
        file_ = -1;
        throw std::system_error(error, std::generic_category(), "stat " + path.string());
    }
    if (status.st_size < 0) {
        ::close(file_);
        file_ = -1;
        throw std::runtime_error("negative file size: " + path.string());
    }
    size_ = static_cast<std::uint64_t>(status.st_size);
}

TensorReader::~TensorReader() {
    if (file_ >= 0) ::close(file_);
}

std::uint64_t TensorReader::size() const noexcept { return size_; }

void TensorReader::read_at(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > size_ || destination.size() > size_ - offset) {
        throw std::out_of_range("positional read exceeds file bounds");
    }
    const auto maximum_offset = static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
    if (offset > maximum_offset || destination.size() > maximum_offset - offset) {
        throw std::out_of_range("positional read offset exceeds platform limit");
    }

    std::size_t completed = 0;
    while (completed < destination.size()) {
        const auto position = static_cast<off_t>(offset + completed);
        const auto result = ::pread(file_, destination.data() + completed,
                                    destination.size() - completed, position);
        if (result < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "positional read");
        }
        if (result == 0) throw std::runtime_error("unexpected end of file during positional read");
        completed += static_cast<std::size_t>(result);
    }
}

} // namespace stream_llm
