#include "stream_llm/buffer_pool.hpp"
#include "stream_llm/gguf_index.hpp"
#include "stream_llm/tensor_reader.hpp"

#include "ggml.h"
#include "gguf.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>

namespace {

void check(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void check_throws(Function function, const std::string & message) {
    try {
        function();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error(message);
}

class FixtureFile {
public:
    FixtureFile() : path_(std::filesystem::temp_directory_path() /
                          ("stream-llm-runtime-" + std::to_string(::getpid()) + ".gguf")) {
        gguf_context * gguf = gguf_init_empty();
        if (!gguf) throw std::runtime_error("failed to create GGUF fixture context");
        ggml_context * tensors = ggml_init({4096, nullptr, false});
        if (!tensors) {
            gguf_free(gguf);
            throw std::runtime_error("failed to create tensor fixture context");
        }

        gguf_set_val_str(gguf, "general.architecture", "test");
        gguf_set_val_u32(gguf, "test.block_count", 2);
        gguf_set_val_f32(gguf, "test.scale", 1.25F);
        auto * tensor = ggml_new_tensor_2d(tensors, GGML_TYPE_F32, 2, 3);
        ggml_set_name(tensor, "test.weight");
        const std::array<float, 6> values{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
        std::memcpy(tensor->data, values.data(), sizeof(values));
        gguf_add_tensor(gguf, tensor);
        const bool written = gguf_write_to_file(gguf, path_.c_str(), false);
        ggml_free(tensors);
        gguf_free(gguf);
        if (!written) throw std::runtime_error("failed to write GGUF fixture");
    }

    ~FixtureFile() {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    const std::filesystem::path & path() const { return path_; }

private:
    std::filesystem::path path_;
};

void test_buffer_pool() {
    stream_llm::BufferLease survivor;
    {
        stream_llm::BufferPool pool{512, 256, 64};
        check(pool.buffer_count() == 2, "unexpected buffer count");
        check(pool.available() == 2, "buffers should start available");
        auto first = pool.acquire(128);
        survivor = pool.acquire(256);
        check(reinterpret_cast<std::uintptr_t>(first.data()) % 64 == 0, "buffer is not aligned");
        check(first.bytes().size() == 128, "lease span has wrong size");
        check(pool.available() == 0, "leased buffers must not be available");
        check_throws([&] { static_cast<void>(pool.acquire(1)); }, "exhausted pool did not fail");
        first = {};
        check(pool.available() == 1, "released buffer was not returned");
    }
    check(survivor && survivor.capacity() == 256, "lease did not preserve buffer lifetime");
}

void test_gguf_and_storage() {
    FixtureFile fixture;
    stream_llm::GgufIndex index{fixture.path()};
    check(index.tensor_count() == 1, "unexpected tensor count");
    check(index.string("general.architecture") == "test", "string metadata mismatch");
    check(index.u32("test.block_count") == 2, "u32 metadata mismatch");
    check(std::abs(index.f32("test.scale") - 1.25F) < 1e-6F, "f32 metadata mismatch");

    const auto & tensor = index.tensor("test.weight");
    check(tensor.type_name == "f32", "tensor type mismatch");
    check(tensor.rank == 2 && tensor.dimensions[0] == 2 && tensor.dimensions[1] == 3,
          "tensor shape mismatch");
    check(tensor.size == 6 * sizeof(float), "tensor byte size mismatch");

    stream_llm::TensorReader reader{fixture.path()};
    stream_llm::BufferPool pool{128, 64, index.alignment()};
    auto lease = pool.acquire(tensor.size);
    reader.read_at(tensor.file_offset, lease.bytes());
    const auto * values = reinterpret_cast<const float *>(lease.data());
    for (std::size_t i = 0; i < 6; ++i) {
        check(values[i] == static_cast<float>(i + 1), "positional tensor read mismatch");
    }

    check_throws([&] { static_cast<void>(index.tensor("missing")); }, "missing tensor did not fail");
    check_throws([&] { static_cast<void>(index.u32("general.architecture")); },
                 "wrong metadata type did not fail");
    std::array<std::byte, 1> byte{};
    check_throws([&] { reader.read_at(reader.size(), byte); }, "out-of-bounds read did not fail");
}

void test_reference_model(const std::filesystem::path & path) {
    stream_llm::GgufIndex index{path};
    check(index.string("general.architecture") == "qwen2", "reference architecture mismatch");
    check(index.u32("qwen2.block_count") == 36, "reference layer count mismatch");
    const auto & tensor = index.tensor("blk.0.attn_k.weight");
    check(tensor.size > 0, "reference tensor is empty");

    stream_llm::TensorReader reader{path};
    stream_llm::BufferPool pool{tensor.size, tensor.size, index.alignment()};
    auto lease = pool.acquire(tensor.size);
    reader.read_at(tensor.file_offset, lease.bytes());
}

} // namespace

int main(int argc, char ** argv) {
    try {
        test_buffer_pool();
        test_gguf_and_storage();
        if (argc == 2) test_reference_model(argv[1]);
        if (argc > 2) throw std::runtime_error("runtime-tests accepts at most one GGUF path");
        std::cout << "runtime interface tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "runtime interface tests failed: " << error.what() << '\n';
        return 1;
    }
}
