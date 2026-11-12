#include "stream_llm/buffer_pool.hpp"
#include "stream_llm/generator.hpp"
#include "stream_llm/gguf_index.hpp"
#include "stream_llm/qwen_executor.hpp"
#include "stream_llm/qwen_layer.hpp"
#include "stream_llm/tensor_reader.hpp"
#include "stream_llm/tokenizer.hpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
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

constexpr std::array<const char *, 12> qwen_tensor_roles{
    "attn_norm.weight", "attn_q.weight", "attn_q.bias", "attn_k.weight",
    "attn_k.bias", "attn_v.weight", "attn_v.bias", "attn_output.weight",
    "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
};

class QwenFixtureFile {
public:
    QwenFixtureFile() : path_(std::filesystem::temp_directory_path() /
                              ("stream-llm-qwen-layer-" + std::to_string(::getpid()) + ".gguf")) {
        gguf_context * gguf = gguf_init_empty();
        if (!gguf) throw std::runtime_error("failed to create Qwen GGUF fixture context");
        ggml_context * tensors = ggml_init({64 * 1024, nullptr, false});
        if (!tensors) {
            gguf_free(gguf);
            throw std::runtime_error("failed to create Qwen tensor fixture context");
        }

        gguf_set_val_str(gguf, "general.architecture", "qwen2");
        gguf_set_val_u32(gguf, "qwen2.block_count", 1);
        for (std::size_t i = 0; i < qwen_tensor_roles.size(); ++i) {
            auto * tensor = ggml_new_tensor_2d(tensors, GGML_TYPE_F32, 2, 2);
            const std::string name = "blk.0." + std::string{qwen_tensor_roles[i]};
            ggml_set_name(tensor, name.c_str());
            std::array<float, 4> values{};
            values.fill(static_cast<float>(i + 1));
            std::memcpy(tensor->data, values.data(), sizeof(values));
            gguf_add_tensor(gguf, tensor);
        }
        const bool written = gguf_write_to_file(gguf, path_.c_str(), false);
        ggml_free(tensors);
        gguf_free(gguf);
        if (!written) throw std::runtime_error("failed to write Qwen GGUF fixture");
    }

    ~QwenFixtureFile() {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    const std::filesystem::path & path() const { return path_; }

private:
    std::filesystem::path path_;
};

struct BackendDel {
    void operator()(ggml_backend * backend) const { ggml_backend_free(backend); }
};
using BackendPtr = std::unique_ptr<ggml_backend, BackendDel>;

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

void test_layer_streaming() {
    QwenFixtureFile fixture;
    stream_llm::GgufIndex index{fixture.path()};
    stream_llm::TensorReader reader{fixture.path()};
    BackendPtr backend{ggml_backend_cpu_init()};
    check(backend != nullptr, "failed to initialize test CPU backend");
    const auto alignment = ggml_backend_get_alignment(backend.get());
    stream_llm::BufferPool pool{4096, 2048, alignment};
    stream_llm::QwenLayerStreamer streamer{index, reader, pool, backend.get()};
    const auto layout = streamer.map(0);
    check(layout.tensors().size() == qwen_tensor_roles.size(), "Qwen tensor map is incomplete");
    check(layout.size() <= pool.buffer_capacity(), "fixture layer exceeds test buffer");
    check_throws([&] { static_cast<void>(streamer.map(1)); }, "out-of-range layer did not fail");

    {
        auto layer = streamer.load(layout);
        check(pool.available() == 1, "bound layer did not retain its lease");
        check(layer.layer() == 0 && layer.weight_bytes() == layout.size(), "bound layer metadata mismatch");
        std::array<float, 4> values{};
        ggml_backend_tensor_get(layer.tensor("attn_q.weight"), values.data(), 0, sizeof(values));
        check(values[0] == 2.0F && values[3] == 2.0F, "bound layer tensor data mismatch");
        check_throws([&] { static_cast<void>(layer.tensor("missing")); }, "missing bound tensor did not fail");
        layer.synchronize();
    }
    check(pool.available() == 2, "layer lease was not returned after synchronization and destruction");

    stream_llm::BufferPool small_pool{64, 64, alignment};
    stream_llm::QwenLayerStreamer small_streamer{index, reader, small_pool, backend.get()};
    check_throws([&] { static_cast<void>(small_streamer.load(layout)); },
                 "undersized layer buffer did not fail");
}

void test_reference_model(const std::filesystem::path & path) {
    stream_llm::GgufIndex index{path};
    check(index.string("general.architecture") == "qwen2", "reference architecture mismatch");
    check(index.u32("qwen2.block_count") == 36, "reference layer count mismatch");
    stream_llm::TensorReader reader{path};
    BackendPtr backend{ggml_backend_cpu_init()};
    check(backend != nullptr, "failed to initialize reference CPU backend");
    constexpr std::size_t layer_budget = 64U * 1024U * 1024U;
    stream_llm::BufferPool pool{layer_budget, layer_budget, ggml_backend_get_alignment(backend.get())};
    stream_llm::QwenLayerStreamer streamer{index, reader, pool, backend.get()};
    for (std::size_t layer_index = 0; layer_index < 36; ++layer_index) {
        const auto mapped = streamer.map(layer_index);
        check(mapped.tensors().size() == qwen_tensor_roles.size(), "reference layer map is incomplete");
        check(mapped.size() <= layer_budget, "reference layer exceeds buffer budget");
    }
    const auto layout = streamer.map(0);
    {
        auto layer = streamer.load(layout);
        check(pool.available() == 0, "reference layer did not retain its lease");
        check(ggml_nbytes(layer.tensor("attn_k.weight")) == index.tensor("blk.0.attn_k.weight").size,
              "reference bound tensor size mismatch");
    }
    check(pool.available() == 1, "reference layer lease was not released");

    stream_llm::Tokenizer tokenizer{path};
    const auto hello_tokens = tokenizer.tokenize("Hello world");
    check(hello_tokens == std::vector<std::int32_t>({9707, 1879}), "reference tokenization mismatch");
    check(tokenizer.detokenize(hello_tokens) == "Hello world", "reference detokenization mismatch");
    check(tokenizer.is_end_of_generation(151645), "Qwen end-of-generation token was not recognized");
    check(!tokenizer.is_end_of_generation(9707), "normal token was marked as end-of-generation");
    check(!tokenizer.is_end_of_generation(-1), "invalid token was marked as end-of-generation");

    const std::vector<stream_llm::ChatMessage> messages{
        {"system", "You are helpful."},
        {"user", "Hello"},
    };
    const auto prompt = tokenizer.apply_chat_template(messages, true);
    check(prompt.find("<|im_start|>system\nYou are helpful.<|im_end|>") != std::string::npos,
          "system message was not formatted by the Qwen chat template");
    check(prompt.ends_with("<|im_start|>assistant\n"), "assistant generation prompt is missing");
    const auto prompt_tokens = tokenizer.tokenize_chat(messages, true);
    check(tokenizer.detokenize(prompt_tokens) == prompt, "chat prompt token round trip mismatch");
    const std::vector<stream_llm::ChatMessage> invalid_messages{{"user", std::string{"x\0y", 3}}};
    check_throws([&] { static_cast<void>(tokenizer.apply_chat_template(invalid_messages)); },
                 "embedded null chat content did not fail");

    stream_llm::QwenExecutor executor{path, layer_budget};
    check_throws([&] { static_cast<void>(executor.decode(-1)); }, "invalid token ID did not fail");
    check(executor.position() == 0, "failed decode changed executor position");
    auto logits = executor.decode(9707);
    check(logits.size() == executor.config().vocabulary_size, "reference logit count mismatch");
    check(static_cast<std::size_t>(std::max_element(logits.begin(), logits.end()) - logits.begin()) == 358,
          "reference greedy token mismatch at position 0");
    logits = executor.decode(1879);
    check(static_cast<std::size_t>(std::max_element(logits.begin(), logits.end()) - logits.begin()) == 0,
          "reference greedy token mismatch at position 1");
    check(executor.position() == 2, "executor position did not advance");
    executor.reset();
    check(executor.position() == 0, "executor reset did not clear position");
    logits = executor.decode(9707);
    check(static_cast<std::size_t>(std::max_element(logits.begin(), logits.end()) - logits.begin()) == 358,
          "executor reset did not clear K/V state");

    stream_llm::GreedyGenerator generator{path, layer_budget};
    const auto zero_result = generator.generate("Hello world", 0);
    check(zero_result.tokens.empty() && zero_result.text.empty(), "zero-token generation produced output");
    check(zero_result.stop_reason == stream_llm::GenerationStopReason::maximum_tokens,
          "zero-token generation reported the wrong stop reason");
    const auto generated = generator.generate("Hello world", 1);
    check(generated.tokens == std::vector<std::int32_t>{0}, "greedy generation selected the wrong token");
    check(generated.text == "!", "generated token was not detokenized");
    check(generated.stop_reason == stream_llm::GenerationStopReason::maximum_tokens,
          "maximum-token stop was not reported");
}

} // namespace

int main(int argc, char ** argv) {
    try {
        test_buffer_pool();
        test_gguf_and_storage();
        test_layer_streaming();
        if (argc == 2) test_reference_model(argv[1]);
        if (argc > 2) throw std::runtime_error("runtime-tests accepts at most one GGUF path");
        std::cout << "runtime interface tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "runtime interface tests failed: " << error.what() << '\n';
        return 1;
    }
}
