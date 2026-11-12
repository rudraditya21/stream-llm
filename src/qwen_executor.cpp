#include "stream_llm/qwen_executor.hpp"

#include "stream_llm/buffer_pool.hpp"
#include "stream_llm/gguf_index.hpp"
#include "stream_llm/qwen_layer.hpp"
#include "stream_llm/tensor_reader.hpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stream_llm {

namespace {

constexpr std::size_t graph_size = 128;

struct BackendDel {
    void operator()(ggml_backend * backend) const { ggml_backend_free(backend); }
};
struct ContextDel {
    void operator()(ggml_context * context) const { ggml_free(context); }
};
struct BackendBufferDel {
    void operator()(ggml_backend_buffer * buffer) const { ggml_backend_buffer_free(buffer); }
};
using BackendPtr = std::unique_ptr<ggml_backend, BackendDel>;
using ContextPtr = std::unique_ptr<ggml_context, ContextDel>;
using BackendBufferPtr = std::unique_ptr<ggml_backend_buffer, BackendBufferDel>;

[[noreturn]] void fail(const std::string & message) { throw std::runtime_error(message); }

void emit(const QwenTraceCallback & trace, std::string_view name, std::int32_t layer,
          const std::vector<float> & values) {
    if (trace) trace(name, layer, values);
}

BackendPtr make_backend() {
    BackendPtr backend{ggml_backend_cpu_init()};
    if (!backend) fail("failed to initialize CPU backend");
    ggml_backend_cpu_set_n_threads(backend.get(), 1);
    return backend;
}

ContextPtr make_context(std::size_t tensors = 128) {
    const std::size_t bytes = tensors * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(graph_size, false) + 4096;
    ContextPtr context{ggml_init({bytes, nullptr, true})};
    if (!context) fail("failed to create ggml compute context");
    return context;
}

void set_tensor(ggml_tensor * tensor, const void * data, std::size_t bytes) {
    if (ggml_nbytes(tensor) != bytes) fail("ggml input byte size mismatch");
    ggml_backend_tensor_set(tensor, data, 0, bytes);
}

std::vector<float> read_f32(ggml_tensor * tensor) {
    if (tensor->type != GGML_TYPE_F32) fail("expected an F32 output tensor");
    std::vector<float> values(static_cast<std::size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

class ExternalTensor {
public:
    ExternalTensor(const TensorReader & reader, BufferPool & pool, ggml_backend_t backend,
                   const TensorInfo & source, std::uint64_t file_offset,
                   std::size_t rows, std::size_t bytes)
        : backend_(backend), lease_(pool.acquire(bytes)) {
        if (source.rank != 1 && source.rank != 2) fail("external tensor must have one or two dimensions");
        reader.read_at(file_offset, lease_.bytes());
        const auto type = static_cast<ggml_type>(source.type);
        if (!ggml_validate_row_data(type, lease_.data(), bytes)) fail("invalid tensor bytes: " + source.name);

        wrapper_.reset(ggml_backend_cpu_buffer_from_ptr(lease_.data(), lease_.size()));
        if (!wrapper_) fail("failed to wrap external tensor: " + source.name);
        ggml_backend_buffer_set_usage(wrapper_.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        context_ = make_context(8);
        if (source.rank == 1) {
            tensor_ = ggml_new_tensor_1d(context_.get(), type, source.dimensions[0]);
        } else {
            tensor_ = ggml_new_tensor_2d(context_.get(), type, source.dimensions[0], rows);
        }
        ggml_set_name(tensor_, source.name.c_str());
        if (ggml_backend_tensor_alloc(wrapper_.get(), tensor_, lease_.data()) != GGML_STATUS_SUCCESS) {
            fail("failed to bind external tensor: " + source.name);
        }
    }

    ~ExternalTensor() { ggml_backend_synchronize(backend_); }
    ExternalTensor(const ExternalTensor &) = delete;
    ExternalTensor & operator=(const ExternalTensor &) = delete;

    ggml_tensor * get() const noexcept { return tensor_; }

private:
    ggml_backend_t backend_;
    BufferLease lease_;
    BackendBufferPtr wrapper_;
    ContextPtr context_;
    ggml_tensor * tensor_ = nullptr;
};

ExternalTensor load_tensor(const TensorReader & reader, BufferPool & pool,
                           ggml_backend_t backend, const TensorInfo & source) {
    return ExternalTensor{reader, pool, backend, source, source.file_offset,
                          static_cast<std::size_t>(source.dimensions[1]), source.size};
}

ExternalTensor load_rows(const TensorReader & reader, BufferPool & pool,
                         ggml_backend_t backend, const TensorInfo & source,
                         std::size_t first_row, std::size_t rows) {
    if (source.rank != 2 || source.dimensions[0] <= 0 || source.dimensions[1] <= 0) {
        fail("row streaming requires a two-dimensional tensor: " + source.name);
    }
    const auto row_count = static_cast<std::size_t>(source.dimensions[1]);
    if (first_row > row_count || rows > row_count - first_row) fail("tensor row range is out of bounds");
    const auto type = static_cast<ggml_type>(source.type);
    const std::size_t row_bytes = ggml_row_size(type, source.dimensions[0]);
    if (rows > std::numeric_limits<std::size_t>::max() / row_bytes) fail("tensor row byte size overflow");
    if (first_row > (std::numeric_limits<std::uint64_t>::max() - source.file_offset) / row_bytes) {
        fail("tensor row offset overflow");
    }
    return ExternalTensor{reader, pool, backend, source,
                          source.file_offset + first_row * row_bytes, rows, rows * row_bytes};
}

QwenConfig read_config(const GgufIndex & index) {
    QwenConfig config{
        index.u32("qwen2.embedding_length"),
        index.u32("qwen2.feed_forward_length"),
        index.u32("qwen2.block_count"),
        index.u32("qwen2.attention.head_count"),
        index.u32("qwen2.attention.head_count_kv"),
        0,
        0,
        index.f32("qwen2.attention.layer_norm_rms_epsilon"),
        index.f32("qwen2.rope.freq_base"),
    };
    if (config.head_count == 0 || config.kv_head_count == 0 ||
        config.layer_count == 0 || config.feed_forward_length == 0 ||
        config.embedding_length % config.head_count != 0 ||
        config.head_count % config.kv_head_count != 0 || config.rms_epsilon <= 0.0F ||
        config.rope_base <= 0.0F) {
        fail("unsupported Qwen attention dimensions");
    }
    config.head_dimension = config.embedding_length / config.head_count;
    if (config.head_dimension % 2 != 0) fail("Qwen RoPE requires an even head dimension");
    const auto & embedding = index.tensor("token_embd.weight");
    if (embedding.rank != 2 || embedding.dimensions[0] != static_cast<std::int64_t>(config.embedding_length) ||
        embedding.dimensions[1] <= 0) {
        fail("token embedding shape does not match Qwen configuration");
    }
    config.vocabulary_size = static_cast<std::size_t>(embedding.dimensions[1]);
    if (config.vocabulary_size == 0) fail("Qwen vocabulary is empty");
    return config;
}

void expect_shape(const TensorInfo & tensor, std::int64_t first, std::int64_t second = 0) {
    const std::uint32_t rank = second == 0 ? 1U : 2U;
    if (tensor.rank != rank || tensor.dimensions[0] != first ||
        (rank == 2 && tensor.dimensions[1] != second)) {
        fail("unexpected tensor shape: " + tensor.name);
    }
}

const TensorInfo & layer_tensor(const QwenLayerLayout & layout, std::string_view role) {
    const auto found = std::find_if(layout.tensors().begin(), layout.tensors().end(),
        [role](const LayerTensorRange & tensor) { return tensor.role == role; });
    if (found == layout.tensors().end()) fail("missing mapped layer tensor: " + std::string{role});
    return found->source;
}

void validate_layer(const QwenLayerLayout & layout, const QwenConfig & config) {
    const auto embedding = static_cast<std::int64_t>(config.embedding_length);
    const auto kv_width = static_cast<std::int64_t>(config.kv_head_count * config.head_dimension);
    const auto feed_forward = static_cast<std::int64_t>(config.feed_forward_length);
    expect_shape(layer_tensor(layout, "attn_norm.weight"), embedding);
    expect_shape(layer_tensor(layout, "attn_q.weight"), embedding, embedding);
    expect_shape(layer_tensor(layout, "attn_q.bias"), embedding);
    expect_shape(layer_tensor(layout, "attn_k.weight"), embedding, kv_width);
    expect_shape(layer_tensor(layout, "attn_k.bias"), kv_width);
    expect_shape(layer_tensor(layout, "attn_v.weight"), embedding, kv_width);
    expect_shape(layer_tensor(layout, "attn_v.bias"), kv_width);
    expect_shape(layer_tensor(layout, "attn_output.weight"), embedding, embedding);
    expect_shape(layer_tensor(layout, "ffn_norm.weight"), embedding);
    expect_shape(layer_tensor(layout, "ffn_gate.weight"), embedding, feed_forward);
    expect_shape(layer_tensor(layout, "ffn_up.weight"), embedding, feed_forward);
    expect_shape(layer_tensor(layout, "ffn_down.weight"), feed_forward, embedding);
}

struct Projection {
    std::vector<float> normalized;
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
};

Projection project(const BoundLayer & layer, const QwenConfig & config,
                   const std::vector<float> & hidden, ggml_backend_t backend,
                   std::int32_t layer_index, const QwenTraceCallback & trace) {
    auto context = make_context();
    auto * input = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, config.embedding_length);
    auto * norm = ggml_rms_norm(context.get(), input, config.rms_epsilon);
    norm = ggml_mul(context.get(), norm, layer.tensor("attn_norm.weight"));
    auto * query = ggml_add(context.get(),
        ggml_mul_mat(context.get(), layer.tensor("attn_q.weight"), norm), layer.tensor("attn_q.bias"));
    auto * key = ggml_add(context.get(),
        ggml_mul_mat(context.get(), layer.tensor("attn_k.weight"), norm), layer.tensor("attn_k.bias"));
    auto * value = ggml_add(context.get(),
        ggml_mul_mat(context.get(), layer.tensor("attn_v.weight"), norm), layer.tensor("attn_v.bias"));
    auto * graph = ggml_new_graph_custom(context.get(), graph_size, false);
    ggml_build_forward_expand(graph, query);
    ggml_build_forward_expand(graph, key);
    ggml_build_forward_expand(graph, value);
    BackendBufferPtr buffer{ggml_backend_alloc_ctx_tensors(context.get(), backend)};
    if (!buffer) fail("failed to allocate Q/K/V graph");
    set_tensor(input, hidden.data(), hidden.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) fail("Q/K/V projection failed");
    Projection result{read_f32(norm), read_f32(query), read_f32(key), read_f32(value)};
    emit(trace, "attn_norm", layer_index, result.normalized);
    emit(trace, "Qcur_pre", layer_index, result.query);
    emit(trace, "Kcur_pre", layer_index, result.key);
    emit(trace, "Vcur", layer_index, result.value);
    return result;
}

void apply_rope(std::vector<float> & values, std::size_t heads, std::size_t head_dimension,
                std::size_t position, float base) {
    const std::size_t half = head_dimension / 2;
    for (std::size_t head_index = 0; head_index < heads; ++head_index) {
        float * head = values.data() + head_index * head_dimension;
        for (std::size_t index = 0; index < half; ++index) {
            const double frequency = std::pow(base, -2.0 * static_cast<double>(index) /
                                                     static_cast<double>(head_dimension));
            const double angle = static_cast<double>(position) * frequency;
            const float first = head[index];
            const float second = head[index + half];
            head[index] = static_cast<float>(first * std::cos(angle) - second * std::sin(angle));
            head[index + half] = static_cast<float>(first * std::sin(angle) + second * std::cos(angle));
        }
    }
}

struct LayerCache {
    std::vector<ggml_fp16_t> keys;
    std::vector<ggml_fp16_t> values;
    std::size_t tokens = 0;

    void append(const std::vector<float> & key, const std::vector<float> & value) {
        if (key.size() != value.size()) fail("K/V cache shape mismatch");
        const auto old_size = keys.size();
        keys.resize(old_size + key.size());
        values.resize(old_size + value.size());
        ggml_fp32_to_fp16_row(key.data(), keys.data() + old_size, static_cast<std::int64_t>(key.size()));
        ggml_fp32_to_fp16_row(value.data(), values.data() + old_size, static_cast<std::int64_t>(value.size()));
        ++tokens;
    }

    void clear() {
        keys.clear();
        values.clear();
        tokens = 0;
    }

    void truncate(std::size_t token_count, std::size_t width) {
        keys.resize(token_count * width);
        values.resize(token_count * width);
        tokens = token_count;
    }

    std::vector<float> key_values() const {
        std::vector<float> result(keys.size());
        ggml_fp16_to_fp32_row(keys.data(), result.data(), static_cast<std::int64_t>(keys.size()));
        return result;
    }

    std::vector<float> value_values() const {
        std::vector<float> result(values.size());
        ggml_fp16_to_fp32_row(values.data(), result.data(), static_cast<std::int64_t>(values.size()));
        return result;
    }
};

std::vector<float> attend(const std::vector<float> & query, const LayerCache & cache,
                          const QwenConfig & config, std::int32_t layer_index,
                          const QwenTraceCallback & trace) {
    if (cache.tokens == 0 || query.size() != config.embedding_length) fail("invalid attention state");
    const std::size_t kv_width = config.kv_head_count * config.head_dimension;
    if (cache.keys.size() != cache.tokens * kv_width || cache.values.size() != cache.keys.size()) {
        fail("invalid K/V cache storage");
    }
    const std::size_t group_size = config.head_count / config.kv_head_count;
    const float scale = 1.0F / std::sqrt(static_cast<float>(config.head_dimension));
    std::vector<float> output(config.embedding_length, 0.0F);
    std::vector<float> scores(cache.tokens);
    std::vector<float> raw_scores;
    std::vector<float> probabilities;
    if (trace) {
        raw_scores.resize(config.head_count * cache.tokens);
        probabilities.resize(raw_scores.size());
    }

    for (std::size_t query_head = 0; query_head < config.head_count; ++query_head) {
        const std::size_t kv_head = query_head / group_size;
        for (std::size_t token = 0; token < cache.tokens; ++token) {
            float dot = 0.0F;
            const std::size_t cache_base = token * kv_width + kv_head * config.head_dimension;
            const std::size_t query_base = query_head * config.head_dimension;
            for (std::size_t dimension = 0; dimension < config.head_dimension; ++dimension) {
                dot += query[query_base + dimension] * ggml_fp16_to_fp32(cache.keys[cache_base + dimension]);
            }
            if (trace) raw_scores[query_head * cache.tokens + token] = dot;
            scores[token] = dot * scale;
        }
        const float maximum = *std::max_element(scores.begin(), scores.end());
        float denominator = 0.0F;
        for (float & score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (std::size_t token = 0; token < cache.tokens; ++token) {
            const float probability = scores[token] / denominator;
            if (trace) probabilities[query_head * cache.tokens + token] = probability;
            const std::size_t cache_base = token * kv_width + kv_head * config.head_dimension;
            const std::size_t output_base = query_head * config.head_dimension;
            for (std::size_t dimension = 0; dimension < config.head_dimension; ++dimension) {
                output[output_base + dimension] +=
                    probability * ggml_fp16_to_fp32(cache.values[cache_base + dimension]);
            }
        }
    }
    emit(trace, "kq", layer_index, raw_scores);
    emit(trace, "kq_soft_max", layer_index, probabilities);
    return output;
}

std::vector<float> finish_layer(const BoundLayer & layer, const QwenConfig & config,
                                const std::vector<float> & hidden,
                                const std::vector<float> & attention,
                                ggml_backend_t backend, std::int32_t layer_index,
                                const QwenTraceCallback & trace) {
    auto context = make_context();
    auto * hidden_input = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, config.embedding_length);
    auto * attention_input = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, config.embedding_length);
    auto * projected = ggml_mul_mat(context.get(), layer.tensor("attn_output.weight"), attention_input);
    auto * residual = ggml_add(context.get(), hidden_input, projected);
    auto * norm = ggml_rms_norm(context.get(), residual, config.rms_epsilon);
    norm = ggml_mul(context.get(), norm, layer.tensor("ffn_norm.weight"));
    auto * gate = ggml_mul_mat(context.get(), layer.tensor("ffn_gate.weight"), norm);
    auto * up = ggml_mul_mat(context.get(), layer.tensor("ffn_up.weight"), norm);
    auto * activated = ggml_mul(context.get(), ggml_silu(context.get(), gate), up);
    auto * down = ggml_mul_mat(context.get(), layer.tensor("ffn_down.weight"), activated);
    auto * output = ggml_add(context.get(), residual, down);
    auto * graph = ggml_new_graph_custom(context.get(), graph_size, false);
    ggml_build_forward_expand(graph, output);
    BackendBufferPtr buffer{ggml_backend_alloc_ctx_tensors(context.get(), backend)};
    if (!buffer) fail("failed to allocate layer graph");
    set_tensor(hidden_input, hidden.data(), hidden.size() * sizeof(float));
    set_tensor(attention_input, attention.data(), attention.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) fail("layer graph failed");
    auto result = read_f32(output);
    if (trace) {
        emit(trace, "attn_projected", layer_index, read_f32(projected));
        emit(trace, "ffn_inp", layer_index, read_f32(residual));
        emit(trace, "ffn_norm", layer_index, read_f32(norm));
        emit(trace, "ffn_gate", layer_index, read_f32(gate));
        emit(trace, "ffn_up", layer_index, read_f32(up));
        emit(trace, "ffn_swiglu", layer_index, read_f32(activated));
        emit(trace, "ffn_out", layer_index, read_f32(down));
        emit(trace, "l_out", layer_index, result);
    }
    return result;
}

} // namespace

struct QwenExecutor::Impl {
    Impl(const std::filesystem::path & path, std::size_t capacity)
        : index(path),
          reader(path),
          backend(make_backend()),
          config(read_config(index)),
          pool(capacity, capacity, ggml_backend_get_alignment(backend.get())),
          streamer(index, reader, pool, backend.get()),
          caches(config.layer_count) {
        for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
            const auto layout = streamer.map(layer);
            validate_layer(layout, config);
            if (layout.size() > capacity) fail("Qwen layer exceeds weight buffer capacity");
        }
        const auto & embedding = index.tensor("token_embd.weight");
        const auto embedding_row = ggml_row_size(static_cast<ggml_type>(embedding.type), embedding.dimensions[0]);
        const auto & output = index.tensor("output.weight");
        expect_shape(output, static_cast<std::int64_t>(config.embedding_length),
                     static_cast<std::int64_t>(config.vocabulary_size));
        expect_shape(index.tensor("output_norm.weight"), static_cast<std::int64_t>(config.embedding_length));
        const auto output_row = ggml_row_size(static_cast<ggml_type>(output.type), output.dimensions[0]);
        if (embedding_row > capacity || output_row > capacity || index.tensor("output_norm.weight").size > capacity) {
            fail("global Qwen tensor operation exceeds weight buffer capacity");
        }
    }

    std::vector<float> embedding(std::int32_t token) {
        if (token < 0 || static_cast<std::size_t>(token) >= config.vocabulary_size) {
            throw std::out_of_range("token ID is outside the Qwen vocabulary");
        }
        const auto & source = index.tensor("token_embd.weight");
        auto row = load_rows(reader, pool, backend.get(), source, static_cast<std::size_t>(token), 1);
        auto context = make_context(16);
        auto * indices = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, 1);
        auto * output = ggml_get_rows(context.get(), row.get(), indices);
        auto * graph = ggml_new_graph_custom(context.get(), graph_size, false);
        ggml_build_forward_expand(graph, output);
        BackendBufferPtr buffer{ggml_backend_alloc_ctx_tensors(context.get(), backend.get())};
        if (!buffer) fail("failed to allocate embedding graph");
        const std::int32_t row_index = 0;
        set_tensor(indices, &row_index, sizeof(row_index));
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) fail("embedding lookup failed");
        return read_f32(output);
    }

    std::vector<float> normalize(const std::vector<float> & hidden) {
        const auto & source = index.tensor("output_norm.weight");
        auto weight = load_tensor(reader, pool, backend.get(), source);
        auto context = make_context(16);
        auto * input = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, config.embedding_length);
        auto * output = ggml_mul(context.get(),
            ggml_rms_norm(context.get(), input, config.rms_epsilon), weight.get());
        auto * graph = ggml_new_graph_custom(context.get(), graph_size, false);
        ggml_build_forward_expand(graph, output);
        BackendBufferPtr buffer{ggml_backend_alloc_ctx_tensors(context.get(), backend.get())};
        if (!buffer) fail("failed to allocate final normalization graph");
        set_tensor(input, hidden.data(), hidden.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) fail("final normalization failed");
        return read_f32(output);
    }

    std::vector<float> logits(const std::vector<float> & normalized) {
        const auto & source = index.tensor("output.weight");
        if (source.rank != 2 || source.dimensions[0] != static_cast<std::int64_t>(config.embedding_length) ||
            source.dimensions[1] != static_cast<std::int64_t>(config.vocabulary_size)) {
            fail("output weight shape does not match Qwen configuration");
        }
        const std::size_t row_bytes = ggml_row_size(static_cast<ggml_type>(source.type), source.dimensions[0]);
        const std::size_t rows_per_chunk = pool.buffer_capacity() / row_bytes;
        if (rows_per_chunk == 0) fail("output row exceeds weight buffer capacity");
        std::vector<float> result(config.vocabulary_size);

        for (std::size_t first = 0; first < config.vocabulary_size; first += rows_per_chunk) {
            const std::size_t rows = std::min(rows_per_chunk, config.vocabulary_size - first);
            auto weight = load_rows(reader, pool, backend.get(), source, first, rows);
            auto context = make_context(16);
            auto * input = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, config.embedding_length);
            auto * output = ggml_mul_mat(context.get(), weight.get(), input);
            auto * graph = ggml_new_graph_custom(context.get(), graph_size, false);
            ggml_build_forward_expand(graph, output);
            BackendBufferPtr buffer{ggml_backend_alloc_ctx_tensors(context.get(), backend.get())};
            if (!buffer) fail("failed to allocate output projection graph");
            set_tensor(input, normalized.data(), normalized.size() * sizeof(float));
            if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
                fail("output projection failed");
            }
            const auto chunk = read_f32(output);
            std::copy(chunk.begin(), chunk.end(), result.begin() + static_cast<std::ptrdiff_t>(first));
        }

        if (const auto * bias_source = index.find_tensor("output.bias")) {
            auto bias = load_tensor(reader, pool, backend.get(), *bias_source);
            if (bias.get()->type != GGML_TYPE_F32 ||
                static_cast<std::size_t>(ggml_nelements(bias.get())) != result.size()) {
                fail("unsupported output bias tensor");
            }
            std::vector<float> values(result.size());
            ggml_backend_tensor_get(bias.get(), values.data(), 0, values.size() * sizeof(float));
            for (std::size_t i = 0; i < result.size(); ++i) result[i] += values[i];
        }
        return result;
    }

    GgufIndex index;
    TensorReader reader;
    BackendPtr backend;
    QwenConfig config;
    BufferPool pool;
    QwenLayerStreamer streamer;
    std::vector<LayerCache> caches;
    std::size_t position = 0;
};

QwenExecutor::QwenExecutor(const std::filesystem::path & model_path, std::size_t weight_buffer_capacity)
    : impl_(std::make_unique<Impl>(model_path, weight_buffer_capacity)) {}

QwenExecutor::~QwenExecutor() = default;
QwenExecutor::QwenExecutor(QwenExecutor &&) noexcept = default;
QwenExecutor & QwenExecutor::operator=(QwenExecutor &&) noexcept = default;

const QwenConfig & QwenExecutor::config() const noexcept { return impl_->config; }
std::size_t QwenExecutor::position() const noexcept { return impl_->position; }

std::vector<float> QwenExecutor::decode(std::int32_t token) {
    return decode(token, {});
}

std::vector<float> QwenExecutor::decode(std::int32_t token, const QwenTraceCallback & trace) {
    const std::size_t previous_position = impl_->position;
    const std::size_t kv_width = impl_->config.kv_head_count * impl_->config.head_dimension;
    if (!std::all_of(impl_->caches.begin(), impl_->caches.end(),
                     [previous_position](const LayerCache & cache) { return cache.tokens == previous_position; })) {
        fail("Qwen K/V caches are inconsistent with the token position");
    }
    try {
        auto hidden = impl_->embedding(token);
        emit(trace, "embd", -1, hidden);
        for (std::size_t layer_index = 0; layer_index < impl_->config.layer_count; ++layer_index) {
            const auto layout = impl_->streamer.map(layer_index);
            auto layer = impl_->streamer.load(layout);
            const auto trace_layer = static_cast<std::int32_t>(layer_index);
            auto projection = project(layer, impl_->config, hidden, impl_->backend.get(), trace_layer, trace);
            apply_rope(projection.query, impl_->config.head_count, impl_->config.head_dimension,
                       impl_->position, impl_->config.rope_base);
            apply_rope(projection.key, impl_->config.kv_head_count, impl_->config.head_dimension,
                       impl_->position, impl_->config.rope_base);
            emit(trace, "Qcur", trace_layer, projection.query);
            emit(trace, "Kcur", trace_layer, projection.key);
            impl_->caches[layer_index].append(projection.key, projection.value);
            if (trace) {
                emit(trace, "cache_k", trace_layer, impl_->caches[layer_index].key_values());
                emit(trace, "cache_v", trace_layer, impl_->caches[layer_index].value_values());
            }
            const auto attention = attend(projection.query, impl_->caches[layer_index], impl_->config,
                                          trace_layer, trace);
            emit(trace, "kqv_out", trace_layer, attention);
            hidden = finish_layer(layer, impl_->config, hidden, attention, impl_->backend.get(),
                                  trace_layer, trace);
        }
        const auto normalized = impl_->normalize(hidden);
        emit(trace, "result_norm", -1, normalized);
        auto result = impl_->logits(normalized);
        emit(trace, "result_output", -1, result);
        ++impl_->position;
        return result;
    } catch (...) {
        for (auto & cache : impl_->caches) cache.truncate(previous_position, kv_width);
        throw;
    }
}

void QwenExecutor::reset() {
    for (auto & cache : impl_->caches) cache.clear();
    impl_->position = 0;
}

} // namespace stream_llm
