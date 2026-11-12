#include "stream_llm/qwen_executor.hpp"
#include "stream_llm/tokenizer.hpp"

#include "llama.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t top_count = 5;

struct ModelDeleter {
    void operator()(llama_model * model) const { llama_model_free(model); }
};

struct ContextDeleter {
    void operator()(llama_context * context) const { llama_free(context); }
};

using ModelPtr = std::unique_ptr<llama_model, ModelDeleter>;
using ContextPtr = std::unique_ptr<llama_context, ContextDeleter>;

void discard_log(ggml_log_level, const char *, void *) {}

std::size_t parse_count(std::string_view text) {
    std::size_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0) {
        throw std::invalid_argument("max tokens must be a positive integer");
    }
    return value;
}

std::string quoted_piece(const stream_llm::Tokenizer & tokenizer, std::int32_t token) {
    const auto piece = tokenizer.detokenize(std::span<const std::int32_t>{&token, 1});
    std::string escaped;
    escaped.reserve(piece.size());
    for (const unsigned char byte : piece) {
        switch (byte) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (byte < 0x20 || byte == 0x7f) {
                    constexpr char hex[] = "0123456789abcdef";
                    escaped += "\\x";
                    escaped += hex[byte >> 4U];
                    escaped += hex[byte & 0x0fU];
                } else {
                    escaped += static_cast<char>(byte);
                }
        }
    }
    return escaped;
}

std::vector<std::int32_t> top_tokens(const std::vector<float> & logits) {
    std::vector<std::int32_t> tokens(logits.size());
    std::iota(tokens.begin(), tokens.end(), 0);
    const auto count = std::min(top_count, tokens.size());
    std::partial_sort(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(count), tokens.end(),
                      [&](std::int32_t left, std::int32_t right) {
                          if (logits[left] == logits[right]) return left < right;
                          return logits[left] > logits[right];
                      });
    tokens.resize(count);
    return tokens;
}

void validate_logits(const std::vector<float> & logits, std::string_view runtime) {
    if (logits.empty()) throw std::runtime_error(std::string{runtime} + " returned no logits");
    if (std::any_of(logits.begin(), logits.end(), [](float value) { return !std::isfinite(value); })) {
        throw std::runtime_error(std::string{runtime} + " returned non-finite logits");
    }
}

void print_candidates(std::string_view name,
                      const std::vector<float> & logits,
                      const std::vector<float> & other,
                      const stream_llm::Tokenizer & tokenizer) {
    std::cout << "  " << name << " top " << top_count << ":\n";
    std::size_t rank = 1;
    for (const auto token : top_tokens(logits)) {
        std::cout << "    " << rank++ << ": token=" << token
                  << " piece=\"" << quoted_piece(tokenizer, token) << '"'
                  << " logit=" << logits[token]
                  << " other=" << other[token]
                  << " delta=" << logits[token] - other[token] << '\n';
    }
}

struct CapturedTensor {
    std::vector<float> values;
    std::array<std::int64_t, GGML_MAX_DIMS> dimensions{};
};

using ReferenceCaptures = std::map<std::string, std::vector<CapturedTensor>>;
using StreamCaptures = std::map<std::string, std::vector<float>>;

bool capture_name(std::string_view name) {
    constexpr std::array<std::string_view, 16> prefixes{
        "attn_norm-", "Qcur-", "Kcur-", "Vcur-", "kq-", "kq_soft_max-",
        "kqv_out-", "ffn_inp-", "ffn_norm-", "ffn_up-", "ffn_gate-",
        "ffn_swiglu-", "ffn_out-", "l_out-", "result_norm", "result_output",
    };
    return name == "embd" || std::any_of(prefixes.begin(), prefixes.end(),
        [name](std::string_view prefix) { return name.starts_with(prefix); });
}

struct ReferenceCaptureState {
    ReferenceCaptures tensors;
    std::string error;

    void clear() {
        tensors.clear();
        error.clear();
    }

    static bool callback(ggml_tensor * tensor, bool ask, void * user_data) noexcept {
        auto & state = *static_cast<ReferenceCaptureState *>(user_data);
        const std::string_view name = tensor->name;
        if (!capture_name(name)) return false;
        if (ask) return true;
        try {
            if (!ggml_is_contiguous(tensor)) {
                state.error = "reference checkpoint is not contiguous: " + std::string{name};
                return false;
            }
            CapturedTensor captured;
            std::copy(std::begin(tensor->ne), std::end(tensor->ne), captured.dimensions.begin());
            const auto count = static_cast<std::size_t>(ggml_nelements(tensor));
            captured.values.resize(count);
            if (tensor->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(tensor, captured.values.data(), 0, count * sizeof(float));
            } else if (tensor->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> values(count);
                ggml_backend_tensor_get(tensor, values.data(), 0, count * sizeof(ggml_fp16_t));
                ggml_fp16_to_fp32_row(values.data(), captured.values.data(), static_cast<std::int64_t>(count));
            } else if (tensor->type == GGML_TYPE_BF16) {
                std::vector<ggml_bf16_t> values(count);
                ggml_backend_tensor_get(tensor, values.data(), 0, count * sizeof(ggml_bf16_t));
                for (std::size_t index = 0; index < count; ++index) {
                    captured.values[index] = ggml_bf16_to_fp32(values[index]);
                }
            } else {
                state.error = "unsupported reference checkpoint type: " +
                    std::string{ggml_type_name(tensor->type)};
                return false;
            }
            state.tensors[std::string{name}].push_back(std::move(captured));
            return true;
        } catch (const std::exception & error) {
            state.error = error.what();
            return false;
        } catch (...) {
            state.error = "unknown reference capture failure";
            return false;
        }
    }
};

std::string checkpoint_key(std::string_view name, std::int32_t layer) {
    return layer < 0 ? std::string{name} : std::string{name} + '-' + std::to_string(layer);
}

std::vector<float> fp16_round(std::span<const float> values) {
    std::vector<ggml_fp16_t> half(values.size());
    std::vector<float> result(values.size());
    ggml_fp32_to_fp16_row(values.data(), half.data(), static_cast<std::int64_t>(values.size()));
    ggml_fp16_to_fp32_row(half.data(), result.data(), static_cast<std::int64_t>(values.size()));
    return result;
}

struct Difference {
    double mean_absolute = 0.0;
    float maximum_absolute = 0.0F;
    std::size_t maximum_index = 0;
};

Difference difference(std::span<const float> left, std::span<const float> right) {
    if (left.size() != right.size()) throw std::runtime_error("checkpoint sizes differ");
    Difference result;
    for (std::size_t index = 0; index < left.size(); ++index) {
        const auto value = std::abs(left[index] - right[index]);
        result.mean_absolute += value;
        if (value > result.maximum_absolute) {
            result.maximum_absolute = value;
            result.maximum_index = index;
        }
    }
    if (!left.empty()) result.mean_absolute /= left.size();
    return result;
}

class ReferenceRuntime {
public:
    ReferenceRuntime(const std::filesystem::path & model_path, std::size_t context_tokens) {
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_.reset(llama_model_load_from_file(model_path.c_str(), model_params));
        if (!model_) throw std::runtime_error("failed to load llama.cpp reference model");

        vocab_ = llama_model_get_vocab(model_.get());
        if (!vocab_) throw std::runtime_error("reference model has no vocabulary");
        vocabulary_size_ = llama_vocab_n_tokens(vocab_);
        if (vocabulary_size_ <= 0) throw std::runtime_error("reference vocabulary is empty");
        if (context_tokens > std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("requested trace context is too large");
        }

        auto context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<std::uint32_t>(context_tokens);
        context_params.n_batch = 1;
        context_params.n_ubatch = 1;
        context_params.n_threads = 1;
        context_params.n_threads_batch = 1;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        context_params.offload_kqv = false;
        context_params.no_perf = true;
        context_params.cb_eval = ReferenceCaptureState::callback;
        context_params.cb_eval_user_data = &capture_;
        context_.reset(llama_init_from_model(model_.get(), context_params));
        if (!context_) throw std::runtime_error("failed to create llama.cpp reference context");
    }

    std::vector<float> decode(std::int32_t token) {
        capture_.clear();
        llama_token input = token;
        if (llama_decode(context_.get(), llama_batch_get_one(&input, 1)) != 0) {
            if (!capture_.error.empty()) throw std::runtime_error(capture_.error);
            throw std::runtime_error("llama.cpp reference decode failed");
        }
        if (!capture_.error.empty()) throw std::runtime_error(capture_.error);
        const auto * logits = llama_get_logits_ith(context_.get(), -1);
        if (!logits) throw std::runtime_error("llama.cpp reference returned no logits");
        return {logits, logits + vocabulary_size_};
    }

    bool is_eog(std::int32_t token) const { return llama_vocab_is_eog(vocab_, token); }
    std::size_t vocabulary_size() const { return static_cast<std::size_t>(vocabulary_size_); }
    const ReferenceCaptures & captures() const { return capture_.tensors; }

private:
    ModelPtr model_;
    ContextPtr context_;
    const llama_vocab * vocab_ = nullptr;
    std::int32_t vocabulary_size_ = 0;
    ReferenceCaptureState capture_;
};

class ActivationComparator {
public:
    explicit ActivationComparator(const stream_llm::QwenConfig & config) : config_(config) {}

    void compare(std::size_t position,
                 const StreamCaptures & stream,
                 const ReferenceCaptures & reference) {
        update_expected_cache(reference);
        std::optional<std::string> first_difference;
        compare_checkpoint(position, "embd", -1, stream, reference, first_difference);
        constexpr std::array<std::string_view, 16> layer_checkpoints{
            "attn_norm", "Qcur_pre", "Kcur_pre", "Qcur", "Kcur", "Vcur",
            "cache_k", "cache_v", "kq", "kq_soft_max", "kqv_out", "ffn_inp",
            "ffn_norm", "ffn_up", "ffn_gate", "ffn_swiglu",
        };
        for (std::size_t layer = 0; layer < config_.layer_count; ++layer) {
            const auto layer_index = static_cast<std::int32_t>(layer);
            for (const auto name : layer_checkpoints) {
                compare_checkpoint(position, name, layer_index, stream, reference, first_difference);
            }
            compare_checkpoint(position, "ffn_out", layer_index, stream, reference, first_difference);
            compare_checkpoint(position, "l_out", layer_index, stream, reference, first_difference);
        }
        compare_checkpoint(position, "result_norm", -1, stream, reference, first_difference);
        compare_checkpoint(position, "result_output", -1, stream, reference, first_difference);
        std::cout << "activation_summary position=" << position
                  << " rope_position=" << position
                  << " attention_scale=" << 1.0F / std::sqrt(static_cast<float>(config_.head_dimension));
        if (first_difference) std::cout << " first_difference=" << *first_difference;
        else std::cout << " first_difference=none";
        std::cout << '\n';
    }

private:
    void update_expected_cache(const ReferenceCaptures & reference) {
        for (std::size_t layer = 0; layer < config_.layer_count; ++layer) {
            const auto suffix = '-' + std::to_string(layer);
            append_cache("cache_k" + suffix, reference.at("Kcur" + suffix).back().values);
            append_cache("cache_v" + suffix, reference.at("Vcur" + suffix).back().values);
        }
    }

    void append_cache(const std::string & key, const std::vector<float> & values) {
        const auto rounded = fp16_round(values);
        auto & cache = expected_cache_[key];
        cache.insert(cache.end(), rounded.begin(), rounded.end());
    }

    const std::vector<float> & reference_values(std::string_view name,
                                                std::int32_t layer,
                                                const ReferenceCaptures & reference,
                                                std::vector<float> & extracted) const {
        const auto stream_key = checkpoint_key(name, layer);
        if (name == "cache_k" || name == "cache_v") return expected_cache_.at(stream_key);

        std::string reference_name{name};
        bool pre_rope = false;
        if (name == "Qcur_pre" || name == "Kcur_pre") {
            reference_name.resize(reference_name.size() - 4);
            pre_rope = true;
        }
        const auto key = checkpoint_key(reference_name, layer);
        const auto & captures = reference.at(key);
        if (captures.empty()) throw std::runtime_error("reference checkpoint is empty: " + key);
        const auto capture_index = pre_rope && captures.size() > 1 ? captures.size() - 2 : captures.size() - 1;
        const auto & captured = captures[capture_index];

        if ((name == "kq" || name == "kq_soft_max") &&
            captured.values.size() != config_.head_count * (expected_cache_.at("cache_k-" +
                std::to_string(layer)).size() /
                (config_.kv_head_count * config_.head_dimension))) {
            const auto active_tokens = expected_cache_.at("cache_k-" + std::to_string(layer)).size() /
                (config_.kv_head_count * config_.head_dimension);
            const auto stride = static_cast<std::size_t>(captured.dimensions[0]);
            if (stride < active_tokens || captured.values.size() < stride * config_.head_count) {
                throw std::runtime_error("unexpected reference attention-score shape: " + key);
            }
            extracted.resize(active_tokens * config_.head_count);
            for (std::size_t head = 0; head < config_.head_count; ++head) {
                std::copy_n(captured.values.begin() + static_cast<std::ptrdiff_t>(head * stride),
                            active_tokens,
                            extracted.begin() + static_cast<std::ptrdiff_t>(head * active_tokens));
            }
            return extracted;
        }
        return captured.values;
    }

    void compare_checkpoint(std::size_t position,
                            std::string_view name,
                            std::int32_t layer,
                            const StreamCaptures & stream,
                            const ReferenceCaptures & reference,
                            std::optional<std::string> & first_difference) const {
        const auto key = checkpoint_key(name, layer);
        const auto stream_found = stream.find(key);
        if (stream_found == stream.end()) throw std::runtime_error("missing stream checkpoint: " + key);
        std::vector<float> extracted;
        const auto & expected = reference_values(name, layer, reference, extracted);
        const auto stats = difference(stream_found->second, expected);
        std::cout << "activation position=" << position
                  << " checkpoint=" << key
                  << " size=" << stream_found->second.size()
                  << " mean_abs=" << stats.mean_absolute
                  << " max_abs=" << stats.maximum_absolute
                  << " max_index=" << stats.maximum_index << '\n';
        if (stats.maximum_absolute > 1e-5F && !first_difference) first_difference = key;
    }

    const stream_llm::QwenConfig & config_;
    std::map<std::string, std::vector<float>> expected_cache_;
};

int run(const std::filesystem::path & model_path, std::string_view prompt, std::size_t max_tokens) {
    llama_log_set(discard_log, nullptr);
    stream_llm::Tokenizer tokenizer{model_path};
    const auto prompt_tokens = tokenizer.tokenize(prompt, true, true);
    if (prompt_tokens.empty()) throw std::invalid_argument("prompt produced no tokens");
    if (prompt_tokens.size() > std::numeric_limits<std::size_t>::max() - max_tokens) {
        throw std::length_error("trace context size overflow");
    }

    stream_llm::QwenExecutor stream{model_path};
    ReferenceRuntime reference{model_path, prompt_tokens.size() + max_tokens};
    if (stream.config().vocabulary_size != reference.vocabulary_size()) {
        throw std::runtime_error("runtime vocabulary sizes differ");
    }
    ActivationComparator activation_comparator{stream.config()};

    std::cout << std::setprecision(9)
              << "prompt_tokens=" << prompt_tokens.size() << '\n'
              << "max_generated_tokens=" << max_tokens << '\n';

    std::vector<float> stream_logits;
    std::vector<float> reference_logits;
    const auto decode_pair = [&](std::int32_t token, std::size_t absolute_position) {
        StreamCaptures captures;
        stream_logits = stream.decode(token,
            [&](std::string_view name, std::int32_t layer, std::span<const float> values) {
                captures[checkpoint_key(name, layer)] = {values.begin(), values.end()};
            });
        reference_logits = reference.decode(token);
        activation_comparator.compare(absolute_position, captures, reference.captures());
    };
    for (std::size_t position = 0; position < prompt_tokens.size(); ++position) {
        decode_pair(prompt_tokens[position], position);
    }

    std::optional<std::size_t> first_mismatch;
    for (std::size_t position = 0; position < max_tokens; ++position) {
        validate_logits(stream_logits, "stream-llm");
        validate_logits(reference_logits, "llama.cpp");
        if (stream_logits.size() != reference_logits.size()) {
            throw std::runtime_error("runtime logit counts differ");
        }

        double absolute_sum = 0.0;
        float maximum_absolute = 0.0F;
        std::size_t maximum_index = 0;
        for (std::size_t index = 0; index < stream_logits.size(); ++index) {
            const auto difference = std::abs(stream_logits[index] - reference_logits[index]);
            absolute_sum += difference;
            if (difference > maximum_absolute) {
                maximum_absolute = difference;
                maximum_index = index;
            }
        }

        const auto stream_top = top_tokens(stream_logits).front();
        const auto reference_top = top_tokens(reference_logits).front();
        std::cout << "position=" << position
                  << " absolute_position=" << prompt_tokens.size() + position << '\n'
                  << "  stream_token=" << stream_top
                  << " piece=\"" << quoted_piece(tokenizer, stream_top) << "\"\n"
                  << "  reference_token=" << reference_top
                  << " piece=\"" << quoted_piece(tokenizer, reference_top) << "\"\n"
                  << "  mean_abs_logit_diff=" << absolute_sum / stream_logits.size() << '\n'
                  << "  max_abs_logit_diff=" << maximum_absolute
                  << " token=" << maximum_index << '\n';
        print_candidates("stream-llm", stream_logits, reference_logits, tokenizer);
        print_candidates("llama.cpp", reference_logits, stream_logits, tokenizer);

        if (stream_top != reference_top && !first_mismatch) {
            first_mismatch = position;
            std::cout << "  first_mismatch=yes\n";
        }

        // Teacher-force the reference token so both K/V caches retain an identical token history.
        if (reference.is_eog(reference_top)) {
            std::cout << "reference_eog=yes\n";
            break;
        }
        std::cout << "  teacher_forced_token=" << reference_top
                  << " piece=\"" << quoted_piece(tokenizer, reference_top) << "\"\n";
        decode_pair(reference_top, prompt_tokens.size() + position);
    }

    if (first_mismatch) {
        std::cout << "result=mismatch first_generated_position=" << *first_mismatch << '\n';
        return 1;
    }
    std::cout << "result=match\n";
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: parity-trace MODEL PROMPT [MAX_TOKENS]\n";
        return 2;
    }
    try {
        const auto max_tokens = argc == 4 ? parse_count(argv[3]) : 16U;
        return run(argv[1], argv[2], max_tokens);
    } catch (const std::exception & error) {
        std::cerr << "parity-trace: " << error.what() << '\n';
        return 2;
    }
}
