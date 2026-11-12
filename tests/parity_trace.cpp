#include "stream_llm/qwen_executor.hpp"
#include "stream_llm/tokenizer.hpp"

#include "llama.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
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
        context_params.no_perf = true;
        context_.reset(llama_init_from_model(model_.get(), context_params));
        if (!context_) throw std::runtime_error("failed to create llama.cpp reference context");
    }

    std::vector<float> decode(std::int32_t token) {
        llama_token input = token;
        if (llama_decode(context_.get(), llama_batch_get_one(&input, 1)) != 0) {
            throw std::runtime_error("llama.cpp reference decode failed");
        }
        const auto * logits = llama_get_logits_ith(context_.get(), -1);
        if (!logits) throw std::runtime_error("llama.cpp reference returned no logits");
        return {logits, logits + vocabulary_size_};
    }

    bool is_eog(std::int32_t token) const { return llama_vocab_is_eog(vocab_, token); }
    std::size_t vocabulary_size() const { return static_cast<std::size_t>(vocabulary_size_); }

private:
    ModelPtr model_;
    ContextPtr context_;
    const llama_vocab * vocab_ = nullptr;
    std::int32_t vocabulary_size_ = 0;
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

    std::cout << std::setprecision(9)
              << "prompt_tokens=" << prompt_tokens.size() << '\n'
              << "max_generated_tokens=" << max_tokens << '\n';

    std::vector<float> stream_logits;
    std::vector<float> reference_logits;
    for (const auto token : prompt_tokens) {
        stream_logits = stream.decode(token);
        reference_logits = reference.decode(token);
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
        stream_logits = stream.decode(reference_top);
        reference_logits = reference.decode(reference_top);
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
