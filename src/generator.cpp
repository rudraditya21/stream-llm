#include "stream_llm/generator.hpp"

#include "stream_llm/qwen_executor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace stream_llm {

namespace {

std::int32_t greedy_token(const std::vector<float> & logits) {
    if (logits.empty()) throw std::runtime_error("cannot decode an empty logit vector");
    if (std::any_of(logits.begin(), logits.end(), [](float value) { return !std::isfinite(value); })) {
        throw std::runtime_error("cannot decode non-finite logits");
    }
    // shortcut: generation is greedy-only, add a sampling policy when sampling is implemented.
    return static_cast<std::int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

} // namespace

struct GreedyGenerator::Impl {
    Impl(const std::filesystem::path & model_path, std::size_t weight_buffer_capacity)
        : tokenizer(model_path), executor(model_path, weight_buffer_capacity) {}

    Tokenizer tokenizer;
    QwenExecutor executor;
};

GreedyGenerator::GreedyGenerator(const std::filesystem::path & model_path,
                                 std::size_t weight_buffer_capacity)
    : impl_(std::make_unique<Impl>(model_path, weight_buffer_capacity)) {}

GreedyGenerator::~GreedyGenerator() = default;
GreedyGenerator::GreedyGenerator(GreedyGenerator &&) noexcept = default;
GreedyGenerator & GreedyGenerator::operator=(GreedyGenerator &&) noexcept = default;

GenerationResult GreedyGenerator::generate(std::string_view prompt, std::size_t max_new_tokens) {
    return generate_formatted(prompt, max_new_tokens);
}

GenerationResult GreedyGenerator::generate_chat(std::span<const ChatMessage> messages,
                                                 std::size_t max_new_tokens) {
    return generate_formatted(impl_->tokenizer.apply_chat_template(messages, true), max_new_tokens);
}

GenerationResult GreedyGenerator::generate_formatted(std::string_view prompt,
                                                      std::size_t max_new_tokens) {
    impl_->executor.reset();
    GenerationResult result{{}, {}, GenerationStopReason::maximum_tokens};
    if (max_new_tokens == 0) return result;

    const auto prompt_tokens = impl_->tokenizer.tokenize(prompt, true, true);
    if (prompt_tokens.empty()) throw std::invalid_argument("generation prompt produced no tokens");

    std::vector<float> logits;
    for (const auto token : prompt_tokens) logits = impl_->executor.decode(token);

    result.tokens.reserve(max_new_tokens);
    for (std::size_t count = 0; count < max_new_tokens; ++count) {
        const auto token = greedy_token(logits);
        if (impl_->tokenizer.is_end_of_generation(token)) {
            result.stop_reason = GenerationStopReason::end_of_generation;
            break;
        }
        result.tokens.push_back(token);
        if (count + 1 < max_new_tokens) logits = impl_->executor.decode(token);
    }
    result.text = impl_->tokenizer.detokenize(result.tokens, false, false);
    return result;
}

} // namespace stream_llm
