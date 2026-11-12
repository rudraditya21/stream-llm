#pragma once

#include "stream_llm/tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace stream_llm {

using TokenCallback = std::function<void(std::string_view)>;

enum class GenerationStopReason {
    end_of_generation,
    maximum_tokens,
};

struct GenerationResult {
    std::string text;
    std::vector<std::int32_t> tokens;
    GenerationStopReason stop_reason;
};

class GreedyGenerator {
public:
    explicit GreedyGenerator(const std::filesystem::path & model_path,
                             std::size_t weight_buffer_capacity = 64U * 1024U * 1024U);
    ~GreedyGenerator();

    GreedyGenerator(const GreedyGenerator &) = delete;
    GreedyGenerator & operator=(const GreedyGenerator &) = delete;
    GreedyGenerator(GreedyGenerator &&) noexcept;
    GreedyGenerator & operator=(GreedyGenerator &&) noexcept;

    GenerationResult generate(std::string_view prompt, std::size_t max_new_tokens,
                              TokenCallback on_token = {});
    GenerationResult generate_chat(std::span<const ChatMessage> messages,
                                   std::size_t max_new_tokens,
                                   TokenCallback on_token = {});

private:
    GenerationResult generate_formatted(std::string_view prompt, std::size_t max_new_tokens,
                                        const TokenCallback & on_token);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace stream_llm
