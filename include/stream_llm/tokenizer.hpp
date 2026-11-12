#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace stream_llm {

struct ChatMessage {
    std::string role;
    std::string content;
};

class Tokenizer {
public:
    explicit Tokenizer(const std::filesystem::path & model_path);
    ~Tokenizer();

    Tokenizer(const Tokenizer &) = delete;
    Tokenizer & operator=(const Tokenizer &) = delete;
    Tokenizer(Tokenizer &&) noexcept;
    Tokenizer & operator=(Tokenizer &&) noexcept;

    std::vector<std::int32_t> tokenize(std::string_view text,
                                       bool add_special = true,
                                       bool parse_special = true) const;
    std::string detokenize(std::span<const std::int32_t> tokens,
                           bool remove_special = false,
                           bool render_special = true) const;
    std::string apply_chat_template(std::span<const ChatMessage> messages,
                                    bool add_assistant_prompt = true) const;
    std::vector<std::int32_t> tokenize_chat(std::span<const ChatMessage> messages,
                                            bool add_assistant_prompt = true) const;
    bool is_end_of_generation(std::int32_t token) const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace stream_llm
