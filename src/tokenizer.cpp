#include "stream_llm/tokenizer.hpp"

#include "llama.h"

#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace stream_llm {

namespace {

struct ModelDel {
    void operator()(llama_model * model) const { llama_model_free(model); }
};
using ModelPtr = std::unique_ptr<llama_model, ModelDel>;

void initialize_llama_backend() {
    // shortcut: keep the backend registry for process lifetime, add shared shutdown ownership for library unloading.
    static std::once_flag initialized;
    std::call_once(initialized, llama_backend_init);
}

std::int32_t checked_length(std::size_t size, const char * description) {
    if (size > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::length_error(std::string{description} + " exceeds llama.cpp API limits");
    }
    return static_cast<std::int32_t>(size);
}

std::int32_t required_size(std::int32_t result, const char * description) {
    if (result == std::numeric_limits<std::int32_t>::min()) {
        throw std::length_error(std::string{description} + " exceeds llama.cpp API limits");
    }
    return result < 0 ? -result : result;
}

} // namespace

struct Tokenizer::Impl {
    explicit Impl(const std::filesystem::path & model_path) {
        initialize_llama_backend();
        auto params = llama_model_default_params();
        params.vocab_only = true;
        params.n_gpu_layers = 0;
        model.reset(llama_model_load_from_file(model_path.c_str(), params));
        if (!model) throw std::runtime_error("failed to load tokenizer metadata: " + model_path.string());
        vocab = llama_model_get_vocab(model.get());
        if (!vocab) throw std::runtime_error("model does not contain a tokenizer vocabulary");
        vocabulary_size = llama_vocab_n_tokens(vocab);
        if (vocabulary_size <= 0) throw std::runtime_error("model tokenizer vocabulary is empty");
        chat_template = llama_model_chat_template(model.get(), nullptr);
        if (!chat_template) throw std::runtime_error("model does not contain a supported chat template");
    }

    ModelPtr model;
    const llama_vocab * vocab = nullptr;
    std::int32_t vocabulary_size = 0;
    const char * chat_template = nullptr;
};

Tokenizer::Tokenizer(const std::filesystem::path & model_path)
    : impl_(std::make_unique<Impl>(model_path)) {}

Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer &&) noexcept = default;
Tokenizer & Tokenizer::operator=(Tokenizer &&) noexcept = default;

std::vector<std::int32_t> Tokenizer::tokenize(std::string_view text,
                                              bool add_special,
                                              bool parse_special) const {
    const auto text_length = checked_length(text.size(), "text input");
    const auto first = llama_tokenize(impl_->vocab, text.data(), text_length, nullptr, 0,
                                      add_special, parse_special);
    const auto capacity = required_size(first, "tokenized input");
    if (capacity == 0) return {};
    std::vector<std::int32_t> tokens(static_cast<std::size_t>(capacity));
    const auto count = llama_tokenize(impl_->vocab, text.data(), text_length, tokens.data(), capacity,
                                      add_special, parse_special);
    if (count < 0) throw std::runtime_error("tokenization failed after sizing output");
    tokens.resize(static_cast<std::size_t>(count));
    return tokens;
}

std::string Tokenizer::detokenize(std::span<const std::int32_t> tokens,
                                  bool remove_special,
                                  bool render_special) const {
    const auto token_count = checked_length(tokens.size(), "token input");
    if (token_count == 0) return {};
    const auto first = llama_detokenize(impl_->vocab, tokens.data(), token_count, nullptr, 0,
                                        remove_special, render_special);
    const auto capacity = required_size(first, "detokenized input");
    if (capacity == 0) return {};
    std::string text(static_cast<std::size_t>(capacity), '\0');
    const auto count = llama_detokenize(impl_->vocab, tokens.data(), token_count,
                                        text.data(), capacity, remove_special, render_special);
    if (count < 0) throw std::runtime_error("detokenization failed after sizing output");
    text.resize(static_cast<std::size_t>(count));
    return text;
}

std::string Tokenizer::apply_chat_template(std::span<const ChatMessage> messages,
                                           bool add_assistant_prompt) const {
    std::vector<llama_chat_message> chat;
    chat.reserve(messages.size());
    for (const auto & message : messages) {
        if (message.role.find('\0') != std::string::npos || message.content.find('\0') != std::string::npos) {
            throw std::invalid_argument("chat messages cannot contain embedded null bytes");
        }
        chat.push_back({message.role.c_str(), message.content.c_str()});
    }

    const auto first = llama_chat_apply_template(impl_->chat_template, chat.data(), chat.size(),
                                                 add_assistant_prompt, nullptr, 0);
    if (first < 0) throw std::runtime_error("chat-template application failed");
    if (first == 0) return {};
    std::string formatted(static_cast<std::size_t>(first), '\0');
    const auto count = llama_chat_apply_template(impl_->chat_template, chat.data(), chat.size(),
                                                 add_assistant_prompt, formatted.data(), first);
    if (count < 0 || count > first) throw std::runtime_error("chat-template application failed after sizing output");
    formatted.resize(static_cast<std::size_t>(count));
    return formatted;
}

std::vector<std::int32_t> Tokenizer::tokenize_chat(std::span<const ChatMessage> messages,
                                                   bool add_assistant_prompt) const {
    return tokenize(apply_chat_template(messages, add_assistant_prompt), true, true);
}

bool Tokenizer::is_end_of_generation(std::int32_t token) const noexcept {
    return token >= 0 && token < impl_->vocabulary_size && llama_vocab_is_eog(impl_->vocab, token);
}

} // namespace stream_llm
