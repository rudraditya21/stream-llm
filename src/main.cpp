#include "stream_llm/generator.hpp"

#include "llama.h"

#include <charconv>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view usage =
    "Usage: stream-llm --model PATH --prompt TEXT --max-tokens COUNT [--chat [--system TEXT]]\n";

struct Options {
    std::filesystem::path model;
    std::string prompt;
    std::size_t max_tokens = 0;
    bool chat = false;
    std::optional<std::string> system;
};

void discard_dependency_log(ggml_log_level, const char *, void *) {}

std::string_view option_value(int & index, int argc, char ** argv, std::string_view option) {
    if (++index >= argc) throw std::invalid_argument(std::string{option} + " requires a value");
    return argv[index];
}

Options parse_options(int argc, char ** argv) {
    std::optional<std::filesystem::path> model;
    std::optional<std::string> prompt;
    std::optional<std::size_t> max_tokens;
    std::optional<std::string> system;
    bool chat = false;

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--model") {
            if (model) throw std::invalid_argument("--model cannot be repeated");
            model = option_value(index, argc, argv, argument);
        } else if (argument == "--prompt") {
            if (prompt) throw std::invalid_argument("--prompt cannot be repeated");
            prompt = option_value(index, argc, argv, argument);
        } else if (argument == "--max-tokens") {
            if (max_tokens) throw std::invalid_argument("--max-tokens cannot be repeated");
            const auto value = option_value(index, argc, argv, argument);
            std::size_t parsed = 0;
            const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if (error != std::errc{} || end != value.data() + value.size() || parsed == 0) {
                throw std::invalid_argument("--max-tokens must be a positive integer");
            }
            max_tokens = parsed;
        } else if (argument == "--chat") {
            if (chat) throw std::invalid_argument("--chat cannot be repeated");
            chat = true;
        } else if (argument == "--system") {
            if (system) throw std::invalid_argument("--system cannot be repeated");
            system = option_value(index, argc, argv, argument);
        } else {
            throw std::invalid_argument("unknown option: " + std::string{argument});
        }
    }

    if (!model || !prompt || !max_tokens) {
        throw std::invalid_argument("--model, --prompt, and --max-tokens are required");
    }
    if (prompt->empty()) throw std::invalid_argument("--prompt cannot be empty");
    if (system && !chat) throw std::invalid_argument("--system requires --chat");
    if (!std::filesystem::is_regular_file(*model)) {
        throw std::invalid_argument("--model must name an existing file");
    }
    return {*model, *prompt, *max_tokens, chat, system};
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::string_view{argv[1]} == "--help") {
        std::cout << usage;
        return 0;
    }

    try {
        const auto options = parse_options(argc, argv);
        llama_log_set(discard_dependency_log, nullptr);
        stream_llm::GreedyGenerator generator{options.model};
        const auto stream_piece = [](std::string_view piece) {
            std::cout << piece << std::flush;
        };
        if (options.chat) {
            std::vector<stream_llm::ChatMessage> messages;
            if (options.system) messages.push_back({"system", *options.system});
            messages.push_back({"user", options.prompt});
            static_cast<void>(generator.generate_chat(messages, options.max_tokens, stream_piece));
        } else {
            static_cast<void>(generator.generate(options.prompt, options.max_tokens, stream_piece));
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "stream-llm: " << error.what() << '\n' << usage;
        return 1;
    }
}
