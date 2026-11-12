# stream-llm

Run language models beyond your available memory.

`stream-llm` is an experimental C++ inference runtime intended to execute GGUF
language models with explicit, bounded weight residency. The runtime currently
provides GGUF tensor indexing, positional file reads, aligned bounded buffer
leases, streamed Qwen2.5 layer-weight binding, and deterministic token-ID
execution on the CPU backend. Qwen tokenization and chat-template formatting are
available through the runtime API and the CLI supports deterministic greedy text
generation with incremental token output.

## Requirements

- A C++20 compiler
- CMake 3.20 or newer
- A build tool supported by CMake, such as Ninja or Make
- The pinned llama.cpp checkout used by this project

The initial development target is Intel x86-64 macOS. The build setup is kept
portable for future Linux and Windows support.

## Build

The workspace layout used during development places the pinned llama.cpp checkout
at `../upstream/llama.cpp`, which CMake detects automatically. For another layout,
pass the llama.cpp root explicitly:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTREAM_LLM_LLAMA_CPP_DIR=/path/to/llama.cpp
cmake --build build
```

For a debug build:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## Run

```sh
./build/stream-llm \
  --model /path/to/model.gguf \
  --prompt "Hello world" \
  --max-tokens 32
```

Apply the model's embedded chat template for instruction-style prompts:

```sh
./build/stream-llm \
  --model /path/to/model.gguf \
  --chat \
  --system "You are a helpful assistant." \
  --prompt "What is the capital of France?" \
  --max-tokens 32
```

`--system` is optional and is valid only with `--chat`. Without `--chat`, the
prompt is passed to the tokenizer unchanged.

Show the available options without loading a model:

```sh
./build/stream-llm --help
```

## Test

After building:

```sh
ctest --test-dir build --output-on-failure
```

To include the optional index/read smoke test against the reference model:

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DSTREAM_LLM_TEST_MODEL=/Users/apple/Desktop/stream-llm/Models/qwen2.5-3b-instruct-q4_k_m.gguf
cmake --build build
ctest --test-dir build --output-on-failure
```

## Install

Choose an install prefix and install the executable:

```sh
cmake --install build --prefix ./install
./install/bin/stream-llm --help
```
