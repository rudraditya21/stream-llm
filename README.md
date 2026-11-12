# stream-llm

Run language models beyond your available memory.

`stream-llm` is an experimental C++ inference runtime intended to execute GGUF
language models with explicit, bounded weight residency. The runtime currently
provides GGUF tensor indexing, positional file reads, aligned bounded buffer
leases, streamed Qwen2.5 layer-weight binding, and deterministic token-ID
execution on the CPU backend. Qwen tokenization and chat-template formatting are
available through the runtime API; text generation is not yet connected to the
CLI.

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
./build/stream-llm
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
./install/bin/stream-llm
```

## Runtime foundations

- `GgufIndex` parses metadata and creates a validated tensor byte-range index
  without loading tensor payloads.
- `TensorReader` performs bounded positional reads, so concurrent callers do not
  share or mutate a file cursor.
- `BufferPool` enforces a fixed allocation budget and returns move-only leases.
  A buffer cannot be reacquired until its lease is destroyed or moved over.
- A lease keeps its backing pool state alive. Backend work must be synchronized
  before releasing the lease that owns referenced weight bytes.
- `QwenLayerStreamer` maps the 12 tensors required by each dense Qwen2.5 layer,
  packs their byte ranges into one lease, and binds that external memory to ggml
  tensor descriptors.
- `BoundLayer` owns the lease and synchronizes its CPU backend before destruction,
  so referenced weight memory cannot return to the pool while backend work is
  outstanding. The backend must outlive every layer bound to it.
- `QwenExecutor` performs embedding lookup, RMS normalization, Q/K/V projection,
  NEOX RoPE, grouped-query attention, persistent F16 K/V caching, gated FFN
  execution, final normalization, and bounded chunked vocabulary projection.
- `decode(token_id)` returns the complete logit vector and advances the token
  position. `reset()` clears position and all persistent K/V state.
- `Tokenizer` loads only vocabulary metadata through llama.cpp's public API. It
  provides tokenization, detokenization, end-of-generation detection, embedded
  chat-template formatting, and formatted-chat tokenization without loading
  model weights through llama.cpp.
- `GreedyGenerator` resets inference state for each request, evaluates text or
  chat prompts, selects the highest-logit token, stops before emitting an EOG
  token, and enforces the caller's maximum-new-token limit.

The optional reference-model test verifies Qwen tokenization and chat formatting,
then executes all 36 layers, checks greedy token IDs against the pinned llama.cpp
reference, and verifies bounded greedy generation. Sampling and CLI integration
remain later stages.
