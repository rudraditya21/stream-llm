# stream-llm

Run language models beyond your available memory.

`stream-llm` is an experimental C++ inference runtime intended to execute GGUF
language models with explicit, bounded weight residency. The runtime currently
provides GGUF tensor indexing, positional file reads, and aligned bounded buffer
leases. Model execution is not implemented yet.

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
pass its `ggml` directory explicitly:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTREAM_LLM_GGML_DIR=/path/to/llama.cpp/ggml
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

The next stage is mapping Qwen2.5 layer tensors into these leases and binding
them to ggml tensor descriptors.
