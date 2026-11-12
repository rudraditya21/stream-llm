# stream-llm

Run language models beyond your available memory.

`stream-llm` is an experimental C++ inference runtime intended to execute GGUF
language models with explicit, bounded weight residency. The project is currently
at the bootstrap stage: the build and test setup works, but model inference is not
implemented yet.

## Requirements

- A C++20 compiler
- CMake 3.20 or newer
- A build tool supported by CMake, such as Ninja or Make

The initial development target is Intel x86-64 macOS. The build setup is kept
portable for future Linux and Windows support.

## Build

From the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
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

## Install

Choose an install prefix and install the executable:

```sh
cmake --install build --prefix ./install
./install/bin/stream-llm
```

## Project status

The next engineering stage is a feasibility study of ggml and llama.cpp model
loading, tensor allocation, graph execution, and backend buffer ownership. The
runtime architecture will be committed only after that integration strategy is
validated against the upstream source.
