cmake_minimum_required(VERSION 3.20)

project(
    stream_llm
    VERSION 0.1.0
    DESCRIPTION "Run language models beyond your available memory"
    LANGUAGES C CXX
)

set(STREAM_LLM_GGML_DIR "" CACHE PATH "Path to the pinned llama.cpp/ggml source directory")
set(STREAM_LLM_LLAMA_CPP_DIR "" CACHE PATH "Path to the pinned llama.cpp source directory")
if(NOT STREAM_LLM_LLAMA_CPP_DIR)
    if(STREAM_LLM_GGML_DIR)
        get_filename_component(STREAM_LLM_LLAMA_CPP_DIR "${STREAM_LLM_GGML_DIR}/.." ABSOLUTE)
    else()
        get_filename_component(STREAM_LLM_LLAMA_CPP_CANDIDATE "${CMAKE_CURRENT_SOURCE_DIR}/../upstream/llama.cpp" ABSOLUTE)
        if(EXISTS "${STREAM_LLM_LLAMA_CPP_CANDIDATE}/CMakeLists.txt")
            set(STREAM_LLM_LLAMA_CPP_DIR "${STREAM_LLM_LLAMA_CPP_CANDIDATE}")
        endif()
    endif()
endif()
if(NOT EXISTS "${STREAM_LLM_LLAMA_CPP_DIR}/include/llama.h")
    message(FATAL_ERROR "Set STREAM_LLM_LLAMA_CPP_DIR to the pinned llama.cpp directory")
endif()

set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
add_subdirectory("${STREAM_LLM_LLAMA_CPP_DIR}" "${CMAKE_BINARY_DIR}/llama" EXCLUDE_FROM_ALL)

add_library(stream_llm_runtime STATIC
    src/buffer_pool.cpp
    src/generator.cpp
    src/gguf_index.cpp
    src/qwen_executor.cpp
    src/qwen_layer.cpp
    src/tensor_reader.cpp
    src/tokenizer.cpp
)
target_include_directories(stream_llm_runtime PUBLIC include)
target_compile_features(stream_llm_runtime PUBLIC cxx_std_20)
target_link_libraries(stream_llm_runtime PUBLIC llama)

add_executable(stream-llm src/main.cpp)
target_link_libraries(stream-llm PRIVATE stream_llm_runtime)

function(stream_llm_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

stream_llm_enable_warnings(stream_llm_runtime)
stream_llm_enable_warnings(stream-llm)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME cli.help COMMAND stream-llm --help)
    set_tests_properties(cli.help PROPERTIES PASS_REGULAR_EXPRESSION "Usage: stream-llm")
    add_test(NAME cli.missing_arguments COMMAND stream-llm)
    set_tests_properties(cli.missing_arguments PROPERTIES WILL_FAIL TRUE)
    add_test(NAME cli.invalid_max_tokens
        COMMAND stream-llm --model unused.gguf --prompt test --max-tokens invalid)
    set_tests_properties(cli.invalid_max_tokens PROPERTIES WILL_FAIL TRUE)
    add_test(NAME cli.system_requires_chat
        COMMAND stream-llm --model unused.gguf --prompt test --max-tokens 1 --system test)
    set_tests_properties(cli.system_requires_chat PROPERTIES WILL_FAIL TRUE)

    add_executable(runtime-tests tests/runtime_tests.cpp)
    target_link_libraries(runtime-tests PRIVATE stream_llm_runtime)
    stream_llm_enable_warnings(runtime-tests)
    add_test(NAME runtime.interfaces COMMAND runtime-tests)

    add_executable(parity-trace tests/parity_trace.cpp)
    target_link_libraries(parity-trace PRIVATE stream_llm_runtime)
    stream_llm_enable_warnings(parity-trace)

    set(STREAM_LLM_TEST_MODEL "" CACHE FILEPATH "Optional GGUF used by the model index smoke test")
    if(STREAM_LLM_TEST_MODEL)
        add_test(NAME runtime.reference_model COMMAND runtime-tests "${STREAM_LLM_TEST_MODEL}")
        add_test(NAME cli.reference_model COMMAND stream-llm
            --model "${STREAM_LLM_TEST_MODEL}"
            --prompt "Hello world"
            --max-tokens 1
        )
        set_tests_properties(cli.reference_model PROPERTIES PASS_REGULAR_EXPRESSION "!")
        add_test(NAME cli.reference_chat COMMAND stream-llm
            --model "${STREAM_LLM_TEST_MODEL}"
            --chat
            --prompt "x"
            --max-tokens 1
        )
    endif()
endif()

install(TARGETS stream-llm RUNTIME DESTINATION bin)
