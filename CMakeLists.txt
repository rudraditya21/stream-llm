cmake_minimum_required(VERSION 3.20)

project(
    stream_llm
    VERSION 0.1.0
    DESCRIPTION "Run language models beyond your available memory"
    LANGUAGES C CXX
)

set(STREAM_LLM_GGML_DIR "" CACHE PATH "Path to the pinned llama.cpp/ggml source directory")
if(NOT STREAM_LLM_GGML_DIR)
    get_filename_component(STREAM_LLM_GGML_CANDIDATE "${CMAKE_CURRENT_SOURCE_DIR}/../upstream/llama.cpp/ggml" ABSOLUTE)
    if(EXISTS "${STREAM_LLM_GGML_CANDIDATE}/CMakeLists.txt")
        set(STREAM_LLM_GGML_DIR "${STREAM_LLM_GGML_CANDIDATE}")
    endif()
endif()
if(NOT EXISTS "${STREAM_LLM_GGML_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Set STREAM_LLM_GGML_DIR to the pinned llama.cpp/ggml directory")
endif()

set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory("${STREAM_LLM_GGML_DIR}" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)

add_library(stream_llm_runtime STATIC
    src/buffer_pool.cpp
    src/gguf_index.cpp
    src/qwen_layer.cpp
    src/tensor_reader.cpp
)
target_include_directories(stream_llm_runtime PUBLIC include)
target_compile_features(stream_llm_runtime PUBLIC cxx_std_20)
target_link_libraries(stream_llm_runtime PUBLIC ggml)

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
    add_test(NAME cli.runs COMMAND stream-llm)
    set_tests_properties(cli.runs PROPERTIES PASS_REGULAR_EXPRESSION "stream-llm")

    add_executable(runtime-tests tests/runtime_tests.cpp)
    target_link_libraries(runtime-tests PRIVATE stream_llm_runtime)
    stream_llm_enable_warnings(runtime-tests)
    add_test(NAME runtime.interfaces COMMAND runtime-tests)

    set(STREAM_LLM_TEST_MODEL "" CACHE FILEPATH "Optional GGUF used by the model index smoke test")
    if(STREAM_LLM_TEST_MODEL)
        add_test(NAME runtime.reference_model COMMAND runtime-tests "${STREAM_LLM_TEST_MODEL}")
    endif()
endif()

install(TARGETS stream-llm RUNTIME DESTINATION bin)
