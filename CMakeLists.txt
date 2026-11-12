cmake_minimum_required(VERSION 3.20)

project(
    stream_llm
    VERSION 0.1.0
    DESCRIPTION "Run language models beyond your available memory"
    LANGUAGES CXX
)

add_executable(stream-llm src/main.cpp)

target_compile_features(stream-llm PRIVATE cxx_std_20)

if(MSVC)
    target_compile_options(stream-llm PRIVATE /W4 /permissive-)
else()
    target_compile_options(stream-llm PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    add_test(NAME cli.runs COMMAND stream-llm)
    set_tests_properties(cli.runs PROPERTIES PASS_REGULAR_EXPRESSION "stream-llm")
endif()

install(TARGETS stream-llm RUNTIME DESTINATION bin)
