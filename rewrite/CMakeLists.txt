cmake_minimum_required(VERSION 3.16)
project(leviathan_rewrite LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_executable(leviathan
    src/main.cpp
    src/chess.cpp
    src/search.cpp
)
target_compile_options(leviathan PRIVATE -O3 -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME rewrite_selftest COMMAND leviathan --selftest)
