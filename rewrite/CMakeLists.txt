cmake_minimum_required(VERSION 3.16)
project(leviathan_rewrite LANGUAGES C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LEVIATHAN_FATHOM_DIR "" CACHE PATH "Path to a materialized pinned Fathom source tree")

add_executable(leviathan
    src/main.cpp
    src/chess.cpp
    src/evaluator.cpp
    src/search.cpp
    src/tablebase.cpp
)
target_compile_options(leviathan PRIVATE -O3 -Wall -Wextra -Wpedantic)

if(LEVIATHAN_FATHOM_DIR)
    set(FATHOM_SRC "${LEVIATHAN_FATHOM_DIR}/src")
    foreach(required tbprobe.c tbprobe.h tbchess.c tbconfig.h stdendian.h)
        if(NOT EXISTS "${FATHOM_SRC}/${required}")
            message(FATAL_ERROR "Pinned Fathom source is incomplete: missing ${FATHOM_SRC}/${required}")
        endif()
    endforeach()
    target_sources(leviathan PRIVATE
        "${FATHOM_SRC}/tbprobe.c"
        "${FATHOM_SRC}/tbchess.c"
    )
    target_include_directories(leviathan PRIVATE "${FATHOM_SRC}")
    target_compile_definitions(leviathan PRIVATE LEVIATHAN_HAS_FATHOM=1 TB_NO_HELPER_API=1)
    find_package(Threads REQUIRED)
    target_link_libraries(leviathan PRIVATE Threads::Threads)
    message(STATUS "Leviathan: Fathom donor enabled from ${LEVIATHAN_FATHOM_DIR}")
endif()

enable_testing()
add_test(NAME rewrite_selftest COMMAND leviathan --selftest)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME donor_provenance_audit
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/audit_donors.py
                ${CMAKE_CURRENT_SOURCE_DIR}/donors/DONOR_REGISTRY.json
                ${CMAKE_CURRENT_SOURCE_DIR}/imports
    )
endif()
