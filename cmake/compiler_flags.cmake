# Compiler flags and options for Astate project
# This file manages all compiler flags and build options

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ASTATE_ENABLE_WARNINGS "Enable compiler warning errors" ON)
option(ASTATE_ENABLE_TESTS "Enable unit tests" OFF)
option(ASTATE_ENABLE_PYTHON "Enable Python bindings" ON)
option(ASTATE_ENABLE_TOOLS "Enable build tools" ON)
option(ASTATE_BUILD_SHARED "Build shared libraries" ON)
option(ASTATE_ENABLE_DEBUG "Enable debug information" ON)
option(ASTATE_ENABLE_LTO "Enable Link Time Optimization with parallel compilation" OFF)

if(MSVC)
    add_compile_options(/W4 /permissive-)
    if(ASTATE_ENABLE_WARNINGS)
        add_compile_options(/WX)
    endif()
else()
    add_compile_options(-Wall -Wextra)
    if(ASTATE_ENABLE_WARNINGS)
        add_compile_options(-Wpedantic -Werror)
    endif()

    add_compile_options(
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-unused-function
        -Wno-unused-result
        -Wno-sign-compare
        -Wno-missing-field-initializers
        -Wno-deprecated-declarations
        -Wno-gnu-zero-variadic-macro-arguments
        -Wno-strict-prototypes
        -Wno-newline-eof
        -Wno-int-conversion
        -Wno-implicit-function-declaration
        -Wno-zero-length-array
        -Wno-language-extension-token
        -Wno-gnu-pointer-arith
    )
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
        -march=native
        -flto=thin
        -ffast-math
        -fstrict-aliasing
        -funroll-loops
    )

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 18)
        add_compile_options(
            -Wno-nan-infinity-disabled
            -Wno-zero-length-array
        )
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(
            -fomit-frame-pointer
            -fvectorize
        )
    endif()
endif()

function(suppress_warnings_for_third_party_libraries)
endfunction()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_options(-O3 -g)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -g")
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    add_compile_options(-Os)
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -Os")
elseif(CMAKE_BUILD_TYPE STREQUAL "ASAN")
    add_compile_options(-fsanitize=leak -g)
    set(CMAKE_CXX_FLAGS_ASAN "${CMAKE_CXX_FLAGS_ASAN} -fsanitize=leak -g")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall,-Wextra")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
endif()

include(ProcessorCount)
ProcessorCount(N)
if(NOT N EQUAL 0)
    set(CMAKE_BUILD_PARALLEL_LEVEL ${N})
endif()

if(ASTATE_ENABLE_LTO AND NOT MSVC)
    if(NOT N EQUAL 0)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=${N}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -flto=${N}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=${N}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -flto=${N}")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -flto=${N}")
    else()
        # Fallback to auto if processor count not available
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=auto")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -flto=auto")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=auto")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -flto=auto")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -flto=auto")
    endif()

    message(STATUS "LTO optimization enabled with ${N} parallel jobs")
elseif(NOT ASTATE_ENABLE_LTO)
    # Disable LTO to avoid warnings
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
    message(STATUS "LTO optimization disabled")
endif()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    message(STATUS "Using ccache for compilation acceleration")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "=== Compiler Configuration ===")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Parallel build: ${CMAKE_BUILD_PARALLEL_LEVEL}")
message(STATUS "==============================")