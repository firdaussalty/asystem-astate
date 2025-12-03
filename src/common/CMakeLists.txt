list(
    APPEND
    COMMON_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/option.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_topology_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/retry/counting_retry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/retry/sleeping_retry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/retry/exponential_backoff_retry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/retry/timeout_retry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/retry/time_bounded_retry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/retry/exponential_time_bounded_retry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/retry/retry_utils.cpp
)

add_library(astate_common STATIC ${COMMON_SRCS})

set_target_properties(astate_common PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(astate_common
    PRIVATE
        ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(astate_common
    PUBLIC
        ${ASTATE_COMMON_DEPS}
        ${ASTATE_CUDA_DEPS}
        ${ASTATE_TEST_DEPS}
)

target_compile_definitions(astate_common
    PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
        $<$<CONFIG:Release>:NDEBUG>
)

add_library(astate:: ALIAS astate_common)

install(TARGETS astate_common
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION include/astate_common
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.hh"
        PATTERN "*.cuh"
)

if(ASTATE_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

message(STATUS "Configured astate_common library (static)")
