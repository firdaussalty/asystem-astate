add_executable(common_test
    option_test.cpp
    string_utils_test.cpp
    retry_test.cpp
    counting_and_sleep_retry_test.cpp
    thread_pool_test.cpp
    numa_aware_allocator_test.cpp
)

target_include_directories(common_test
    PRIVATE
)

find_package(Threads REQUIRED)

target_link_libraries(common_test PRIVATE
    astate_common
    numa
    z
    ${ASTATE_COMMON_DEPS}
    ${ASTATE_TEST_DEPS}
    ${ASTATE_PYTHON_DEPS}
)

add_test(NAME common_test COMMAND common_test)
set_tests_properties(common_test PROPERTIES
    LABELS astate_test
)
