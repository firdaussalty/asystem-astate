# client test
add_executable(client_test 
    atensor_serializer_test.cpp
    utils_test.cpp
    tensor_sharded_ops_test.cpp
)
target_include_directories(client_test
    PRIVATE
        ${CLIENT_INCLUDE_DIRS}
)
target_link_libraries(client_test
    PRIVATE
        astate_client
        astate_common
        ${ASTATE_COMMON_DEPS}
        ${ASTATE_TEST_DEPS}
	    ${ASTATE_PYTHON_DEPS}
        ${ASTATE_CUDA_DEPS}
        ${ASTATE_TRANSFER_DEPS}
)
add_test(NAME client_test COMMAND client_test)
set_tests_properties(client_test PROPERTIES
    LABELS astate_test
)

# transfer_test
add_executable(transfer_test
    http_transporter_test.cpp
    file_config_center_test.cpp
)
target_include_directories(transfer_test
    PRIVATE
        ${CLIENT_INCLUDE_DIRS}
)
target_link_libraries(transfer_test
    PRIVATE
        astate_client
        astate_common
        ${ASTATE_COMMON_DEPS}
        ${ASTATE_TEST_DEPS}
        ${ASTATE_TRANSFER_DEPS}
        ${ASTATE_PYTHON_DEPS}
)
add_test(NAME transfer_test COMMAND transfer_test)
set_tests_properties(transfer_test PROPERTIES
    LABELS astate_test
)

# tensor_transfer_pull_test_simple
add_executable(tensor_transfer_pull_test
    tensor_transfer_pull_test.cpp
)
target_include_directories(tensor_transfer_pull_test
    PRIVATE
        ${CLIENT_INCLUDE_DIRS}
)
target_link_libraries(tensor_transfer_pull_test
    PRIVATE
        astate_client
        astate_common
        leveldb
        z
        ${ASTATE_COMMON_DEPS}
        ${ASTATE_TEST_DEPS}
        ${ASTATE_TRANSFER_DEPS}
        ${ASTATE_PYTHON_DEPS}
        ${ASTATE_CUDA_DEPS}
)
add_test(NAME tensor_transfer_pull_test COMMAND tensor_transfer_pull_test)
set_tests_properties(tensor_transfer_pull_test PROPERTIES
    LABELS astate_test
)
