list(
  APPEND
  CORE_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/atensor_storage.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/atensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/in_memory_tensor_table.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/remote_tensor_table.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sharded_key.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tensor_sharded_ops.cpp
)

add_library(astate_core STATIC ${CORE_SRCS})

set_target_properties(astate_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION include/astate_core
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.hh"
        PATTERN "*.cuh"
)

target_link_libraries(astate_core
  PUBLIC
  ${ASTATE_COMMON_DEPS}
  astate_common
  astate_discovery
  astate_protocol
  astate_transfer
  astate_transport
)
