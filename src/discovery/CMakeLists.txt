list(
  APPEND
  DISCOVERY_SRCS
  ${CMAKE_CURRENT_LIST_DIR}/discovery_manager.cpp
  ${CMAKE_CURRENT_LIST_DIR}/file_config_center.cpp
  ${CMAKE_CURRENT_LIST_DIR}/http_config_center.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tcpstore_discovery.cpp
)

add_library(astate_discovery STATIC ${DISCOVERY_SRCS})

set_target_properties(astate_discovery PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(astate_discovery
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/astate_discovery>
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION include/astate_discovery
    FILES_MATCHING
        PATTERN "*.h"  PATTERN "*.hpp" PATTERN "*.hh"
)

target_link_libraries(astate_discovery PUBLIC ${ASTATE_COMMON_DEPS} astate_common)