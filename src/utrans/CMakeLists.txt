file(GLOB_RECURSE UTRANS_SRCS
        ${PROJECT_SOURCE_DIR}/src/*.c
)

add_library(astate_utrans STATIC ${UTRANS_SRCS})

set_target_properties(astate_utrans PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(astate_utrans PUBLIC ${ASTATE_TRANSFER_DEPS})

target_include_directories(astate_utrans
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/astate_utrans>
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION include/astate_utrans
    FILES_MATCHING
        PATTERN "*.h"  PATTERN "*.hpp" PATTERN "*.hh"
)