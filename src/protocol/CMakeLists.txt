if(NOT DEFINED PROTOC_EXECUTABLE OR PROTOC_EXECUTABLE STREQUAL "")
  find_program(PROTOC_EXECUTABLE protoc)
endif()
if(NOT PROTOC_EXECUTABLE)
  message(FATAL_ERROR "protoc not found; please ensure Protobuf is installed and PROTOC_EXECUTABLE is set.")
endif()

set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(PROTO_GEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/gen)

# 创建生成目录
file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

file(GLOB PROTO_FILES ${PROTO_DIR}/*.proto)

message("PROTO_FILES: ${PROTO_FILES}")

foreach(PROTO_FILE ${PROTO_FILES})
  get_filename_component(PROTO_FILE_WE ${PROTO_FILE} NAME_WE)
  # Build the corresponding .h/.cc file paths of current proto file.
  set(PROTO_FILE_HDR "${PROTO_GEN_DIR}/${PROTO_FILE_WE}.pb.h")
  set(PROTO_FILE_SRC "${PROTO_GEN_DIR}/${PROTO_FILE_WE}.pb.cc")
  # Record the .h/.cc files into 'PROTO_HDRS' and 'PROTO_SRC'.
  list(APPEND PROTO_HDRS ${PROTO_FILE_HDR})
  list(APPEND PROTO_SRCS ${PROTO_FILE_SRC})

  add_custom_command(
    OUTPUT ${PROTO_FILE_HDR} ${PROTO_FILE_SRC}
    COMMAND
      ${PROTOC_EXECUTABLE}
      ARGS --cpp_out "${PROTO_GEN_DIR}" -I ${PROTO_DIR}
      ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating protobuf files (proto2) into ${PROTO_GEN_DIR}"
  )
endforeach()

list(
  APPEND
  PROTOCOL_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/messages.cpp
  ${PROTO_SRCS}
)

include_directories(${PROTO_GEN_DIR})

add_library(astate_protocol STATIC ${PROTOCOL_SRCS})

set_target_properties(astate_protocol PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(astate_protocol
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/astate_protocol>
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION include/astate_protocol
    FILES_MATCHING
        PATTERN "*.h"  PATTERN "*.hpp" PATTERN "*.hh"
)

target_link_libraries(astate_protocol PUBLIC ${ASTATE_COMMON_DEPS} astate_common)