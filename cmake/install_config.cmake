# Installation configuration for Astate project
# This file manages installation settings and packaging

# Include required modules
include(CMakePackageConfigHelpers)

# Set installation directories
set(ASTATE_INSTALL_INCLUDEDIR "include")
set(ASTATE_INSTALL_LIBDIR "lib")
set(ASTATE_INSTALL_BINDIR "bin")
set(ASTATE_INSTALL_DATADIR "share/astate")

# Configure package configuration file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/astate_config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/astate_config.cmake"
    INSTALL_DESTINATION ${ASTATE_INSTALL_LIBDIR}/cmake/astate
)

# Configure package version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/astate_config_version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install package configuration files
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/astate_config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/astate_config_version.cmake"
    DESTINATION ${ASTATE_INSTALL_LIBDIR}/cmake/
)

set(INSTALL_TARGETS_LIST "")

if(TARGET astate_core)
    list(APPEND INSTALL_TARGETS_LIST astate_core)
endif()

if(TARGET astate_common)
    list(APPEND INSTALL_TARGETS_LIST astate_common)
endif()

if(TARGET astate_discovery)
    list(APPEND INSTALL_TARGETS_LIST astate_discovery)
endif()

if(TARGET astate_protocol)
    list(APPEND INSTALL_TARGETS_LIST astate_protocol)
endif()

if(TARGET astate_transfer)
    list(APPEND INSTALL_TARGETS_LIST astate_transfer)
endif()

if(TARGET astate_transport)
    list(APPEND INSTALL_TARGETS_LIST astate_transport)
endif()

if(TARGET astate_utrans)
    list(APPEND INSTALL_TARGETS_LIST astate_utrans)
endif()

if(INSTALL_TARGETS_LIST)
    install(TARGETS ${INSTALL_TARGETS_LIST}
        EXPORT astate_targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

install(
    EXPORT astate_targets
    FILE astate_targets.cmake
    NAMESPACE astate::
    DESTINATION ${ASTATE_INSTALL_LIBDIR}/cmake/astate
)

# Create and install find module
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/find_astate.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/find_astate.cmake"
    @ONLY
)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/find_astate.cmake"
    DESTINATION ${ASTATE_INSTALL_DATADIR}/cmake
)

# Install documentation
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    install(
        FILES "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
        DESTINATION ${ASTATE_INSTALL_DATADIR}
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
    install(
        FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
        DESTINATION ${ASTATE_INSTALL_DATADIR}
    )
endif()

# Install Python package if enabled
if(ASTATE_ENABLE_PYTHON)
    # Install Python package files
    install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/astate/python/astate"
        DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
        FILES_MATCHING PATTERN "*.py"
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
    )

    # Install Python package metadata
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/astate/python/pyproject.toml")
        install(
            FILES "${CMAKE_CURRENT_SOURCE_DIR}/astate/python/pyproject.toml"
            DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/astate
        )
    endif()
endif()

# Print installation configuration
message(STATUS "=== Installation Configuration ===")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Include directory: ${ASTATE_INSTALL_INCLUDEDIR}")
message(STATUS "Library directory: ${ASTATE_INSTALL_LIBDIR}")
message(STATUS "Binary directory: ${ASTATE_INSTALL_BINDIR}")
message(STATUS "Data directory: ${ASTATE_INSTALL_DATADIR}")
message(STATUS "Install targets: ${INSTALL_TARGETS_LIST}")
message(STATUS "===================================")
