if(ASTATE_ENABLE_PYTHON)
  pybind11_add_module(astate_cpp
    ./export/export.cpp
  )

  set_target_properties(astate_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/astate/_core
  )

  target_link_options(astate_cpp PRIVATE -Wl,--no-undefined)

  target_link_libraries(astate_cpp
    PRIVATE
      astate_core
      ${ASTATE_COMMON_DEPS}
      ${ASTATE_CUDA_DEPS}
      ${ASTATE_PYTHON_DEPS}
      ${ASTATE_TRANSFER_DEPS}
  )

  install(TARGETS astate_cpp
    LIBRARY DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/astate/_core
  )
endif()
