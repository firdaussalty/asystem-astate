message(STATUS "Configuring Astate submodules...")

add_subdirectory(common)
add_subdirectory(protocol)
add_subdirectory(discovery)
add_subdirectory(transfer)
add_subdirectory(transport)
add_subdirectory(core)

set(SAVED_CMAKE_C_CLANG_TIDY ${CMAKE_C_CLANG_TIDY})

unset(CMAKE_C_CLANG_TIDY)
add_subdirectory(utrans)

set(CMAKE_C_CLANG_TIDY ${SAVED_CMAKE_C_CLANG_TIDY})