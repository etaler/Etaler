project(Etaler CXX)


############################################################################
# Find Intel TBB include and libs

set(TBB_DIR "$ENV{TBB_ROOT_DIR}/cmake")

find_package(TBB REQUIRED)

if(NOT ${TBB_FOUND})
    message(FATAL_ERROR "Could not locate Intel TBB include & libs")
else()
    message(STATUS "Found existing Intel TBB targets ${TBB_IMPORTED_TARGETS}")
endif()


############################################################################
# Find OpenCL include and libs

if(ETALER_ENABLE_OPENCL)
    find_package(OpenCL REQUIRED)

    if("${OpenCL_INCLUDE_DIR}" STREQUAL "" OR "${OpenCL_LIBRARY}" STREQUAL "")
        message(FATAL_ERROR "Could not locate OpenCL include & libs")
    else()
        include_directories(${OpenCL_INCLUDE_DIRS})
        message(STATUS "Found existing OpenCL in ${OpenCL_INCLUDE_DIR}")
        message(STATUS "Found existing OpenCL lib ${OpenCL_LIBRARY}")
    endif()
endif()


############################################################################
# Setup the Etaler library building

set(CMAKE_CXX_VISIBILITY_PRESET hidden)

add_library(Etaler SHARED Backends/CPUBackend.cpp Core/DefaultBackend.cpp Core/Serialize.cpp Core/Tensor.cpp
	Algorithms/SpatialPooler.cpp Algorithms/TemporalMemory.cpp Core/TypeHelpers.cpp)

# Auto-generate the export header for shared library macros
include(GenerateExportHeader)
generate_export_header(Etaler)
include_directories(${PROJECT_BINARY_DIR})

target_include_directories(Etaler PRIVATE 3rdparty/pcg-cpp/include)
target_include_directories(Etaler PRIVATE 3rdparty/cereal/include)
target_include_directories(Etaler PRIVATE 3rdparty)

target_link_libraries(Etaler ${TBB_IMPORTED_TARGETS})

if(ETALER_ENABLE_OPENCL)
	target_sources(Etaler PRIVATE Backends/OpenCLBackend.cpp)

	if(NOT APPLE)
		target_link_libraries(Etaler ${OpenCL_LIBRARIES})
	else()
		target_link_libraries(Etaler "-framework OpenCL")
	endif()

	# Install OpenCL kernels
	install(DIRECTORY ${CMAKE_SOURCE_DIR}/kernels
		DESTINATION share/Etaler
		FILES_MATCHING PATTERN "*.cl")

	message("Etaler: OpenCL backend enabled.")
endif()

if(APPLE AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
	target_link_libraries(Etaler stdc++)
endif()

find_package(CxaDemangle)
if(HAVE_CXA_DEMANGLE)
	target_compile_definitions(Etaler PRIVATE HAVE_CXA_DEMANGLE)
endif()

install(FILES ${CMAKE_SOURCE_DIR}/cmake/FindEtaler.cmake
	DESTINATION share/Etaler COMPONENT cmake)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/Etaler
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp")

install(FILES ${PROJECT_BINARY_DIR}/Etaler_export.h DESTINATION include)

install(TARGETS Etaler
	LIBRARY DESTINATION lib)
