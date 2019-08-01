project(Etaler CXX)


############################################################################
# Setup the Etaler library building

set(CMAKE_CXX_VISIBILITY_PRESET hidden)

add_library(Etaler SHARED Backends/CPUBackend.cpp Core/DefaultBackend.cpp Core/Serialize.cpp Core/Tensor.cpp
	Algorithms/SpatialPooler.cpp Algorithms/TemporalMemory.cpp Core/TypeHelpers.cpp)

# Auto-generate the export header for shared library macros
include(GenerateExportHeader)
generate_export_header(Etaler EXPORT_FILE_NAME Etaler_export.h)

target_include_directories(Etaler PRIVATE 3rdparty/pcg-cpp/include)
target_include_directories(Etaler PRIVATE 3rdparty/cereal/include)
target_include_directories(Etaler PRIVATE 3rdparty/half_precision)
target_include_directories(Etaler PRIVATE 3rdparty)

if(${TBB_FOUND})
	target_link_libraries(Etaler ${TBB_IMPORTED_TARGETS})
else()
	target_link_libraries(Etaler tbb)
endif()

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

	message(STATUS "Etaler: OpenCL backend enabled.")
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
