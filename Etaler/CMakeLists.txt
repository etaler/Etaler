project(Etaler CXX)

add_library(Etaler SHARED Backends/CPUBackend.cpp Core/DefaultBackend.cpp Core/Serialize.cpp Core/Tensor.cpp
	Algorithms/SpatialPooler.cpp Algorithms/TemporalMemory.cpp Core/TypeHelpers.cpp)
target_link_libraries(Etaler tbb)
target_include_directories(Etaler PRIVATE 3rdparty/pcg-cpp/include)

if(ETALER_ENABLE_OPENCL)
	target_sources(Etaler PRIVATE Backends/OpenCLBackend.cpp)


	if(NOT APPLE)
		target_link_libraries(Etaler OpenCL)
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

install(TARGETS Etaler
	LIBRARY DESTINATION lib)
