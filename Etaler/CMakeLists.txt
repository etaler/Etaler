project(Etaler CXX)

add_library(Etaler SHARED Backends/CPUBackend.cpp Core/DefaultBackend.cpp Core/Serialize.cpp Core/Tensor.cpp Algorithms/SpatialPooler.cpp Algorithms/TemporalMemory.cpp)
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

install(FILES ${CMAKE_SOURCE_DIR}/cmake/FindEtaler.cmake
	DESTINATION share/Etaler COMPONENT cmake)

INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/Etaler
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp")

INSTALL(TARGETS Etaler
	LIBRARY DESTINATION lib)
