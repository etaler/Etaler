project(Etaler CXX)

list(APPEND ETLAR_SOURCE Backends/CPUBackend.cpp Core/DefaultBackend.cpp Core/Serialize.cpp Core/Tensor.cpp Algorithms/SpatialPooler.cpp Algorithms/TemporalMemory.cpp)
list(APPEND ETALER_LINK_LIBRARIES tbb)

if(ETALER_ENABLE_OPENCL)
	list(APPEND ETLAR_SOURCE Backends/OpenCLBackend.cpp)


	if(NOT APPLE)
		list(APPEND ETALER_LINK_LIBRARIES OpenCL)
	else()
		list(APPEND ETALER_LINK_LIBRARIES "-framework OpenCL")
	endif()


	install(DIRECTORY ${CMAKE_SOURCE_DIR}/kernels
		DESTINATION share/Etaler
		FILES_MATCHING PATTERN "*.cl")

	message("Etaler: OpenCL backend enabled.")
endif()

if(APPLE)
	list(APPEND ETALER_LINK_LIBRARIES stdc++)
endif()


include_directories(3rdparty/pcg-cpp/include)
add_library(Etaler SHARED ${ETLAR_SOURCE})
target_link_libraries(Etaler ${ETALER_LINK_LIBRARIES})

install(FILES ${CMAKE_SOURCE_DIR}/cmake/FindEtaler.cmake
	DESTINATION share/Etaler COMPONENT cmake)

INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/Etaler
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp")

INSTALL(TARGETS Etaler
	LIBRARY DESTINATION lib)
