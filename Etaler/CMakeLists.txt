project(Etaler CXX)

list(APPEND ETLAR_SOURCE Backends/CPUBackend.cpp Core/DefaultBackend.cpp Core/Serialize.cpp Core/Tensor.cpp Algorithms/SpatialPooler.cpp)
list(APPEND ETALER_LINK_LIBRARIES tbb)

if(ETALER_ENABLE_OPENCL)
	list(APPEND ETLAR_SOURCE Backends/OpenCLBackend.cpp)
	list(APPEND ETALER_LINK_LIBRARIES OpenCL)
	message("Etaler: OpenCL backend enabled.")
endif()


include_directories(3rdparty/pcg-cpp/include)
add_library(Etaler SHARED ${ETLAR_SOURCE})
target_link_libraries(Etaler ${ETALER_LINK_LIBRARIES})

INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/Etaler
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp*|*.h*")

INSTALL(TARGETS Etaler
	LIBRARY DESTINATION lib)