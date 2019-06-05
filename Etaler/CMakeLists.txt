project(Etaler CXX)

############################################################################
# Find Intel TBB include and libs

find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})

############################################################################
# Find OpenCL include and libs

#find_package(OpenCL REQUIRED)
#include_directories(${OpenCL_INCLUDE_DIRS})

find_path(OPENCL_INCLUDE_DIRS 
    NAMES OpenCL/cl.h CL/cl.h
    HINTS
        $ENV{AMDAPPSDKROOT}/include/
        $ENV{INTELOCLSDKROOT}/include/
        $ENV{CUDA_PATH}/include/
        $ENV{OPENCL_ROOT}/include/
    PATHS
        /usr/include /usr/local/include/
)

if(BITNESS EQUAL 64)
    find_library(OPENCL_LIBRARIES
        NAMES OpenCL
        HINTS
            $ENV{AMDAPPSDKROOT}/lib
            $ENV{INTELOCLSDKROOT}/lib
            $ENV{CUDA_PATH}/lib
            $ENV{OPENCL_ROOT}/lib
        PATH_SUFFIXES x86_64 x64
        PATHS /usr/lib64 /usr/lib /usr/local/lib
    )
elseif(BITNESS EQUAL 32)
    find_library(OPENCL_LIBRARIES
        NAMES OpenCL
        HINTS
            $ENV{AMDAPPSDKROOT}/lib
            $ENV{INTELOCLSDKROOT}/lib
            $ENV{CUDA_PATH}/lib
            $ENV{OPENCL_ROOT}/lib
        PATH_SUFFIXES x86 Win32
        PATHS /usr/lib32 /usr/lib /usr/local/lib
    )
endif()

# Fall-back to find a POCL install
if(OPENCL_LIBRARIES STREQUAL "")
    find_library(OPENCL_LIBRARIES
        NAMES pocl
        HINTS $HOME/.local/pocl/lib /usr/lib64 /usr/lib /usr/local/lib
    )
endif()

if(OPENCL_INCLUDE_DIRS STREQUAL "" OR OPENCL_LIBRARIES STREQUAL "")
    message(FATAL_ERROR "Could not locate OpenCL include & libs")
else()
    message(STATUS "Found existing OpenCL in ${OPENCL_INCLUDE_DIRS}")
    message(STATUS "Found existing OpenCL lib ${OPENCL_LIBRARIES}")
endif()

include_directories(${OPENCL_INCLUDE_DIRS})


add_library(Etaler STATIC Backends/CPUBackend.cpp Core/DefaultBackend.cpp Core/Serialize.cpp Core/Tensor.cpp
	Algorithms/SpatialPooler.cpp Algorithms/TemporalMemory.cpp Core/TypeHelpers.cpp)
target_link_libraries(Etaler ${TBB_LIBRARIES})
target_include_directories(Etaler PRIVATE 3rdparty/pcg-cpp/include)
target_include_directories(Etaler PRIVATE 3rdparty/tbb/include)
target_include_directories(Etaler PRIVATE 3rdparty/cereal/include)
target_include_directories(Etaler PRIVATE 3rdparty)


if(ETALER_ENABLE_OPENCL)
	target_sources(Etaler PRIVATE Backends/OpenCLBackend.cpp)


	if(NOT APPLE)
		target_link_libraries(Etaler ${OPENCL_LIBRARIES})
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
