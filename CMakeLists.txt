cmake_minimum_required(VERSION 3.0)
project(Etaler)


SET(BUILD_SHARED_LIBS ON)

if(BUILD_SHARED_LIBS AND MSVC)
	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING
		"Choose the type of build, options are: Debug Release."
	FORCE)
endif(NOT CMAKE_BUILD_TYPE)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

#Enable C++17, may need C++20 in the future
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ETALER_NATIVE_BUILD "Enable compiler optimizing for host processor archicture" OFF)
option(ETALER_ENABLE_SIMD "Enable SIMD optimizations" ON)

# If not building for native archicture: Try to enable SIMD
if((NOT ETALER_NATIVE_BUILD) AND ETALER_ENABLE_SIMD)
	# Detect the CPU archicturer and enable SIMD when possible
	include(EnableSIMD)
endif()

if (${CMAKE_CXX_COMPILER_ID} MATCHES "(GNU)|(Clang)|(AppleClang)") #GCC/clang flags
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter -Wno-unused-function") # common flags
	if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") # GCC flags
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-parameter")
	else() # clang flags
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-absolute-value -Wno-unused-lambda-capture")
	endif()

    if(ETALER_NATIVE_BUILD)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
	endif()
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
else()
	#Add support for MSVC and ICC if needed
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DNOMINMAX")
	# No -march=native equlivent for MSVC. We just force AVX
	if(ETALER_NATIVE_BUILD)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
	endif()
endif()

#Use GOLD if avaliabel (on Linux)
if(UNIX)
	include("cmake/UseGOLD.cmake")
endif()

option(ETALER_ENABLE_OPENCL "Enable the OpenCL backend" OFF)

include_directories(.)
include_directories(${PROJECT_BINARY_DIR}/Etaler)

############################################################################
# Find Intel TBB include and libs

# Is TBB already installed?
include(CheckIncludeFileCXX)
set(CMAKE_REQUIRED_INCLUDES ${CMAKE_INCLUDE_PATH})
Check_Include_File_CXX("tbb/tbb.h" TBB_INCLUDE_FOUND)
if(NOT TBB_INCLUDE_FOUND)
	set(FIND_MODULE_TBB TRUE)
else()
	set(FIND_MODULE_TBB FALSE)
endif()

# No, so lets try and find a binary release install
if(${FIND_MODULE_TBB})
	# tbbvars.sh (part of binary installation of TBB) uses TBBROOT env var
	if(DEFINED ENV{TBBROOT})
		set(TBB_DIR "$ENV{TBBROOT}/cmake")
	else()
		set(TBB_DIR "$ENV{TBB_ROOT_DIR}/cmake")
	endif()

	find_package(TBB)

	if(NOT ${TBB_FOUND})
		message(FATAL_ERROR "Could not locate Intel TBB include & libs")
	else()
		message(STATUS "Found existing Intel TBB")
		set(TBB_IMPORTED_TARGETS ${TBB_LIBRARIES})
	endif()
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


#Buile Etlar itself
add_subdirectory(Etaler)

option(ETALER_BUILD_EXAMPLES "Build examples" ON)
if (ETALER_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

option(ETALER_BUILD_TESTS "Build tests" ON)
if (ETALER_BUILD_TESTS)
	add_subdirectory(tests)
endif()

option(ETALER_BUILD_DOCS "Build documents" OFF)
if (ETALER_BUILD_DOCS)
	add_subdirectory(docs)
endif()
