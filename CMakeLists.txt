cmake_minimum_required(VERSION 3.0)


if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
		"Choose the type of build, options are: Debug Release."
	FORCE)
endif(NOT CMAKE_BUILD_TYPE)

#Enable C++17, may need C++20 in the future
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") #GCC flags
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-unused-but-set-parameter -I/usr/include/python2.7 -lpython2.7  -std=c++11 -L/usr/lib/python2.7")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") #clang flags
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter -Wno-unused-function")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
else()
	#Add support for MSVC and ICC if needed
endif()

option(ETALER_ENABLE_OPENCL "Enable the OpenCL backend" OFF)

include_directories(.)

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


