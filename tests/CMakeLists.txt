project(etaler_tests CXX)

#Find Catch2 via CMake
find_package(Catch2 REQUIRED)

add_executable(etaler_tests common_tests.cpp main.cpp)
target_link_libraries(etaler_tests Etaler Catch2::Catch2)

if(ETALER_ENABLE_OPENCL)
	project(etaler_tests_opencl CXX)
	add_executable(etaler_tests_opencl common_tests.cpp main_opencl.cpp)
	target_link_libraries(etaler_tests_opencl Etaler Catch2::Catch2)
endif()