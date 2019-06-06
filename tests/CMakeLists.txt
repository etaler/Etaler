project(etaler_tests CXX)
include_directories("${PROJECT_BINARY_DIR}/../Etaler")

add_executable(etaler_tests common_tests.cpp main.cpp)
target_link_libraries(etaler_tests Etaler)

if(ETALER_ENABLE_OPENCL)
        project(etaler_tests_opencl CXX)
        add_executable(etaler_tests_opencl common_tests.cpp main_opencl.cpp)
        target_link_libraries(etaler_tests_opencl Etaler)
endif()