project(etaler_tests)
add_executable(etaler_tests test.cpp main.cpp)
target_link_libraries(etaler_tests Etaler)