project(example1 CXX)
add_executable(example1 example1.cpp)
target_link_libraries(example1 Etaler)

project(spbench CXX)
add_executable(spbench spbench.cpp)
target_link_libraries(spbench Etaler)

option(ETALER_ENABLE_MATPLOTLIB_CPP "Enable Matplotlib CPP integration" ON)

project(tmexample CXX)
add_executable(tmexample tmexample.cpp)

if(ETALER_ENABLE_MATPLOTLIB_CPP)
	find_package(PythonLibs 2.7)
	target_include_directories(tmexample PRIVATE ${PYTHON_INCLUDE_DIRS})
	target_link_libraries(tmexample ${PYTHON_LIBRARIES})
endif()
target_link_libraries(tmexample Etaler)

project(tmbench CXX)
add_executable(tmbench tmbench.cpp)
target_link_libraries(tmbench Etaler)


if(ETALER_ENABLE_OPENCL)
	project(backend2backend CXX)
	add_executable(backend2backend backend2backend.cpp)
	target_link_libraries(backend2backend Etaler)
endif()