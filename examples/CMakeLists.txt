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


project(tm_visualize CXX)
link_directories(${CMAKE_SOURCE_DIR}/Etaler/3rdparty/LayersVisualizer/build/)
link_directories(${CMAKE_SOURCE_DIR}/Etaler/3rdparty/LayersVisualizer/Easy3D/build/lib)
include_directories( ${CMAKE_SOURCE_DIR}/Etaler/3rdparty/LayersVisualizer/)
include_directories( ${CMAKE_SOURCE_DIR}/Etaler/3rdparty/LayersVisualizer/Easy3D)
include_directories( ${CMAKE_SOURCE_DIR}/Etaler/3rdparty/LayersVisualizer/imgui)
add_subdirectory(${CMAKE_SOURCE_DIR}/Etaler/3rdparty/LayersVisualizer ${CMAKE_SOURCE_DIR}/Etaler/3rdparty/LayersVisualizer/build/)
find_package(OpenGL REQUIRED)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -g")
add_executable(tm_visualize tm_visualize.cpp) 
target_link_libraries(tm_visualize ${OPENGL_LIBRARIES} Etaler LayersVisualizer glew core viewer glfw rply tinyfiledialogs util LAStools fileio dl glew glfw  ${OPENGL_LIBRARIES} X11 ) 


project(tmbench CXX)
add_executable(tmbench tmbench.cpp)
target_link_libraries(tmbench Etaler)


if(ETALER_ENABLE_OPENCL)
	project(backend2backend CXX)
	add_executable(backend2backend backend2backend.cpp)
	target_link_libraries(backend2backend Etaler)
endif()
