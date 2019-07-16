cmake_minimum_required(VERSION 3.10)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/Easy3D)

file(GLOB IMGUI_SOURCES "${CMAKE_CURRENT_LIST_DIR}/imgui/*.cpp")
file(GLOB IMGUI_IMPL_SOURCES "${CMAKE_CURRENT_LIST_DIR}/imgui/impl/*.cpp")
link_directories(${CMAKE_SOURCE_DIR}/Easy3D/build/lib/)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -g -ggdb")
add_library (LayersVisualizer STATIC  Visualizer.cpp viewer_imgui.cpp ${IMGUI_SOURCES} ${IMGUI_IMPL_SOURCES})
target_include_directories(LayersVisualizer PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(LayersVisualizer PRIVATE ${CMAKE_SOURCE_DIR}/Easy3D/)
target_include_directories(LayersVisualizer PRIVATE ${CMAKE_SOURCE_DIR}/imgui/)
find_package(OpenGL REQUIRED)
target_link_libraries(LayersVisualizer LINK_PUBLIC ${OPENGL_LIBRARIES} glew core viewer glfw rply tinyfiledialogs util LAStools fileio dl glew glfw  ${OPENGL_LIBRARIES} X11 ) 



