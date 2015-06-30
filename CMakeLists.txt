cmake_minimum_required(VERSION 3.2)
project(algo_lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED )

set(HEADER_FILES algorithms.h)
set(SOURCE_FILES main.cpp algorithms.cpp)

add_executable(algo_lib ${SOURCE_FILES})

target_link_libraries( algo_lib ${OpenCV_LIBS} )