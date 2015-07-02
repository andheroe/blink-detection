cmake_minimum_required(VERSION 3.2)
project(blink-detection)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED )

set(HEADER_FILES algorithms.h)
set(SOURCE_FILES main.cpp algorithms.cpp)

add_executable(blink-detection ${SOURCE_FILES})

target_link_libraries( blink-detection ${OpenCV_LIBS} )