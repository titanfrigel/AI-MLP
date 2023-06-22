##
## File: CMakeLists.txt
## Project: Ai2
## Created Date: Sunday, June 4th 2023, 3:11:45 pm
## Author: titanfrigel
## -----
## Last Modified: Tuesday, June 20th 2023, 3:59:24 pm
## Modified By: titanfrigel
##

cmake_minimum_required(VERSION 3.10)

project(ai)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD 23)
set(CMAKE_CXX_FLAGS "-g3")

add_executable(${PROJECT_NAME}
    src/Main.cpp
    # src/AI.cpp
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
    m
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
)
