##
## File: CMakeLists.txt
## Project: AI-MLP
## Created Date: Sunday, June 4th 2023, 3:11:45 pm
## Author: titanfrigel
## -----
## Last Modified: Thursday, June 29th 2023, 4:46:20 pm
## Modified By: titanfrigel
##

cmake_minimum_required(VERSION 3.10)

project(ai)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD 23)
set(CMAKE_CXX_FLAGS "-g3")

add_executable(${PROJECT_NAME}
    src/Main.cpp
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
    m
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
)
