cmake_minimum_required(VERSION 3.18)
project(SVD_OMP_only)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3")

add_executable(SVD_OMP_only main.cpp lib/Matrix.h lib/global.h lib/JacobiMethod.cpp lib/JacobiMethod.h lib/Utils.cpp lib/Utils.h)

target_link_libraries(SVD_OMP_only OpenMP::OpenMP_CXX)
