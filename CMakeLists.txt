cmake_minimum_required(VERSION 3.14)
project(Trillinos_test)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

include_directories(/usr/local/include/trilinos)
find_package(Trilinos)
find_package(Boost REQUIRED COMPONENTS mpi fiber serialization system thread log log_setup)
include_directories(${Trilinos_INCLUDE_DIRECTORIES})

add_executable(Trillinos_test main.cpp LinearEquationSolver.cpp LinearEquationSolver.h)

target_link_libraries(Trillinos_test
        teuchoscomm teuchoscore
        belostpetra ifpack2
        ${Boost_LIBRARIES})
