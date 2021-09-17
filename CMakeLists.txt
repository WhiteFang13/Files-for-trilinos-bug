cmake_minimum_required(VERSION 3.14)
 project(Trillinos_test)
 set(CMAKE_CXX_STANDARD 17)
 include_directories(/usr/local/include/trilinos)
 find_package(Trilinos)
 include_directories(${Trilinos_INCLUDE_DIRECTORIES})
 add_executable(Trillinos_test main.cpp LinearEquationSolver.cpp LinearEquationSolver.h)
 target_link_libraries(Trillinos_test teuchosnumerics teuchoscomm
         epetra epetraext teuchoscore
         belostpetra ifpack2)
