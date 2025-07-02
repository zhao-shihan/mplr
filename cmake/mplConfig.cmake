include(CMakeFindDependencyMacro)

find_dependency(MPI REQUIRED CXX)

include(${CMAKE_CURRENT_LIST_DIR}/mplTargets.cmake)
