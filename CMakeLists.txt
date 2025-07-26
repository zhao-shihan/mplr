# 3.21 required for PROJECT_IS_TOP_LEVEL, 3.10 required for mpi targets, 3.5 required for boost targets
cmake_minimum_required(VERSION 3.21)

project(mplr VERSION 0.25.726 LANGUAGES CXX C)

if(NOT DEFINED CACHE{BUILD_TESTING})
  set(BUILD_TESTING OFF CACHE BOOL "")
endif()
include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# project requires c++17 to build
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI 3.1 REQUIRED CXX)

add_library(mplr INTERFACE)
target_include_directories(mplr
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# convention for allowing use as a subdirectory
add_library(mplr::mplr ALIAS mplr)
target_link_libraries(mplr INTERFACE MPI::MPI_CXX)

option(MPLR_BUILD_EXAMPLES "build the mplr examples" ${PROJECT_IS_TOP_LEVEL})
if(MPLR_BUILD_EXAMPLES)
  find_package(MPI 3.1 REQUIRED C)
  add_subdirectory(examples)
endif()
if(BUILD_TESTING)
  add_subdirectory(test)
endif()

option(MPLR_INSTALL "Generate and install MPLR target" ${PROJECT_IS_TOP_LEVEL})
if(MPLR_INSTALL)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mplConfig.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mplr)

  install(DIRECTORY mplr DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  install(TARGETS mplr EXPORT mplTargets)
  export(EXPORT mplTargets
      NAMESPACE mplr::
      FILE mplTargets.cmake)
  install(EXPORT mplTargets
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mplr
      NAMESPACE mplr::
      FILE mplTargets.cmake)
endif()

# option(MPLR_BUILD_DOCUMENTATION "build the mplr documentation using Doxygen and Sphinx" OFF)
# add_subdirectory(doc)
