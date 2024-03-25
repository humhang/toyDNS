cmake_minimum_required(VERSION 3.20)
if(POLICY CMP0069)
    cmake_policy(SET CMP0069 NEW)
endif()
if (POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif ()
project(toyDNS CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR} CACHE PATH "Installing prefix of the project" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()


# trying to find Kokkos as standalone installation
find_package(Kokkos 4.0.0 QUIET)
if(NOT Kokkos_FOUND)
    message(STATUS "Kokkos not found. Trying to find as Trilinos component")
    # try to find Kokkos as Trilinos component
    find_package(Trilinos 15.0.0 COMPONENTS Kokkos QUIET)
    if(NOT Trilinos_FOUND)
        message(FATAL_ERROR "Cannot find Kokkos as standalone installation or Trilinos components!")
    endif ()
endif ()

if((NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" ) ) AND (NOT Kokkos_ENABLE_CUDA))
    include(CheckIPOSupported)
    check_ipo_supported(RESULT LTOSupported)
    if(LTOSupported)
        message(STATUS "Linking time optimization supported")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif(LTOSupported)
endif()


find_package(HDF5 REQUIRED COMPONENTS CXX HL)


add_subdirectory(src)
