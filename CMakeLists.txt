# 3.18 is required for find_package cudatoolkit
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(PROG_NAME agphys)
project(${PROG_NAME} VERSION 1.0.0 LANGUAGES CXX CUDA)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


set(SAIGA_BUILD_SAMPLES OFF)
set(SAIGA_BUILD_TESTS ON)
set(SAIGA_MODULE_VULKAN OFF)
set(SAIGA_NO_INSTALL ON)
set(SAIGA_USE_SUBMODULES ON)
add_subdirectory(External/saiga)
#To make sure that the host code also sees the cuda headers.
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
include(helper_macros)
DefaultBuildType(RelWithDebInfo)

############# Required LIBRARIES ###############


message("\nSearching for installed libaries...")

#saiga
#find_package(SAIGA REQUIRED QUIET)
PackageHelperTarget(saiga_core SAIGA_FOUND)
PackageHelperTarget(saiga_cuda SAIGA_FOUND)
PackageHelperTarget(saiga_opengl SAIGA_FOUND)

#cuda
find_package(CUDAToolkit 10.2 REQUIRED)
PackageHelperTarget(CUDA::cudart CUDAToolkit_FOUND)

#eigen
#find_package(Eigen3 3.3.90 REQUIRED)
PackageHelperTarget(Eigen3::Eigen EIGEN_FOUND)

#c++17 filesystem
#find_package(Filesystem REQUIRED)
#PackageHelperTarget(std::filesystem FILESYSTEM_FOUND)

#openmp
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
PackageHelper(OpenMP "${OPENMP_FOUND}" "" "")


message("")
message("Linked Libraries:\n${LIB_TARGETS}")
message("")

############# COMPILER FLAGS ###############

OPTION(CUDA_PROFILING "Enables the PROFILING #define" OFF)
OPTION(CUDA_DEBUG "Enables CUDA Debugging in Nsight" OFF)
OPTION(FULL_SPEED "Enables All Optimizations" OFF)


if (CUDA_PROFILING)
    add_definitions(-DCUDA_PROFILING)
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
    SET(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

if (CUDA_DEBUG)
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
    SET(CMAKE_BUILD_TYPE DEBUG)
endif()

if (FULL_SPEED)
    SET(CMAKE_BUILD_TYPE RELEASE)
    add_definitions(-DCUDA_NDEBUG)
else()
    add_definitions(-DCUDA_DEBUG)
endif()

if (NOT MSVC)
    list(APPEND AGPHYS_CUDA_FLAGS "-Xcompiler=-fopenmp")
endif ()

list(APPEND AGPHYS_CUDA_FLAGS "-use_fast_math")
list(APPEND AGPHYS_CUDA_FLAGS "--expt-relaxed-constexpr")
list(APPEND AGPHYS_CUDA_FLAGS "-Xcudafe=--diag_suppress=esa_on_defaulted_function_ignored")
list(APPEND AGPHYS_CUDA_FLAGS "-Xcompiler=-march=native")
list(APPEND AGPHYS_CUDA_FLAGS "--allow-unsupported-compiler")

set(AGPHYS_CUDA_ARCH 52-virtual 75)

message(STATUS "CXX  FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CUDA FLAGS: ${AGPHYS_CUDA_FLAGS}")

if (MSVC)
    #multiprocessor compilation for visual studio
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else()
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-aliasing")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
endif()

############# SOURCE ###############

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)


#############  output dir ############# 

if(MSVC)
    #visual studio working dir
    #set_target_properties(${PROG_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    set_target_properties(saiga_core saiga_cuda saiga_opengl saiga_vision glfw glog assimp
       PROPERTIES
       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()

#copies executable to top level source tree
set_target_properties( ${PROG_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

