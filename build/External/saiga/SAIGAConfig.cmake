# Locates the SAIGA library
# This module defines:
# SAIGA_FOUND
# SAIGA_INCLUDE_DIRS
# SAIGA_LIBRARIES
# SAIGA_LIBRARY2


find_path(SAIGA_INCLUDE_DIRS 
	NAMES 
                saiga/core/Core.h
	PATHS
                C:/Users/linda/Documents/AGPhys Projekt/ships/External/saiga/src
)


find_path(SAIGA_BUILD_INCLUDE_DIRS
        NAMES
                saiga/saiga_buildconfig.h
        PATHS
                C:/Users/linda/Documents/AGPhys Projekt/ships/build/External/saiga/include
)

set(SAIGA_GLBINDING_INCLUDE_DIRS ${SAIGA_INCLUDE_DIRS}/saiga/opengl/glbinding/include)
#set(SAIGA_CONFIG_INCLUDE_DIRS C:/Users/linda/Documents/AGPhys Projekt/ships/build/External/saiga/include)

find_path(SAIGA_SHADER_DIRS
        NAMES
                saiga/shaderConfig.h
        PATHS
                C:/Users/linda/Documents/AGPhys Projekt/ships/External/saiga/shader/include
)


# check each module individually
find_library(SAIGA_CORE_LIBRARY   NAMES saiga_core   PATHS C:/Users/linda/Documents/AGPhys Projekt/ships/build/External/saiga/bin)
find_library(SAIGA_OPENGL_LIBRARY NAMES saiga_opengl PATHS C:/Users/linda/Documents/AGPhys Projekt/ships/build/External/saiga/bin)
find_library(SAIGA_VULKAN_LIBRARY NAMES saiga_vulkan PATHS C:/Users/linda/Documents/AGPhys Projekt/ships/build/External/saiga/bin)
find_library(SAIGA_VISION_LIBRARY NAMES saiga_vision PATHS C:/Users/linda/Documents/AGPhys Projekt/ships/build/External/saiga/bin)
find_library(SAIGA_EXTRA_LIBRARY  NAMES saiga_extra  PATHS C:/Users/linda/Documents/AGPhys Projekt/ships/build/External/saiga/bin)
find_library(SAIGA_CUDA_LIBRARY   NAMES saiga_cuda   PATHS C:/Users/linda/Documents/AGPhys Projekt/ships/build/External/saiga/bin)


# - Append everything to SAIGA_LIBRARIES
if(SAIGA_CORE_LIBRARY)
    set(SAIGA_LIBRARIES ${SAIGA_LIBRARIES} ${SAIGA_CORE_LIBRARY})
endif()

if(SAIGA_OPENGL_LIBRARY)
    set(SAIGA_LIBRARIES ${SAIGA_LIBRARIES} ${SAIGA_OPENGL_LIBRARY})
endif()

if(SAIGA_VULKAN_LIBRARY)
    set(SAIGA_LIBRARIES ${SAIGA_LIBRARIES} ${SAIGA_VULKAN_LIBRARY})
endif()

if(SAIGA_VISION_LIBRARY)
    set(SAIGA_LIBRARIES ${SAIGA_LIBRARIES} ${SAIGA_VISION_LIBRARY})
endif()

if(SAIGA_EXTRA_LIBRARY)
    set(SAIGA_LIBRARIES ${SAIGA_LIBRARIES} ${SAIGA_EXTRA_LIBRARY})
endif()

if(SAIGA_CUDA_LIBRARY)
    set(SAIGA_LIBRARIES ${SAIGA_LIBRARIES} ${SAIGA_CUDA_LIBRARY})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
        SAIGA
	DEFAULT_MSG
        SAIGA_INCLUDE_DIRS
        SAIGA_BUILD_INCLUDE_DIRS
        SAIGA_SHADER_DIRS
        SAIGA_CORE_LIBRARY
)



set(SAIGA_INCLUDE_DIRS ${SAIGA_INCLUDE_DIRS} ${SAIGA_SHADER_DIRS}  ${SAIGA_GLBINDING_INCLUDE_DIRS} ${SAIGA_BUILD_INCLUDE_DIRS})
set(SAIGA_LIBRARY ${SAIGA_LIBRARIES})

# The include pathes of the dependencies.
set(INTERFACE_CORE_INCLUDES   ${SAIGA_INCLUDE_DIRS} "$<BUILD_INTERFACE:C:/Users/linda/Documents/AGPhys Projekt/ships/External/saiga/submodules/eigen>;$<INSTALL_INTERFACE:include/eigen3>;$<BUILD_INTERFACE:C:/Users/linda/Documents/AGPhys Projekt/ships/External/saiga/submodules/glfw/include>;$<INSTALL_INTERFACE:include>;$<BUILD_INTERFACE:C:/Users/linda/Documents/AGPhys Projekt/ships/External/saiga/submodules/zlib>;$<BUILD_INTERFACE:C:/Users/linda/Documents/AGPhys Projekt/ships/build/External/saiga/submodules/zlib>;$<BUILD_INTERFACE:C:/Users/linda/Documents/AGPhys Projekt/ships/External/saiga/submodules/libpng>;$<BUILD_INTERFACE:C:/Users/linda/Documents/AGPhys Projekt/ships/build/External/saiga/submodules/libpng>;$<BUILD_INTERFACE:C:/Users/linda/Documents/AGPhys Projekt/ships/External/saiga/submodules/assimp/code/../include>;$<BUILD_INTERFACE:C:/Users/linda/Documents/AGPhys Projekt/ships/build/External/saiga/submodules/assimp/code/../include>;$<INSTALL_INTERFACE:include>;$<BUILD_INTERFACE:C:/Users/linda/Documents/AGPhys Projekt/ships/build/External/saiga/submodules/glog>;$<BUILD_INTERFACE:C:/Users/linda/Documents/AGPhys Projekt/ships/External/saiga/submodules/glog/src>;$<INSTALL_INTERFACE:include>;$<BUILD_INTERFACE:C:/Users/linda/Documents/AGPhys Projekt/ships/External/saiga/submodules/glog/src/windows>")
set(INTERFACE_EXTRA_INCLUDES  "")
set(INTERFACE_CUDA_INCLUDES   "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include;C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include;C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include;C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include")
set(INTERFACE_OPENGL_INCLUDES "")
set(INTERFACE_VISION_INCLUDES "")
set(INTERFACE_VULKAN_INCLUDES "")

set(INTERFACE_CORE_LIBARIES "DbgHelp")



# Creates a target for each module
macro(saiga_make_target _target _target_lib _libraries _soname _includes)

add_library(${_target} SHARED IMPORTED)

#message(STATUS ${_target})
#message(STATUS ${_target_lib})
#message(STATUS ${_libraries})
#message(STATUS ${_soname})

set_target_properties(${_target} PROPERTIES
  INTERFACE_COMPILE_FEATURES "cxx_std_14"
  INTERFACE_INCLUDE_DIRECTORIES "${_includes}"
  INTERFACE_LINK_LIBRARIES "${${_libraries}}"
)


# Import target "saiga" for configuration "RelWithDebInfo"
set_property(TARGET ${_target} APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(${_target} PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_target_lib}"
  IMPORTED_SONAME_RELWITHDEBINFO "${_soname}"
  )

endmacro()

if(1)
saiga_make_target(Saiga::saiga_core   "${SAIGA_CORE_LIBRARY}"   SAIGA_CORE_LIBRARY "saiga_core.so" "${INTERFACE_CORE_INCLUDES}")
endif()
if(1)
saiga_make_target(Saiga::saiga_vision "${SAIGA_VISION_LIBRARY}" SAIGA_VISION_LIBRARY "saiga_vision.so" "${INTERFACE_VISION_INCLUDES}")
endif()
if()
saiga_make_target(Saiga::saiga_vulkan "${SAIGA_VULKAN_LIBRARY}" SAIGA_VULKAN_LIBRARY "saiga_vulkan.so" "${INTERFACE_VULKAN_INCLUDES}")
endif()
if()
saiga_make_target(Saiga::saiga_extra  "${SAIGA_EXTRA_LIBRARY}"  SAIGA_EXTRA_LIBRARY "saiga_extra.so" "${INTERFACE_EXTRA_INCLUDES}")
endif()
if(1)
saiga_make_target(Saiga::saiga_opengl "${SAIGA_OPENGL_LIBRARY}" SAIGA_OPENGL_LIBRARY "saiga_opengl.so" "${INTERFACE_OPENGL_INCLUDES}")
endif()
if(1)
saiga_make_target(Saiga::saiga_cuda   "${SAIGA_CUDA_LIBRARY}" SAIGA_CUDA_LIBRARY "saiga_cuda.so" "${INTERFACE_CUDA_INCLUDES}")
endif()


