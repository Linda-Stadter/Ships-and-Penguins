FILE(GLOB_RECURSE PROG_SRC  *.cpp)
FILE(GLOB_RECURSE PROG_HEADER  *.h)
FILE(GLOB_RECURSE CUDA_SRC  *.cu)

add_library(CudaParticleSimulation STATIC ${PROG_SRC} ${CUDA_SRC} ${PROG_HEADER})

target_link_libraries(CudaParticleSimulation ${LIB_TARGETS})
target_include_directories(CudaParticleSimulation PUBLIC ".")

set_target_properties(CudaParticleSimulation PROPERTIES CXX_STANDARD 17)
set_property(TARGET CudaParticleSimulation PROPERTY CUDA_ARCHITECTURES ${AGPHYS_CUDA_ARCH})
#set_target_properties(CudaParticleSimulation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(CudaParticleSimulation PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${AGPHYS_CUDA_FLAGS}>)



