FILE(GLOB PROG_SRC *.cpp)
FILE(GLOB PROG_HEADER *.h)

add_executable(${PROG_NAME} ${PROG_SRC} ${PROG_HEADER})
target_include_directories(${PROG_NAME} PUBLIC ".")
target_link_libraries(${PROG_NAME} CudaParticleSimulation)


#visual studio working dir
set_target_properties(${PROG_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_target_properties(${PROG_NAME} PROPERTIES CXX_STANDARD 17)
