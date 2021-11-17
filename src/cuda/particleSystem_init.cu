#include "particleSystem.h"
#include "saiga/cuda/device_helper.h"
#include "saiga/core/util/assert.h"

#include "saiga/core/math/Eigen_Compile_Checker.h"

__host__ void checkError(cudaError_t err);

ParticleSystem::ParticleSystem(int _particleCount)
    : particleCount(_particleCount)
{
    checkError(cudaMalloc((void **)&d_constraintList, sizeof (int) * maxConstraintNum*2));
    checkError(cudaMalloc((void **)&d_constraintCounter, sizeof (int) * 1));
    checkError(cudaMalloc((void **)&d_constraintListWalls, sizeof (int) * maxConstraintNumWalls*2));
    checkError(cudaMalloc((void **)&d_constraintCounterWalls, sizeof (int) * 1));
    checkError(cudaMalloc((void **)&d_rayHitCount, sizeof (int) * 1));
    std::cout << "ParticleSystem initialized!" << std::endl;
    CUDA_SYNC_CHECK_ERROR();
}

ParticleSystem::~ParticleSystem()
{
    checkError(cudaFree(d_constraintList));
    checkError(cudaFree(d_constraintCounter));
    checkError(cudaFree(d_constraintListWalls));
    checkError(cudaFree(d_constraintCounterWalls));
    checkError(cudaFree(d_rayHitCount));
    std::cout << "~ParticleSystem" << std::endl;
}
