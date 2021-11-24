#include "particleSystem.h"
#include "saiga/cuda/device_helper.h"
#include "saiga/core/util/assert.h"

#include "saiga/core/math/Eigen_Compile_Checker.h"

__host__ void checkError(cudaError_t err);

ParticleSystem::ParticleSystem(int _particleCount, vec3 _boxMin, vec3 _boxDim)
    : particleCount(_particleCount), boxMin(_boxMin), boxDim(_boxDim)
{
    cudaStreamCreate(&stream1); cudaStreamCreate(&stream2); cudaStreamCreate(&stream3);

    checkError(cudaMalloc((void **)&d_constraintList, sizeof(int) * maxConstraintNum*2));
    checkError(cudaMalloc((void **)&d_constraintCounter, sizeof(int) * 1));
    checkError(cudaMalloc((void **)&d_constraintListWalls, sizeof(int) * maxConstraintNumWalls*2));
    checkError(cudaMalloc((void **)&d_constraintCounterWalls, sizeof(int) * 1));
    checkError(cudaMalloc((void **)&d_rayHitCount, sizeof(int) * 1));

    float minCellSize = 2.0 * maxParticleRadius;
    cellSize = minCellSize;
    cellDim = {int(ceil(boxDim[0] / cellSize)), int(ceil(boxDim[1] / cellSize)), int(ceil(boxDim[2] / cellSize))};
    cellCount = cellDim[0] * cellDim[1] * cellDim[2];

    checkError(cudaMalloc((void**)&d_particle_list, sizeof(int) * particleCount));
	checkError(cudaMalloc((void**)&d_cell_list, sizeof(int) * cellCount));

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

    checkError(cudaFree(d_particle_list));
	checkError(cudaFree(d_cell_list));

    std::cout << "~ParticleSystem" << std::endl;
}
