#include "particleSystem.h"
#include "saiga/cuda/device_helper.h"
#include "saiga/core/util/assert.h"

#include "saiga/core/math/Eigen_Compile_Checker.h"

__host__ void checkError(cudaError_t err);

ParticleSystem::ParticleSystem(int _particleCount, vec3 _boxMin, vec3 _boxDim)
    : particleCount(_particleCount), boxMin(_boxMin), boxDim(_boxDim)
{
    cudaStreamCreate(&stream1); cudaStreamCreate(&stream2); cudaStreamCreate(&stream3);

    checkError(cudaMalloc((void **)&d_constraintList, sizeof(int) * maxConstraintNum * 2));
    checkError(cudaMalloc((void **)&d_constraintCounter, sizeof(int)));
    checkError(cudaMalloc((void **)&d_constraintListWalls, sizeof(int) * maxConstraintNumWalls * 2));
    checkError(cudaMalloc((void **)&d_constraintCounterWalls, sizeof(int)));
    checkError(cudaMalloc((void **)&d_rayHitCount, sizeof(int)));

    checkError(cudaMalloc((void **)&d_constraintListCloth, sizeof(ClothConstraint) * maxConstraintNumCloth));
    checkError(cudaMalloc((void **)&d_constraintCounterCloth, sizeof(int)));
    checkError(cudaMalloc((void **)&d_constraintListClothBending, sizeof(ClothBendingConstraint) * maxConstraintNumClothBending));
    checkError(cudaMalloc((void **)&d_constraintCounterClothBending, sizeof(int)));

    checkError(cudaMalloc((void **)&d_particleIdLookup, sizeof(int) * particleCount));
    
    checkError(cudaMalloc((void **)&d_rigidBodies, sizeof(RigidBody) * maxRigidBodyCount));

    float minCellSize = 2.0 * maxParticleRadius;
    cellSize = minCellSize;
    cellDim = {int(ceil(boxDim[0] / cellSize)), int(ceil(boxDim[1] / cellSize)), int(ceil(boxDim[2] / cellSize))};
    cellCount = cellDim[0] * cellDim[1] * cellDim[2];

    checkError(cudaMalloc((void**)&d_particle_list, sizeof(int) * particleCount));
	checkError(cudaMalloc((void**)&d_cell_list, sizeof(std::pair<int, int>) * cellCount));
	checkError(cudaMalloc((void**)&d_particle_hash, sizeof(int) * particleCount));

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

	checkError(cudaFree(d_rigidBodies));

    std::cout << "~ParticleSystem" << std::endl;
}
