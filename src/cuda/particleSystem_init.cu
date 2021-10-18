#include "particleSystem.h"
#include "saiga/cuda/device_helper.h"
#include "saiga/core/util/assert.h"

#include "saiga/core/math/Eigen_Compile_Checker.h"

ParticleSystem::ParticleSystem(int _particleCount)
    : particleCount(_particleCount)
{
    std::cout << "ParticleSystem initialized!" << std::endl;
    CUDA_SYNC_CHECK_ERROR();
}

ParticleSystem::~ParticleSystem()
{
    std::cout << "~ParticleSystem" << std::endl;
}
