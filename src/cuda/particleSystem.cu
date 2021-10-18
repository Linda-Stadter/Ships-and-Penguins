#include "particleSystem.h"
#include "saiga/core/util/assert.h"


void ParticleSystem::setDevicePtr(void* particleVbo)
{
    d_particles = ArrayView<Particle>((Particle*) particleVbo, particleCount);
}


void ParticleSystem::update(float dt)
{
    CUDA_SYNC_CHECK_ERROR();
}
