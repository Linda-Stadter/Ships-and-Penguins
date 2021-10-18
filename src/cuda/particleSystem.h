#pragma once

#include "saiga/core/geometry/aabb.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/random.h"
#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/assets/simpleAssetObject.h"

#include "particle.h"

using Saiga::ArrayView;
using Saiga::CUDA::getBlockCount;

class SAIGA_ALIGN(16) ParticleSystem
{
   public:
    struct SimulationParameters
    {
    };
    SimulationParameters sparams;

   public:
    int particleCount;
    ArrayView<Particle> d_particles;

   public:
    ParticleSystem(int _particleCount);
    ~ParticleSystem();

    void update(float dt);
    void setDevicePtr(void* ptr);

    void renderGUI();
};
