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
    ArrayView<Saiga::Plane> d_walls;

    vec3 gravity = {0, -9.81, 0};
    float elast_const = 0.2;
    float spring_const = 800;
    float frict_const = 0.1;

   public:
    ParticleSystem(int _particleCount);
    ~ParticleSystem();

    const unsigned int BLOCK_SIZE = 64;
    const unsigned int BLOCKS     = Saiga::CUDA::getBlockCount(particleCount, BLOCK_SIZE);

    void update(float dt);
    void reset(int x, int z, vec3 corner, float distance);
    void setDevicePtr(void* ptr);

    void renderGUI();
};
