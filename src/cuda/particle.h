#pragma once

#include "saiga/core/geometry/plane.h"
#include "saiga/core/math/math.h"
#include "saiga/cuda/cudaHelper.h"
#include <thrust/device_vector.h>

using namespace Saiga;

struct SAIGA_ALIGN(16) Particle
{
    vec3 predicted;
    float radius;

    vec4 color;

    vec3 position;
    vec3 d_predicted;
    vec3 d_momentum;
    vec3 velocity;
    float massinv;

    // 4.0
    int rbID;
    vec3 relative;

    // 4.4
    vec3 sdf;

    // 6
    float lambda;
    vec3 curl;
};

struct SAIGA_ALIGN(16) ParticleCalc
{
    vec3 predicted;
    float radius;
};

struct SAIGA_ALIGN(16) RigidBody
{
    int particleCount;
    vec3 originOfMass;
    mat3 A;
};
