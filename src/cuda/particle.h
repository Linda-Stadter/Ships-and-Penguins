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
    vec3 momentum;
    vec3 d_momentum;
    vec3 velocity;
    float massinv;
};

struct SAIGA_ALIGN(16) ParticleCalc
{
    vec3 predicted;
    float radius;
};
