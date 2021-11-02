#pragma once

#include "saiga/core/geometry/plane.h"
#include "saiga/core/math/math.h"
#include "saiga/cuda/cudaHelper.h"

using namespace Saiga;

struct SAIGA_ALIGN(16) Particle
{
    vec3 position;
    float radius;

    vec4 color;

    vec3 momentum;
    vec3 d_momentum;
    float massinv;
};
