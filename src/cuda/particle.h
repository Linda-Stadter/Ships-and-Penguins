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

    vec3 d_predicted;

    int fixed; // moved here to align velocity

    vec3 velocity; // used for both velocity and momentum
    float lambda; // moved here to align position
    vec3 position;

    int rbID; // only here for memory vector copy reasons
    vec3 sdf; // used for both sdf and curl

    vec3 d_momentum; // TODO rename to d_velocity (only used between vorticityAndViscosity steps) maybe use something else?
    float massinv;


    // 4.0
    //int rbID;
    vec3 relative; // used for fixed position of trochoidal wave circle center

    // 4.4
    //vec3 sdf; // used for both sdf and curl

    // 6
    //float lambda;
    //vec3 curl;

    int id; // cloth
};

struct SAIGA_ALIGN(16) ParticleCalc
{
    vec3 predicted;
    float radius;
};

struct SAIGA_ALIGN(16) ParticleCalc1
{
    vec3 velocity;
    float lambda; // buffer
    vec3 position;
    int rbID;
};

struct SAIGA_ALIGN(16) ParticleCalc2
{
    vec3 position;
    int rbID;
    vec3 sdf;
    vec3 d_momentum;
    float massinv;
    int buffer;
};

struct SAIGA_ALIGN(16) ParticleCalc3
{
    vec3 position;
    int rbID;
};

struct SAIGA_ALIGN(16) RigidBody
{
    int particleCount;
    vec3 originOfMass;
    mat3 A; // used for both A and later result Q
};
