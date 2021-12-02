#pragma once

#include "saiga/core/geometry/aabb.h"
#include "saiga/core/geometry/ray.h"
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

    int particleCountRB = 0;
    int maxRigidBodyCount = 50;
    int rigidBodyCount = 0;
    RigidBody *d_rigidBodies;
    void constraintsShapeMatchingRB();

    int *d_constraintCounter;
    int *d_constraintList;
    int maxConstraintNum = particleCount*16;

    int *d_constraintCounterWalls;
    int *d_constraintListWalls;
    int maxConstraintNumWalls = particleCount * 4;

    vec3 gravity = {0, -9.81, 0};
    float elast_const = 0.2;
    float spring_const = 800;
    float frict_const = 0.1;

    bool jacobi = true;
    float dampV = 1.0;
    float relaxP = 0.25;
    int solverIterations = 2;
    bool useCalculatedRelaxP = true;

    float lastDt = 0;

    int *d_rayHitCount;

    // GUI
    const char* physics[6] = {"1.0 Force Based", "2.1 Force Based + Constraint Lists", "2.2 Position Based", "Force Based + Linked Cell", "3.0 Position Based + Linked Cell", "4.0 Rigid Body"};
    int physicsMode = 5;
    const char* actions[7] = {"Color", "Impulse", "Explode", "Implode", "Split", "Inflate", "Deflate"};
    int actionMode = 0;

    // action parameters
    vec4 color = {1,1,0,1};
    int explosionForce = 25;
    int splitCount = 10;

    vec3 boxDim;
    vec3 boxMin;
    ivec3 cellDim;
    int cellCount;
    int hashFunction = 1;
    // GUI
    const char* hashes[2] = {"random Hashing", "spatial Hashing"};

    float maxParticleRadius = 0.5;
    float cellSize;
    int* d_particle_list, * d_cell_list;

    // debug
    unsigned long steps = 0;
    cudaStream_t stream1, stream2, stream3;

   public:
    ParticleSystem(int _particleCount, vec3 boxMin, vec3 boxDim);
    ~ParticleSystem();

    const unsigned int BLOCK_SIZE = 128;
    const unsigned int BLOCKS     = Saiga::CUDA::getBlockCount(particleCount, BLOCK_SIZE);

    void update(float dt);
    void reset(int x, int z, vec3 corner, float distance, float randInitMul);
    void ray(Saiga::Ray ray);
    void setDevicePtr(void* ptr);

    void renderGUI();
};
