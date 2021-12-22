#pragma once

#include "saiga/core/geometry/aabb.h"
#include "saiga/core/geometry/ray.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/random.h"
#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/assets/simpleAssetObject.h"

#include "particle.h"

#include <utility>

using Saiga::ArrayView;
using Saiga::CUDA::getBlockCount;

struct ClothConstraint {
    int first;
    int second;
    float dist;
};

struct ClothBendingConstraint {
    int id1;
    int id2;
    int id3;
    int id4;
};

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

    float particleRenderRadius = 0.5;
    float particleRadiusWater = 0.5;
    float particleRadiusCloth = 0.5;
    
    int particleCountRB = 0;
    int maxRigidBodyCount = 50;
    int rigidBodyCount = 0;
    RigidBody *d_rigidBodies;
    void initRigidBodies(float distance, int scenario);

    void constraintsShapeMatchingRB();
    void updateRigidBodies();

    int loadObj(int rigidBodyCount, int particleCountRB, vec3 pos, vec3 rot, vec4 color);
    int loadBox(int rigidBodyCount, int particleCountRB, ivec3 dim, vec3 pos, vec3 rot, vec4 color, bool fixed, float mass);

    int *d_constraintCounter;
    int *d_constraintList;
    int maxConstraintNum = particleCount*16;

    int *d_constraintCounterWalls;
    int *d_constraintListWalls;
    int maxConstraintNumWalls = particleCount * 4;

    int *d_constraintCounterCloth;
    ClothConstraint *d_constraintListCloth;
    int maxConstraintNumCloth = particleCount * 4;

    int *d_constraintCounterClothBending;
    ClothBendingConstraint *d_constraintListClothBending;
    int maxConstraintNumClothBending = particleCount * 4;

    int *d_particleIdLookup;

    vec3 gravity = {0, -9.81, 0};
    float elast_const = 0.2;
    float spring_const = 800;
    float frict_const = 0.1;

    bool jacobi = true;
    float dampV = 1.0;
    float relaxP = 0.25;
    int solverIterations = 2;
    bool useCalculatedRelaxP = true;

    bool testBool = true;
    float testFloat = 0.05;

    float h = 1.0;
    float epsilon_spiky = 0.001;
    float omega_lambda_relax = 200;

    float artificial_pressure_k = 0.1;
    int artificial_pressure_n = 4;
    float delta_q = 0.2; // 0.1 - 0.3

    float c_viscosity = 0.01;
    float epsilon_vorticity = 0.01;

    float particleRadiusRestDensity = 0.25;

    float lastDt = 0;

    int *d_rayHitCount;

    // GUI
    const char* physics[8] = {"1.0 Force Based", "2.1 Force Based + Constraint Lists", "2.2 Position Based", "Force Based + Linked Cell", "3.0 Position Based + Linked Cell", "4.0 Rigid Body", "5.0 Fluid", "Cloth"};
    int physicsMode = 6;
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
    int* d_particle_list;
    std::pair<int, int>* d_cell_list;
    int* d_particle_hash;


    // 4.4
    bool useSDF = true;

    // debug
    unsigned long steps = 0;
    cudaStream_t stream1, stream2, stream3;

   public:
    ParticleSystem(int _particleCount, vec3 boxMin, vec3 boxDim);
    ~ParticleSystem();

    const unsigned int BLOCK_SIZE = 128;
    const unsigned int BLOCKS     = Saiga::CUDA::getBlockCount(particleCount, BLOCK_SIZE);

    void update(float dt);
    void reset(int x, int z, vec3 corner, float distance, float randInitMul, int scenario);
    void ray(Saiga::Ray ray);
    void setDevicePtr(void* ptr);

    void renderGUI();
};
