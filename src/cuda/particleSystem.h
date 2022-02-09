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
    int active;
};

struct ClothBendingConstraint {
    int id1;
    int id2;
    int id3;
    int id4;
};

struct ShipInfo {
    int id;
    int rbID;
    int shipStart;
    int penguinStart;
    int clothStart;
    int clothEnd;
    int constraintsStart;
    int constraintsEnd;
    int penguinID;
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
    float particleRadiusWater = 0.3;
    float particleRadiusCloth = 0.5;
    
    int particleCountRB = 0;
    int maxRigidBodyCount = 50;
    int rigidBodyCount = 0;
    RigidBody *d_rigidBodies;
    void initRigidBodies(float distance, int scenario);

    void constraintsShapeMatchingRB();
    void updateRigidBodies();
    void computeScore();

    int loadObj(int rigidBodyCount, int particleCountRB, vec3 pos, vec3 rot, vec4 color, Saiga::UnifiedModel model, float scaling, float particleMass, float maxParticleCount, bool stripes);
    int loadBox(int rigidBodyCount, int particleCountRB, ivec3 dim, vec3 pos, vec3 rot, vec4 color, bool fixed, float mass, float scaling, float particleRadius, bool noSDF);
    void spawnShip(vec3 spawnPos, vec4 ship_color, Saiga::UnifiedModel shipModel, float scaling, float particleMass, float maxObjParticleCount);

    std::vector<ClothConstraint> clothConstraints;
    std::vector<ClothBendingConstraint> clothBendingConstraints;
    std::vector<ShipInfo> shipInfos;

    ShipInfo *d_shipInfos;
    int *d_shipInfosCounter;
    int maxShipNum = 20;

    int particleFishStart;
    int particleSwordfishStart;
    int fishID;

    int *d_constraintCounter;
    int *d_constraintList;
    int maxConstraintNum = particleCount * 16;

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

    int *d_enemyGridWeight;
    int *d_enemyGridId;
    int enemyGridDim = 20;
    float enemyGridCell = 0;

    vec3 gravity = {0, -9.81, 0};

    float damp_v = 1.0; //0.995;
    float relax_p = 0.25;
    int solver_iterations = 2;
    bool use_calculated_relax_p = true;

    bool test_bool = false;
    float test_float = 0.05;

    float cloth_break_distance = 2.0;

    // friction
    float mu_k = 0.1;
    float mu_s = 0.8;
    float mu_f = 0.8;

    float h = 1.0;
    float epsilon_spiky = 1e-5;
    float omega_lambda_relax = 4;

    float artificial_pressure_k = 0.1;
    int artificial_pressure_n = 4;
    float delta_q = 0.2; // 0.1 - 0.3

    float c_viscosity = 0.01;
    float epsilon_vorticity = 0.007;

    float particle_radius_rest_density = 0.25;

    vec3 wind_direction = {-1, 0, 0};
    float wind_speed = 0.7;

    // everything in seconds
    float last_dt = 0;
    float passed_time = 0;
    float max_time = 180;

    int *d_rayHitCount;
    int *d_score;
    int *d_particleHits;
    int *d_rbHits;

    // wave paramenters
    float wave_number = 20.0;
    float phase_speed = 0.2;
    float steepness = 0.1;

    // controls
    float control_forward = 0;
    float control_rotate = 0;
    void controlRigidBody(int rbID, float forward, float rotate, float dt);
    float control_cannonball = 0;

    vec3 ship_position = {0, 0, 0};
    vec3 camera_direction ={1, 0, 0};
    bool debug_shooting = false;

    // cannon parameters
    float cannonball_speed = 20;
    float cannon_timer = 120;
    float cannon_timer_reset = 120;

    int score = 0;
    bool game_over = false;
    bool regular_ball = true;

    // map parameters
    vec3 mapDim ={0, 0, 0};
    vec3 fluidDim ={0, 0, 0};

    // GUI
    const char* physics[1] = {"Physics"};
    int physics_mode = 0;
    const char* actions[5] = {"Color", "Impulse", "Explode", "Implode", "Print Info"};
    int action_mode = 4;

    // action parameters
    vec4 color = {1,1,0,1};
    int explosion_force = 25;
    int split_count = 10;

    vec3 boxDim;
    vec3 boxMin;
    ivec3 cellDim;
    int cellCount;

    float max_particle_radius = 0.5;
    float cellSize;
    int* d_particle_list;
    std::pair<int, int>* d_cell_list;
    int* d_particle_hash;

    // debug
    unsigned long steps = 0;
    cudaStream_t stream1, stream2, stream3;

   public:
    ParticleSystem(int _particleCount, vec3 boxMin, vec3 boxDim);
    ~ParticleSystem();

    const unsigned int BLOCK_SIZE = 128;
    const unsigned int BLOCKS     = Saiga::CUDA::getBlockCount(particleCount, BLOCK_SIZE);

    void update(float dt);
    void reset(int x, int z, vec3 corner, float distance, float randInitMul, int scenario, vec3 fluidDim, vec3 trochoidal1Dim, vec3 trochoidal2Dim, ivec2 layers);
    void ray(Saiga::Ray ray);
    void setDevicePtr(void* ptr);

    void renderGUI();
    void renderIngameGUI();
};
