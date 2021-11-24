#include "particleSystem.h"
#include "saiga/core/util/assert.h"

//#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/device_helper.h"

#include "saiga/cuda/memory.h"


void ParticleSystem::setDevicePtr(void* particleVbo) {
    d_particles = ArrayView<Particle>((Particle*) particleVbo, particleCount);
}

// 1.1

__global__ void updateParticles(float dt, vec3 gravity, Saiga::ArrayView<Particle>particles) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    Particle &p = particles[ti.thread_id];

    p.momentum += p.d_momentum;
    p.d_momentum = {0,0,0};
    p.position += dt * p.momentum * p.massinv;
    p.momentum += dt * gravity / p.massinv;

    // for compatibility of constraint list in 2.1
    p.predicted = p.position;
    // transition between physics
    p.velocity = p.momentum;
}

__global__ void updateParticlesPBD1(float dt, vec3 gravity, Saiga::ArrayView<Particle>particles, float dampV) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    Particle &p = particles[ti.thread_id];

    // p.velocity += dt * p.massinv * gravity; // falsch auf folie
    p.velocity += dt * gravity;
    // dampVelocities
    p.velocity *= dampV;

    p.predicted = p.position + dt * p.velocity;
}

__global__ void updateParticlesPBD2Iterator(float dt, Saiga::ArrayView<Particle>particles, float relaxP) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    Particle &p = particles[ti.thread_id];

    p.predicted += relaxP * p.d_predicted;
    // reset
    p.d_predicted = {0, 0, 0};
}
__global__ void updateParticlesPBD2(float dt, Saiga::ArrayView<Particle>particles, float relaxP) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    Particle &p = particles[ti.thread_id];

    //p.predicted += relaxP * p.d_predicted;
    p.velocity = (p.predicted - p.position) / dt;
    p.position = p.predicted;
    // reset
    p.d_predicted = {0, 0, 0};

    // velocity Update? hier? in eigenem kernel?
    // TODO friction restitution

    // transition between physics
    p.momentum = p.velocity;
}

__global__ void resetParticles(int x, int z, vec3 corner, float distance, Saiga::ArrayView<Particle>particles, float randInitMul) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    Particle &p = particles[ti.thread_id];

    int xPos = (ti.thread_id) % x;
    int zPos = ((ti.thread_id - xPos) / x) % z;
    int yPos = (((ti.thread_id - xPos) / x) - zPos) / z;
    vec3 pos = {xPos, yPos, zPos};
    //printf("%i, %i, %i; ", xPos, zPos, yPos);

    // pseudo random position offset
    int rand = ti.thread_id + p.position[0];
    p.position = corner + pos * distance + vec3{rand % 11, rand % 17, rand % 13} * randInitMul;

    p.momentum = {0, 0, 0};
    p.velocity = {0, 0, 0};
    p.massinv = 1/1;
    p.predicted = p.position;
    // 2.3
    p.color = {0, 1, 0, 1};
    p.radius = 0.5;
}

void ParticleSystem::reset(int x, int z, vec3 corner, float distance, float randInitMul) {
    resetParticles<<<BLOCKS, BLOCK_SIZE>>>(x, z, corner, distance, d_particles, randInitMul);
    CUDA_SYNC_CHECK_ERROR();
}

// 1.2
// positive overlap
inline __device__ float collideSpherePlane(float r, vec3 pos, Saiga::Plane &plane) {
    return r - (pos.dot(plane.normal) - plane.d);
    //return plane.sphereOverlap(particle.position, particle.radius);
}

// 1.3
// positive overlap
inline __device__ float collideSphereSphere(float r1, float r2, vec3 pos1, vec3 pos2) {
    return (r1 + r2) - (pos1 - pos2).norm();
}

inline __device__ vec3 elasticCollision(vec3 nc1, float eps_e, vec3 p1, vec3 p2, float inv_m1, float inv_m2) {
    return nc1 * ((1.0 + eps_e) * (p2 * inv_m2 - p1 * inv_m1).dot(nc1)) / (inv_m1 + inv_m2);
    // dp1e
}
inline __device__ vec3 springCollision(float d0, float dt, float k, vec3 nc1) {
    return d0 * k * nc1 * dt;
    // dp1s
}
inline __device__ vec3 frictionCollision(vec3 p1, vec3 p2, vec3 n1, float mu) {
    vec3 pt1 = p1 - p1.dot(n1) * n1;
    vec3 pt2 = p2 - p2.dot(n1) * n1;
    vec3 t1 = pt1.normalized();
    vec3 ptr = pt1 - pt2;
    float x = ptr.norm();
    float fx = mu / 2;
    if (x < mu) {
        fx = - (x * x) / (2 * mu) + x;
    }
    return - fx * t1;
    // dp1f
}

// plane particle
__device__ void resolveCollision(Particle &particle, Saiga::Plane &plane, float d0, float dt, float elast_const, float spring_const, float frict_const) {
    bool alive = particle.momentum.dot(plane.normal) * particle.massinv < 0;
    vec3 dp1e = {0, 0, 0};
    if (alive)
        dp1e = elasticCollision(plane.normal, elast_const, particle.momentum, {0,0,0}, particle.massinv, 0.0);
    vec3 dp1s = springCollision(d0, dt, spring_const, plane.normal);
    vec3 p1 = particle.momentum;
    vec3 p2 = {0, 0, 0};
    vec3 dp1f = frictionCollision(p1, p2, plane.normal, frict_const);

    particle.d_momentum += dp1e + dp1s + dp1f;
}

// particle particle
__device__ vec3 resolveCollision(Particle &particleA, Particle &particleB, float d0, float dt, float elast_const, float spring_const, float frict_const) {
    vec3 n1 = (particleA.position - particleB.position).normalized();
    bool alive = particleA.momentum.dot(n1) * particleA.massinv - particleB.momentum.dot(n1) * particleB.massinv < 0;
    vec3 dp1e = {0, 0, 0};
    if (alive)
        dp1e = elasticCollision(n1, elast_const, particleA.momentum, particleB.momentum, particleA.massinv, particleB.massinv);
    vec3 dp1s = springCollision(d0, dt, spring_const, n1);
    vec3 p1 = particleA.momentum;
    vec3 p2 = particleB.momentum;
    vec3 dp1f = frictionCollision(p1, p2, n1, frict_const);


    vec3 d_momentum = dp1e + dp1s + dp1f;
    return d_momentum;
}

__global__ void collisionWalls(float dt, Saiga::ArrayView<Particle> particles, Saiga::ArrayView<Saiga::Plane> d_walls, float elast_const, float spring_const, float frict_const) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    Particle &p = particles[ti.thread_id];

    for (auto wall: d_walls) {
        float d0 = collideSpherePlane(p.radius, p.position, wall);
        if (d0 > 0) {
            resolveCollision(p, wall, d0, dt, elast_const, spring_const, frict_const);
        }
    }
}

__global__ void collisionParticles(float dt, Saiga::ArrayView<Particle>particles, float elast_const, float spring_const, float frict_const) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    Particle pa = particles[ti.thread_id];

    vec3 d_momentum = {0, 0, 0};

    for (int i = 0; i < particles.size(); i++) {
        if (i == ti.thread_id)
            continue;
        
        Particle pb = particles[i];

        float d0 = collideSphereSphere(pa.radius, pb.radius, pa.position, pb.position);
        if (d0 > 0) {
            d_momentum += resolveCollision(pa, pb, d0, dt, elast_const, spring_const, frict_const);
        }
    }
    particles[ti.thread_id].d_momentum += d_momentum;
}

__global__ void resetConstraintCounter(int *constraintCounter, int *constraintCounterWalls) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= 1)
        return;
    *constraintCounter = 0;
    *constraintCounterWalls = 0;
}

__global__ void resetConstraints(int *constraints, int maxConstraintNum, int *constraintCounter, int *constraintCounterWalls) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id == 0) {
        *constraintCounter = 0;
        *constraintCounterWalls = 0;
    }
    if (ti.thread_id >= maxConstraintNum)
        return;
    
    constraints[ti.thread_id * 2 + 0] = -1;
    constraints[ti.thread_id * 2 + 1] = -1;
}

__global__ void resetCounter(int *counter) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= 1)
        return;
    *counter = 0;
}

__global__ void createConstraintParticles(Saiga::ArrayView<Particle>particles, int *constraints, int *constraintCounter, int maxConstraintNum) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    Particle pa = particles[ti.thread_id];

    for (int i = ti.thread_id + 1; i < particles.size(); i++) {        
        Particle pb = particles[i];

        float d0 = collideSphereSphere(pa.radius, pb.radius, pa.predicted, pb.predicted);
        if (d0 > 0) {
            int idx = atomicAdd(constraintCounter, 1);
            if (idx >= maxConstraintNum - 1) {
                *constraintCounter = maxConstraintNum;
                return;
            }
            constraints[idx*2 + 0] = ti.thread_id;
            constraints[idx*2 + 1] = i;
        }
    }
}

__global__ void createConstraintWalls(Saiga::ArrayView<Particle> particles, Saiga::ArrayView<Saiga::Plane> walls, int *constraints, int *constraintCounter, int maxConstraintNum) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    Particle p = particles[ti.thread_id];

    for (int i = 0; i < walls.size(); i++) {
        Saiga::Plane wall = walls[i];
        
        float d0 = collideSpherePlane(p.radius, p.predicted, wall);
        if (d0 > 0) {
            int idx = atomicAdd(constraintCounter, 1);
            if (idx >= maxConstraintNum - 1) {
                *constraintCounter = maxConstraintNum;
                return;
            }
            constraints[idx*2 + 0] = ti.thread_id;
            constraints[idx*2 + 1] = i;
        }
    }
}

// 2.1 just like resolveCollision but directly changes BOTH particles with atomic funcitons
__device__ void resolveConstraint(Particle &particleA, Particle &particleB, float d0, float dt, float elast_const, float spring_const, float frict_const) {
    vec3 n1 = (particleA.position - particleB.position).normalized();
    bool alive = particleA.momentum.dot(n1) * particleA.massinv - particleB.momentum.dot(n1) * particleB.massinv < 0;
    vec3 dp1e = {0, 0, 0};
    if (alive)
        dp1e = elasticCollision(n1, elast_const, particleA.momentum, particleB.momentum, particleA.massinv, particleB.massinv);
    vec3 dp1s = springCollision(d0, dt, spring_const, n1);
    vec3 p1 = particleA.momentum;
    vec3 p2 = particleB.momentum;
    vec3 dp1f = frictionCollision(p1, p2, n1, frict_const);


    vec3 d_momentum = dp1e + dp1s + dp1f;
    atomicAdd(&particleA.momentum[0], d_momentum[0]);
    atomicAdd(&particleA.momentum[1], d_momentum[1]);
    atomicAdd(&particleA.momentum[2], d_momentum[2]);

    atomicAdd(&particleB.momentum[0], -d_momentum[0]);
    atomicAdd(&particleB.momentum[1], -d_momentum[1]);
    atomicAdd(&particleB.momentum[2], -d_momentum[2]);
}

// 2.1 just like resolveCollision but directly changes BOTH particles with atomic funcitons
__device__ void resolveConstraint(Particle &particle, Saiga::Plane &plane, float d0, float dt, float elast_const, float spring_const, float frict_const) {
    bool alive = particle.momentum.dot(plane.normal) * particle.massinv < 0;
    vec3 dp1e = {0, 0, 0};
    if (alive)
        dp1e = elasticCollision(plane.normal, elast_const, particle.momentum, {0,0,0}, particle.massinv, 0.0);
    vec3 dp1s = springCollision(d0, dt, spring_const, plane.normal);
    vec3 p1 = particle.momentum;
    vec3 p2 = {0, 0, 0};
    vec3 dp1f = frictionCollision(p1, p2, plane.normal, frict_const);

    vec3 d_momentum = dp1e + dp1s + dp1f;
    atomicAdd(&particle.momentum[0], d_momentum[0]);
    atomicAdd(&particle.momentum[1], d_momentum[1]);
    atomicAdd(&particle.momentum[2], d_momentum[2]);
}

__global__ void resolveConstraintParticles(Saiga::ArrayView<Particle> particles, int *constraints, int *constraintCounter, int maxConstraintNum, float dt, float elast_const, float spring_const, float frict_const) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= *constraintCounter || ti.thread_id >= maxConstraintNum)
        return;
    int idxA = constraints[ti.thread_id*2 + 0];
    int idxB = constraints[ti.thread_id*2 + 1];
    Particle &pa = particles[idxA];
    Particle &pb = particles[idxB];

    float d0 = collideSphereSphere(pa.radius, pb.radius, pa.position, pb.position);
    resolveConstraint(pa, pb, d0, dt, elast_const, spring_const, frict_const);
}

__global__ void resolveConstraintWalls(Saiga::ArrayView<Particle> particles, Saiga::ArrayView<Saiga::Plane> walls, int *constraints, int *constraintCounter, int maxConstraintNum, float dt, float elast_const, float spring_const, float frict_const) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= *constraintCounter || ti.thread_id >= maxConstraintNum)
        return;
    int idxP = constraints[ti.thread_id*2 + 0];
    int idxW = constraints[ti.thread_id*2 + 1];
    Particle &p = particles[idxP];
    Saiga::Plane &w = walls[idxW];

    float d0 = collideSpherePlane(p.radius, p.position, w);
    resolveConstraint(p, w, d0, dt, elast_const, spring_const, frict_const);
}

// 2.2
__device__ vec3 resolvePBD(Particle &particleA, Particle &particleB) {
    vec3 p1 = particleA.predicted;
    vec3 p2 = particleB.predicted;
    float mi1 = particleA.massinv;
    float mi2 = particleB.massinv;
    float d = -collideSphereSphere(particleA.radius, particleB.radius, p1, p2); // float d = (p1-p2).norm() - (particleA.radius + particleB.radius);
    vec3 n = (p1 - p2).normalized();
    vec3 dx1 = - (mi1 / (mi1 + mi2)) * d * n;
    return dx1;
}

__device__ vec3 resolvePBD(Particle &particle, Saiga::Plane &wall) {
    float mi1 = particle.massinv;
    float mi2 = 0;
    float d = -collideSpherePlane(particle.radius, particle.predicted, wall);
    //float d = -wall.sphereOverlap(particle.predicted, particle.radius);
    vec3 n = wall.normal;
    vec3 dx1 = - (mi1 / (mi1 + mi2)) * d * n;
    return dx1;
}

__global__ void solverPBDParticles(Saiga::ArrayView<Particle> particles, int *constraints, int *constraintCounter, int maxConstraintNum, float relaxP, bool jacobi) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= *constraintCounter || ti.thread_id >= maxConstraintNum)
        return;
    int idxA = constraints[ti.thread_id*2 + 0];
    int idxB = constraints[ti.thread_id*2 + 1];
    Particle &pa = particles[idxA];
    Particle &pb = particles[idxB];


    ParticleCalc pa_copy;
    Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[idxA]), &pa_copy);
    ParticleCalc pb_copy;
    Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[idxB]), &pb_copy);

    float m1 = pa.massinv;
    float m2 = pb.massinv;

    float d = collideSphereSphere(pa_copy.radius, pb_copy.radius, pa_copy.predicted, pb_copy.predicted);
    vec3 n = (pa_copy.predicted - pb_copy.predicted).normalized();
    float m = (m1 / (m1 + m2));
    vec3 dx1 = m * d * n; //resolvePBD(pa_copy, pb_copy);
    vec3 dx2 = - (1 - m) * d * n;
    // jacobi integration mode: set predicted directly without using d_predicted and apply relax here to dx1
    if (jacobi) {
        atomicAdd(&pa.d_predicted[0], dx1[0]);
        atomicAdd(&pa.d_predicted[1], dx1[1]);
        atomicAdd(&pa.d_predicted[2], dx1[2]);

        atomicAdd(&pb.d_predicted[0], dx2[0]);
        atomicAdd(&pb.d_predicted[1], dx2[1]);
        atomicAdd(&pb.d_predicted[2], dx2[2]);
    } else { // Gauss-Seidel (race conditions)
        dx1 *= relaxP;
        dx2 *= relaxP;

        atomicAdd(&pa.predicted[0], dx1[0]);
        atomicAdd(&pa.predicted[1], dx1[1]);
        atomicAdd(&pa.predicted[2], dx1[2]);
        
        atomicAdd(&pb.predicted[0], dx2[0]);
        atomicAdd(&pb.predicted[1], dx2[1]);
        atomicAdd(&pb.predicted[2], dx2[2]);
    }
}

__global__ void solverPBDWalls(Saiga::ArrayView<Particle> particles, Saiga::ArrayView<Saiga::Plane> walls, int *constraints, int *constraintCounter, int maxConstraintNum, float relaxP, bool jacobi) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= *constraintCounter || ti.thread_id >= maxConstraintNum)
        return;
    int idxP = constraints[ti.thread_id*2 + 0];
    int idxW = constraints[ti.thread_id*2 + 1];
    Particle &p = particles[idxP];
    Saiga::Plane &w = walls[idxW];

    vec3 dx1 = resolvePBD(p, w);

    if (jacobi) {
        atomicAdd(&p.d_predicted[0], dx1[0]);
        atomicAdd(&p.d_predicted[1], dx1[1]);
        atomicAdd(&p.d_predicted[2], dx1[2]);
    } else {
        dx1 *= relaxP;
        atomicAdd(&p.predicted[0], dx1[0]);
        atomicAdd(&p.predicted[1], dx1[1]);
        atomicAdd(&p.predicted[2], dx1[2]);
    }
}

__global__ void reset_cell_list(int* cell_list, int cellCount) {
    Saiga::CUDA::ThreadInfo<> ti;

    if (ti.thread_id < cellCount) {
        cell_list[ti.thread_id] = -1;
    }
}

__device__ ivec3 calculate_cell_idx(vec3 position, float cellSize) {
    return (position / cellSize).cast<int>(); // incorrect but faster
    /*vec3 idxf(position / cellSize);
    idxf = {floor(idxf[0]), floor(idxf[1]), floor(idxf[2])};
    return idxf.cast<int>();*/
}

__device__ int calculate_hash_idx(ivec3 cell_idx, ivec3 cell_dims, int cellCount, int hashFunction) {
    int flat_cell_idx = -1;
    if  (hashFunction == 0) {
        const unsigned int p1 = 73856093;
        const unsigned int p2 = 19349663;
        const unsigned int p3 = 83492791;
        unsigned int i = cell_idx.x();
        unsigned int j = cell_idx.y();
        unsigned int k = cell_idx.z();
        flat_cell_idx = ((i * p1) ^ (j * p2) ^ (k * p3)) % (unsigned int)cellCount;
    } else if (hashFunction == 1) {
        int i2 = ((cell_idx.x() % cell_dims.x()) + cell_dims.x()) % cell_dims.x();
        int j2 = ((cell_idx.y() % cell_dims.y()) + cell_dims.y()) % cell_dims.y();
        int k2 = ((cell_idx.z() % cell_dims.z()) + cell_dims.z()) % cell_dims.z();
        flat_cell_idx = i2 * cell_dims.y() * cell_dims.z() + j2 * cell_dims.z() + k2;
    }
    return flat_cell_idx;
}

__global__ void createLinkedCells(Saiga::ArrayView<Particle> particles, int* cell_list, int* particle_list, ivec3 cell_dims, int cellCount, float cellSize, int hashFunction) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id < particles.size()) {
        ivec3 cell_idx = calculate_cell_idx(particles[ti.thread_id].position, cellSize);
        int flat_cell_idx = calculate_hash_idx(cell_idx, cell_dims, cellCount, hashFunction);
        particle_list[ti.thread_id] = atomicExch(&cell_list[flat_cell_idx], ti.thread_id);
    }
}

// NOTE: cell_dims is irrelevant if hashFunction = 0 (random Hasing) is used, only the cellCount N is relevant in this case

__global__ void createConstraintParticlesLinkedCells(Saiga::ArrayView<Particle> particles, int* cell_list, int* particle_list, int *constraints, int *constraintCounter, int maxConstraintNum, ivec3 cell_dims, int cellCount, float cellSize, int hashFunction) {
    Saiga::CUDA::ThreadInfo<> ti;

    if (ti.thread_id < particles.size()) {
        //Particle pa = particles[ti.thread_id];
        ParticleCalc pa;
        ParticleCalc pb;
        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[ti.thread_id]), &pa);

        ivec3 cell_idx = calculate_cell_idx(pa.predicted, cellSize); // actually pa.position but we only load predicted and its identical here
        
        /*for (int x = -1; x <= 0; x++) {
            for (int y = -1; y <= 1; y++) {
                    if (x == 0 && y > 0)
                        break;
                for (int z = -1; z <= 1; z++) {
                    if (x == 0 && y == 0 && z > 0)
                        break;
                    ivec3 neighbor_cell_idx = cell_idx + ivec3(x, y, z);
                    int neighbor_flat_idx = calculate_hash_idx(neighbor_cell_idx, cell_dims, cellCount, hashFunction);
                    int neighbor_particle_idx = cell_list[neighbor_flat_idx];
                    while (neighbor_particle_idx != -1) {
                        //Particle pb = particles[neighbor_particle_idx];
                        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[neighbor_particle_idx]), &pb);
                        // Exclude current particle (r = 0) from force calculation
                        if (!(x == 0 && y == 0 && z == 0) || neighbor_particle_idx > ti.thread_id) {
                            float d0 = collideSphereSphere(pa.radius, pb.radius, pa.predicted, pb.predicted);
                            if (d0 > 0) {
                                int idx = atomicAdd(constraintCounter, 1);
                                if (idx >= maxConstraintNum - 1) {
                                    *constraintCounter = maxConstraintNum;
                                    return;
                                }
                                constraints[idx*2 + 0] = ti.thread_id; // = tid
                                constraints[idx*2 + 1] = neighbor_particle_idx;
                            }
                        }
                        // Follow linked list
                        neighbor_particle_idx = particle_list[neighbor_particle_idx];
                    }
                }
            }
        }*/
        
        static const int X_CONSTS[14] = {-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0};
        static const int Y_CONSTS[14] = {-1,-1,-1, 0, 0, 0, 1, 1, 1,-1,-1,-1, 0, 0};
        static const int Z_CONSTS[14] = {-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0};


        for (int i = 0; i < 14; i++) {
            int x = X_CONSTS[i];
            int y = Y_CONSTS[i];
            int z = Z_CONSTS[i];
            
            ivec3 neighbor_cell_idx = cell_idx + ivec3(x, y, z);
            int neighbor_flat_idx = calculate_hash_idx(neighbor_cell_idx, cell_dims, cellCount, hashFunction);
            int neighbor_particle_idx = cell_list[neighbor_flat_idx];
            while (neighbor_particle_idx != -1) {
                // Exclude current particle (r = 0) from force calculation
                if (i != 13 || neighbor_particle_idx > ti.thread_id) {
                    //Particle pb = particles[neighbor_particle_idx];
                    Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[neighbor_particle_idx]), &pb);
                    float d0 = collideSphereSphere(pa.radius, pb.radius, pa.predicted, pb.predicted);
                    if (d0 > 0) {
                        int idx = atomicAdd(constraintCounter, 1);
                        if (idx >= maxConstraintNum - 1) {
                            *constraintCounter = maxConstraintNum;
                            return;
                        }
                        constraints[idx*2 + 0] = ti.thread_id; // = tid
                        constraints[idx*2 + 1] = neighbor_particle_idx;
                    }
                }
                // Follow linked list
                neighbor_particle_idx = particle_list[neighbor_particle_idx];
            }
        }
    }
}

// remove unused constraints
/*struct remove_predicate_constraints
{
  __host__ __device__
  bool operator()(const int x)
  {
    return x == -1;
  }
};*/

void ParticleSystem::update(float dt) {
    lastDt = dt;
    if (physicsMode == 0) { // 1.0 Force Based
        // TODO dt reihenfolge anpassen
        collisionWalls<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, d_walls, elast_const, spring_const, frict_const);
        CUDA_SYNC_CHECK_ERROR();
        collisionParticles<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, elast_const, spring_const, frict_const);
        CUDA_SYNC_CHECK_ERROR();
        updateParticles<<<BLOCKS, BLOCK_SIZE>>>(dt, gravity, d_particles);
        CUDA_SYNC_CHECK_ERROR();
    } else if (physicsMode == 1) { // 2.1 Force Based + Constraint Lists
        resetConstraintCounter<<<1, 32>>>(d_constraintCounter, d_constraintCounterWalls);
        CUDA_SYNC_CHECK_ERROR();
        createConstraintParticles<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_constraintList, d_constraintCounter, maxConstraintNum);
        createConstraintWalls<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_walls, d_constraintListWalls, d_constraintCounterWalls, maxConstraintNumWalls);
        CUDA_SYNC_CHECK_ERROR();
        resolveConstraintParticles<<<Saiga::CUDA::getBlockCount(maxConstraintNum, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_constraintList, d_constraintCounter, maxConstraintNum, dt, elast_const, spring_const, frict_const);
        resolveConstraintWalls<<<Saiga::CUDA::getBlockCount(maxConstraintNumWalls, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_walls, d_constraintListWalls, d_constraintCounterWalls, maxConstraintNumWalls, dt, elast_const, spring_const, frict_const);
        CUDA_SYNC_CHECK_ERROR();
        updateParticles<<<BLOCKS, BLOCK_SIZE>>>(dt, gravity, d_particles);
        CUDA_SYNC_CHECK_ERROR();
    } else if (physicsMode == 2) { // 2.2 Position Based
        resetConstraintCounter<<<1, 32>>>(d_constraintCounter, d_constraintCounterWalls);
        CUDA_SYNC_CHECK_ERROR();
        createConstraintParticles<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_constraintList, d_constraintCounter, maxConstraintNum);
        createConstraintWalls<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_walls, d_constraintListWalls, d_constraintCounterWalls, maxConstraintNumWalls);
        CUDA_SYNC_CHECK_ERROR();
        updateParticlesPBD1<<<BLOCKS, BLOCK_SIZE>>>(dt, gravity, d_particles, dampV);
        CUDA_SYNC_CHECK_ERROR();
        // solver Iterations: project Constraints

        float calculatedRelaxP = relaxP;
        for (int i = 0; i < solverIterations; i++) {
            if (useCalculatedRelaxP) {
                calculatedRelaxP = 1 - pow(1 - calculatedRelaxP, 1.0/(i+1));
            }

            solverPBDParticles<<<Saiga::CUDA::getBlockCount(maxConstraintNum, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_constraintList, d_constraintCounter, maxConstraintNum, relaxP, jacobi);
            solverPBDWalls<<<Saiga::CUDA::getBlockCount(maxConstraintNumWalls, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_walls, d_constraintListWalls, d_constraintCounterWalls, maxConstraintNumWalls, relaxP, jacobi);
            CUDA_SYNC_CHECK_ERROR();

            updateParticlesPBD2Iterator<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, calculatedRelaxP);
            CUDA_SYNC_CHECK_ERROR();
        }

        updateParticlesPBD2<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, relaxP);
        CUDA_SYNC_CHECK_ERROR();
    } else if (physicsMode == 3) { // Force Based + Linked Cell
        resetConstraintCounter<<<1, 32>>>(d_constraintCounter, d_constraintCounterWalls);
        CUDA_SYNC_CHECK_ERROR();

        const unsigned int BLOCKS_CELLS = Saiga::CUDA::getBlockCount(cellCount, BLOCK_SIZE);
        reset_cell_list<<<std::max(BLOCKS_CELLS, BLOCKS), BLOCK_SIZE>>>(d_cell_list, cellCount);
        CUDA_SYNC_CHECK_ERROR();
        createLinkedCells<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_cell_list, d_particle_list, cellDim, cellCount, cellSize, hashFunction);
        CUDA_SYNC_CHECK_ERROR();
        createConstraintParticlesLinkedCells<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, hashFunction);

        createConstraintWalls<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_walls, d_constraintListWalls, d_constraintCounterWalls, maxConstraintNumWalls);
        CUDA_SYNC_CHECK_ERROR();
        resolveConstraintParticles<<<Saiga::CUDA::getBlockCount(maxConstraintNum, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_constraintList, d_constraintCounter, maxConstraintNum, dt, elast_const, spring_const, frict_const);
        resolveConstraintWalls<<<Saiga::CUDA::getBlockCount(maxConstraintNumWalls, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_walls, d_constraintListWalls, d_constraintCounterWalls, maxConstraintNumWalls, dt, elast_const, spring_const, frict_const);
        CUDA_SYNC_CHECK_ERROR();
        updateParticles<<<BLOCKS, BLOCK_SIZE>>>(dt, gravity, d_particles);
        CUDA_SYNC_CHECK_ERROR();
    } else if (physicsMode == 4) { // 3.0 Position Based + Linked Cell
        resetConstraintCounter<<<1, 32, 0, stream1>>>(d_constraintCounter, d_constraintCounterWalls);
        //resetConstraints<<<Saiga::CUDA::getBlockCount(maxConstraintNum, BLOCK_SIZE), BLOCK_SIZE, 0, stream1>>>(d_constraintList, maxConstraintNum, d_constraintCounter, d_constraintCounterWalls);
        //CUDA_SYNC_CHECK_ERROR();

        const unsigned int BLOCKS_CELLS = Saiga::CUDA::getBlockCount(cellCount, BLOCK_SIZE);
        reset_cell_list<<<BLOCKS_CELLS, BLOCK_SIZE, 0, stream2>>>(d_cell_list, cellCount);
        //CUDA_SYNC_CHECK_ERROR();
        createLinkedCells<<<BLOCKS, BLOCK_SIZE, 0, stream2>>>(d_particles, d_cell_list, d_particle_list, cellDim, cellCount, cellSize, hashFunction);
        //CUDA_SYNC_CHECK_ERROR();
        createConstraintParticlesLinkedCells<<<BLOCKS, BLOCK_SIZE, 0, stream2>>>(d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, hashFunction);

        createConstraintWalls<<<BLOCKS, BLOCK_SIZE, 0, stream1>>>(d_particles, d_walls, d_constraintListWalls, d_constraintCounterWalls, maxConstraintNumWalls);
        CUDA_SYNC_CHECK_ERROR();

        updateParticlesPBD1<<<BLOCKS, BLOCK_SIZE>>>(dt, gravity, d_particles, dampV);
        
        // TODO constraints
        //thrust::device_ptr<int> d = thrust::device_pointer_cast(d_constraintList);  
        
        //thrust::fill(d, d+N, 2);
        //int N = thrust::remove_if(d, d + maxConstraintNum, remove_predicate_constraints()) - d;

        CUDA_SYNC_CHECK_ERROR();
        // solver Iterations: project Constraints

        float calculatedRelaxP = relaxP;
        for (int i = 0; i < solverIterations; i++) {
            if (useCalculatedRelaxP) {
                calculatedRelaxP = 1 - pow(1 - calculatedRelaxP, 1.0/(i+1));
            }
            // TODO N -> maxConstraintNum
            solverPBDParticles<<<Saiga::CUDA::getBlockCount(maxConstraintNum, BLOCK_SIZE), BLOCK_SIZE, 0, stream1>>>(d_particles, d_constraintList, d_constraintCounter, maxConstraintNum, relaxP, jacobi);
            solverPBDWalls<<<Saiga::CUDA::getBlockCount(maxConstraintNumWalls, BLOCK_SIZE), BLOCK_SIZE, 0, stream2>>>(d_particles, d_walls, d_constraintListWalls, d_constraintCounterWalls, maxConstraintNumWalls, relaxP, jacobi);
            CUDA_SYNC_CHECK_ERROR();

            updateParticlesPBD2Iterator<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, calculatedRelaxP);
            CUDA_SYNC_CHECK_ERROR();
        }

        updateParticlesPBD2<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, relaxP);
        //CUDA_SYNC_CHECK_ERROR();
    }
    cudaDeviceSynchronize();

    steps += 1;
}

// 2.3 Ray
__global__ void rayList(Saiga::ArrayView<Particle> particles, Saiga::Ray ray, thrust::pair<int, float> *list, int *rayHitCount) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    
    Particle &particle = particles[ti.thread_id];
    vec3 z = ray.origin - particle.position;
    float dz = ray.direction.dot(z);
    float Q = (dz * dz) - z.dot(z) + particle.radius * particle.radius;

    if (Q > 0) {
        int idx = atomicAdd(rayHitCount, 1);
        list[idx].first = ti.thread_id;
        list[idx].second = -dz;
    }
}

__global__ void rayColor(Saiga::ArrayView<Particle> particles, Saiga::Ray ray, thrust::pair<int, float> *list, int *rayHitCount, int min, vec4 color) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= 1000)
        return;
    
    if (ti.thread_id == 0)
        particles[list[min].first].color = color;

    list[ti.thread_id].second = 0;
}

__global__ void rayImpulse(Saiga::ArrayView<Particle> particles, Saiga::Ray ray, thrust::pair<int, float> *list, int *rayHitCount, int min) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= 1000)
        return;
    
    if (ti.thread_id == 0)
        particles[list[min].first].velocity += ray.direction * 42;

    list[ti.thread_id].second = 0;
}

__global__ void rayInflate(Saiga::ArrayView<Particle> particles, Saiga::Ray ray, thrust::pair<int, float> *list, int *rayHitCount, int min, bool inflate, float maxParticleRadius) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= 1000)
        return;
    
    if (ti.thread_id == 0) {
        int idx = list[min].first;
        if (inflate) {
            if (particles[idx].radius * 2 > maxParticleRadius)
                return;
            particles[idx].radius *= 2;
            particles[idx].massinv /= 4;
        } else {
            particles[idx].radius /= 2;
            particles[idx].massinv *= 4;
        }
    }
    list[ti.thread_id].second = 0;
}

__global__ void rayRevert(Saiga::ArrayView<Particle> particles, Saiga::Ray ray, thrust::pair<int, float> *list, int *rayHitCount, int min) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= 1000)
        return;
    
    if (ti.thread_id == 0) {
        int idx = list[min].first;
        particles[idx].radius = 0.5;
        particles[idx].velocity = {0,0,0};
        particles[idx].momentum = {0,0,0};
        particles[idx].d_momentum = {0,0,0};
    }

    list[ti.thread_id].second = 0;
}

__global__ void rayExplosion(Saiga::ArrayView<Particle> particles, Saiga::Ray ray, thrust::pair<int, float> *list, int *rayHitCount, int min, bool explode, float explodeMult) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    
    if (ti.thread_id == list[min].first)
        return;

    Particle &explodeParticle = particles[list[min].first];
    Particle &particle = particles[ti.thread_id];
    vec3 dir = particle.position - explodeParticle.position;
    float d = dir.norm();
    if (!explode)
        explodeMult = -explodeMult;
    if (d < 4) {
        particle.velocity += 1.0 / (d) * dir * explodeMult;
        particle.d_momentum += 1.0 / (d) * dir * explodeMult;
    }

    list[ti.thread_id].second = 0;
}

__global__ void raySplit(Saiga::ArrayView<Particle> particles, Saiga::Ray ray, thrust::pair<int, float> *list, int *rayHitCount, int min, int splitCount) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= 1000)
        return;
    
    if (ti.thread_id == 0) {
        Particle &particle = particles[list[min].first];
        particle.radius /= 2;
        // 5 pseudo random indices
        float r = particle.radius;
        int randIdx = min * 1117 % 757 + (int)list[min].first % 137 + *rayHitCount % 17;

        for (int i = 0; i < splitCount; i++) {
            Particle &p = particles[(randIdx + i) % particles.size()];

            // reuse the current particle
            if (i == splitCount - 1) {
                p = particles[list[min].first];
            }

            // pseudo random offset
            int x = p.position[0] * 10.0 + p.velocity[0] * 100.0 + p.momentum[0] * 100.0;
            int y = p.position[1] * 10.0 + p.velocity[1] * 100.0 + p.momentum[1] * 100.0;
            int z = p.position[2] * 10.0 + p.velocity[2] * 100.0 + p.momentum[2] * 100.0;
            vec3 randOffset = vec3{x % 17, y % 17, z % 17} / 17;
            // min offset radius around original position
            if (randOffset[0] >= 0)
                randOffset[0] += r;
            else
                randOffset[0] -= r;
            if (randOffset[1] >= 0)
                randOffset[1] += r;
            else
                randOffset[1] -= r;
            if (randOffset[2] >= 0)
                randOffset[2] += r;
            else
                randOffset[2] -= r;
            // normalize
            randOffset /= 1 + r;
            // set attributes
            p.position = particle.position + randOffset * r * 2;
            p.predicted = particle.predicted + randOffset * r * 2;
            p.d_predicted = particle.d_predicted;
            p.radius = r;
            p.color = particle.color;
            p.velocity = particle.velocity;
            p.momentum = particle.momentum;
        }
    }

    list[ti.thread_id].second = 0;
}

// remove if
struct remove_predicate
{
  __host__ __device__
  bool operator()(const thrust::pair<int, float> x)
  {
    return x.second <= 0.001;
  }
};

// min element
struct compare_predicate
{
  __host__ __device__
  bool operator()(thrust::pair<int, float> a, thrust::pair<int, float> b)
  {
    return a.second < b.second;
  }
};

void ParticleSystem::ray(Saiga::Ray ray) {
    CUDA_SYNC_CHECK_ERROR();
    thrust::device_vector<thrust::pair<int, float>> d_vec(1000);
    //thrust::device_vector<float> d_vec2(1000);
    //resetCounter<<<1, 32>>>(d_rayHitCount);
    
    resetCounter<<<1, 32>>>(d_rayHitCount);
    CUDA_SYNC_CHECK_ERROR();
    rayList<<<BLOCKS, BLOCK_SIZE>>>(d_particles, ray, thrust::raw_pointer_cast(&d_vec[0]), d_rayHitCount);
    CUDA_SYNC_CHECK_ERROR();
    int N = thrust::remove_if(d_vec.begin(), d_vec.end(), remove_predicate()) - d_vec.begin();
    if (N == 0)
        return;
    int min = thrust::min_element(d_vec.begin(), d_vec.begin() + N, compare_predicate()) - d_vec.begin();

    if (actionMode == 0) {
        rayColor<<<BLOCKS, BLOCK_SIZE>>>(d_particles, ray, thrust::raw_pointer_cast(&d_vec[0]), d_rayHitCount, min, color);
    } else if (actionMode == 1) {
        rayImpulse<<<BLOCKS, BLOCK_SIZE>>>(d_particles, ray, thrust::raw_pointer_cast(&d_vec[0]), d_rayHitCount, min);
    } else if (actionMode == 2) {
        rayExplosion<<<BLOCKS, BLOCK_SIZE>>>(d_particles, ray, thrust::raw_pointer_cast(&d_vec[0]), d_rayHitCount, min, true, explosionForce);
    } else if (actionMode == 3) {
        rayExplosion<<<BLOCKS, BLOCK_SIZE>>>(d_particles, ray, thrust::raw_pointer_cast(&d_vec[0]), d_rayHitCount, min, false, explosionForce);
    } else if (actionMode == 4) {
        raySplit<<<BLOCKS, BLOCK_SIZE>>>(d_particles, ray, thrust::raw_pointer_cast(&d_vec[0]), d_rayHitCount, min, splitCount);
    } else if (actionMode == 5) {
        rayInflate<<<BLOCKS, BLOCK_SIZE>>>(d_particles, ray, thrust::raw_pointer_cast(&d_vec[0]), d_rayHitCount, min, true, maxParticleRadius);
        /*CUDA_SYNC_CHECK_ERROR();
        update(lastDt);
        CUDA_SYNC_CHECK_ERROR();
        rayRevert<<<BLOCKS, BLOCK_SIZE>>>(d_particles, ray, thrust::raw_pointer_cast(&d_vec[0]), d_rayHitCount, min);
        CUDA_SYNC_CHECK_ERROR();*/
    } else if (actionMode == 6) {
        rayInflate<<<BLOCKS, BLOCK_SIZE>>>(d_particles, ray, thrust::raw_pointer_cast(&d_vec[0]), d_rayHitCount, min, false, maxParticleRadius);
    }
    CUDA_SYNC_CHECK_ERROR();
}