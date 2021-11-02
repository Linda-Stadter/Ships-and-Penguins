#include "particleSystem.h"
#include "saiga/core/util/assert.h"

//#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/device_helper.h"


void ParticleSystem::setDevicePtr(void* particleVbo)
{
    d_particles = ArrayView<Particle>((Particle*) particleVbo, particleCount);
}

// 1.1

__global__ void updateParticles(float dt, vec3 gravity, Saiga::ArrayView<Particle>particles) {
    Saiga::CUDA::ThreadInfo<> ti;
    if(ti.thread_id >= particles.size())
        return;
    Particle &p = particles[ti.thread_id];

    p.momentum += p.d_momentum;
    p.d_momentum = {0,0,0};
    p.position += dt * p.momentum * p.massinv;
    p.momentum += dt * gravity / p.massinv;
}

//__global__ void collisionWalls(float dt, Saiga::ArrayView<Particle> particles, Saiga::Plane wall, float elast_const, float spring_const, float frict_const);
__global__ void collisionWalls(float dt, Saiga::ArrayView<Particle> particles, Saiga::ArrayView<Saiga::Plane> d_walls, float elast_const, float spring_const, float frict_const);
__global__ void collisionParticles(float dt, Saiga::ArrayView<Particle>particles, float elast_const, float spring_const, float frict_const);

void ParticleSystem::update(float dt)
{
    collisionWalls<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, d_walls, elast_const, spring_const, frict_const);
    //CUDA_SYNC_CHECK_ERROR();
    collisionParticles<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, elast_const, spring_const, frict_const);
    //CUDA_SYNC_CHECK_ERROR();
    updateParticles<<<BLOCKS, BLOCK_SIZE>>>(dt, gravity, d_particles);
    //cudaDeviceSynchronize();
    CUDA_SYNC_CHECK_ERROR();
}

__global__ void resetParticles(int x, int z, vec3 corner, float distance, Saiga::ArrayView<Particle>particles) {
    Saiga::CUDA::ThreadInfo<> ti;
    if(ti.thread_id >= particles.size())
        return;
    Particle &p = particles[ti.thread_id];

    int xPos = (ti.thread_id) % x;
    int zPos = ((ti.thread_id - xPos) / x) % z;
    int yPos = (((ti.thread_id - xPos) / x) - zPos) / z;
    vec3 pos = {xPos, yPos, zPos};

    p.position = corner + pos * distance;

    p.momentum = {0, 0, 0};

    p.massinv = 1/1;
}

void ParticleSystem::reset(int x, int z, vec3 corner, float distance)
{
    resetParticles<<<BLOCKS, BLOCK_SIZE>>>(x, z, corner, distance, d_particles);
    CUDA_SYNC_CHECK_ERROR();
}

// 1.2

__device__ float collideSpherePlane(Particle &particle, Saiga::Plane &plane) {
    return particle.radius - (particle.position.dot(plane.normal) - plane.d);
}
__device__ vec3 elasticCollision(vec3 nc1, float eps_e, vec3 p1, vec3 p2, float inv_m1, float inv_m2) {
    return nc1 * ((1.0 + eps_e) * (p2 * inv_m2 - p1 * inv_m1).dot(nc1)) / (inv_m1 + inv_m2);
    // dp1e
}
__device__ vec3 springCollision(float d0, float dt, float k, vec3 nc1) {
    return d0 * k * nc1 * dt;
    // dp1s
}
__device__ vec3 frictionCollision(vec3 p1, vec3 p2, vec3 n1, float mu) {
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


__global__ void collisionWalls(float dt, Saiga::ArrayView<Particle> particles, Saiga::ArrayView<Saiga::Plane> d_walls, float elast_const, float spring_const, float frict_const) {
    Saiga::CUDA::ThreadInfo<> ti;
    if(ti.thread_id >= particles.size())
        return;
    Particle &p = particles[ti.thread_id];

    for (auto wall: d_walls) {
        float d0 = collideSpherePlane(p, wall);

        if (d0 > 0) {
            resolveCollision(p, wall, d0, dt, elast_const, spring_const, frict_const);
        }
    }
}

// 1.3

__device__ float collideSphereSphere(Particle &particleA, Particle &particleB) {
    return (particleA.radius + particleB.radius) - (particleA.position - particleB.position).norm();
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

__global__ void collisionParticles(float dt, Saiga::ArrayView<Particle>particles, float elast_const, float spring_const, float frict_const) {
    Saiga::CUDA::ThreadInfo<> ti;
    if(ti.thread_id >= particles.size())
        return;
    Particle &pa = particles[ti.thread_id];

    vec3 d_momentum = {0, 0, 0};

    for (int i = 0; i < particles.size(); i++) {
        if (i == ti.thread_id)
            continue;
        
        Particle &pb = particles[i];

        float d0 = collideSphereSphere(pa, pb);

        if (d0 > 0) {
            d_momentum += resolveCollision(pa, pb, d0, dt, elast_const, spring_const, frict_const);
        }
    }

    pa.d_momentum += d_momentum;


}