#define _USE_MATH_DEFINES
#include <cmath>

#include "particleSystem.h"
#include "saiga/core/util/assert.h"

//#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/device_helper.h"

#include "saiga/cuda/memory.h"

// 4.0
#include <Eigen/Core>
#include "saiga/core/math/random.h"

#include "svd3_cuda.h"

// 4.4
#include "saiga/core/geometry/AccelerationStructure.h"
#include "saiga/core/geometry/intersection.h"

void ParticleSystem::setDevicePtr(void* particleVbo) {
    d_particles = ArrayView<Particle>((Particle*) particleVbo, particleCount);
}

// 1.1

__global__ void updateParticles(float dt, vec3 gravity, Saiga::ArrayView<Particle>particles) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    Particle &p = particles[ti.thread_id];

    if (!p.fixed) {

        // TODO refactor momentum replaced by velocity
        p.velocity += p.d_momentum;
        p.position += dt * p.velocity * p.massinv;
        p.velocity += dt * gravity / p.massinv;
    }
    // reset
    p.d_momentum = {0,0,0};

    // for compatibility of constraint list in 2.1
    p.predicted = p.position;
    // transition between physics
    // TODO refactor momentum replaced by velocity
}

__global__ void updateParticlesPBD1(float dt, vec3 gravity, Saiga::ArrayView<Particle>particles, float dampV) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    Particle &p = particles[ti.thread_id];

    if (p.fixed)
        return;

    // p.velocity += dt * p.massinv * gravity; // falsch auf folie
    p.velocity += dt * gravity;
    // dampVelocities
    p.velocity *= dampV;

    p.predicted = p.position + dt * p.velocity;
}

__global__ void updateParticlesPBD1_radius(float dt, vec3 gravity, Saiga::ArrayView<Particle>particles, float dampV, float particleRadiusWater, float particleRadiusCloth) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    Particle &p = particles[ti.thread_id];

    if (p.fixed)
        return;

    if (p.rbID == -2)
        p.radius = particleRadiusWater;
    else if (p.rbID == -3)
        p.radius = particleRadiusCloth;

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

    if (!p.fixed) {
        p.predicted += relaxP * p.d_predicted;
    }
    // reset
    p.d_predicted = {0, 0, 0};
}
__global__ void updateParticlesPBD2(float dt, Saiga::ArrayView<Particle>particles, float relaxP) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    Particle &p = particles[ti.thread_id];

    if (!p.fixed) {
        //p.predicted += relaxP * p.d_predicted;
        p.velocity = (p.predicted - p.position) / dt;
        p.position = p.predicted;
    }
    // reset
    p.d_predicted = {0, 0, 0};

    // velocity Update? hier? in eigenem kernel?
    // TODO friction restitution

    // transition between physics
    // TODO refactor momentum replaced by velocity

    // 6.2
    p.lambda = 0;
}

__global__ void resetParticles(int x, int z, vec3 corner, float distance, Saiga::ArrayView<Particle>particles, float randInitMul, float particleRenderRadius, int rbID, vec4 color) {
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

    // TODO refactor momentum replaced by velocity
    p.velocity = {0, 0, 0};
    p.massinv = 1.0/1.0;
    p.predicted = p.position;
    // 2.3
    p.color = color;
    p.radius = particleRenderRadius;

    p.fixed = false;

    // 4.0
    p.rbID = rbID;
    p.relative = {0,0,0};
    p.sdf = {0,0,0};

    // 6.0
    p.lambda = 0;

    p.id = ti.thread_id; // cloth
}

__global__ void initParticles(int startIdx, int count, int x, int z, vec3 corner, float distance, Saiga::ArrayView<Particle>particles, float randInitMul, float particleRenderRadius, int rbID, vec4 color, bool fixed=false, float mass=1.0) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    if (ti.thread_id < startIdx || ti.thread_id >= startIdx + count)
        return;

    int idx = ti.thread_id - startIdx;
    Particle &p = particles[ti.thread_id];

    int xPos = (idx) % x;
    int zPos = ((idx - xPos) / x) % z;
    int yPos = (((idx - xPos) / x) - zPos) / z;
    vec3 pos = {xPos, yPos, zPos};
    //printf("%i, %i, %i; ", xPos, zPos, yPos);

    // pseudo random position offset
    int rand = ti.thread_id + p.position[0];
    p.position = corner + pos * distance + vec3{rand % 11, rand % 17, rand % 13} * randInitMul;

    // TODO refactor momentum replaced by velocity
    p.velocity = {0, 0, 0};
    p.massinv = 1.0f/mass;
    p.predicted = p.position;
    // 2.3
    p.color = color;
    p.radius = particleRenderRadius;

    p.fixed = fixed;

    // 4.0
    p.rbID = rbID;
    p.relative = {0,0,0};
    p.sdf = {0,0,0};

    // 6.0
    p.lambda = 0;
}

// 4.0 TODO fix or remove
__global__ void initCuboidParticles(Saiga::ArrayView<Particle> particles, int id, vec3 pos, ivec3 dim, vec3 rot, vec4 color, int particleCountRB, RigidBody *rigidBodies) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id > 0)
        return;
    
    mat3 rotMat;
    rotMat = Eigen::AngleAxisf(rot.x(), vec3::UnitZ())
        * Eigen::AngleAxisf(rot.y(), vec3::UnitY())
        * Eigen::AngleAxisf(rot.z(), vec3::UnitZ());
    
    int count = dim.x() * dim.y() * dim.z();

    for (int i = 0; i < dim.x(); i++) {
        for (int j = 0; j < dim.y(); j++) {
            for (int k = 0; k < dim.z(); k++) {
                vec3 p = {i, j, k};
                p = rotMat * p;
                p += pos;
                particles[particleCountRB].position = p;
                particles[particleCountRB].predicted = p;
                particles[particleCountRB].rbID = id;

                particles[particleCountRB].color = color;

                //4.4
                ivec3 idx = {i, j, k};
                ivec3 dim2 = (dim/2);
                ivec3 dir;
                dir.x() = idx.x() < dim2.x() ? idx.x() - dim2.x() : dim2.x() - (dim.x() - idx.x() - 1);
                dir.y() = idx.y() < dim2.y() ? idx.y() - dim2.y() : dim2.y() - (dim.y() - idx.y() - 1);
                dir.z() = idx.z() < dim2.z() ? idx.z() - dim2.z() : dim2.z() - (dim.z() - idx.z() - 1);
                //float m = min(min(fabs(sdf.x()), fabs(sdf.y())), fabs(sdf.z()));

                ivec3 absdir = {abs(dir.x()), abs(dir.y()), abs(dir.z())};

                int minDir = max(max(absdir.x(), absdir.y()), absdir.z());
                vec3 sdf = {0,0,0};

                ivec3 dirSign = dir;
                dirSign.x() = dirSign.x() > 0 ? 1 : dirSign.x();
                dirSign.x() = dirSign.x() < 0 ? -1 : dirSign.x();
                dirSign.y() = dirSign.y() > 0 ? 1 : dirSign.y();
                dirSign.y() = dirSign.y() < 0 ? -1 : dirSign.y();
                dirSign.z() = dirSign.z() > 0 ? 1 : dirSign.z();
                dirSign.z() = dirSign.z() < 0 ? -1 : dirSign.z();

                if (absdir.x() == minDir)
                    sdf.x() = dirSign.x();
                if (absdir.y() == minDir)
                    sdf.y() = dirSign.y();
                if (absdir.z() == minDir)
                    sdf.z() = dirSign.z();


                int mx = min(i + 1, dim.x() - i);
                int my = min(j + 1, dim.y() - j);
                int mz = min(k + 1, dim.z() - k);

                float m = min(min(mx, my), mz);

                particles[particleCountRB].sdf = -m * normalize(sdf);// minus to point inwards

                printf("%i %i %i, %f, %f, %f, %f\n", i, j, k, sdf.x(), sdf.y(), sdf.z(), m);

                particleCountRB++;
            }
        }
    }

    rigidBodies[id].particleCount = count;
}

__global__ void initSingleRigidBodyParticle(Saiga::ArrayView<Particle> particles, int id, vec3 pos, vec3 sdf, vec4 color, int particleCountRB, RigidBody *rigidBodies, bool fixed=false, float mass=1.0) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id > 0)
        return;
    
    particles[particleCountRB].position = pos;
    particles[particleCountRB].predicted = pos;
    particles[particleCountRB].rbID = id;

    particles[particleCountRB].color = color;

    particles[particleCountRB].fixed = fixed;
    particles[particleCountRB].massinv = 1.0f/mass;

    // 4.4
    particles[particleCountRB].sdf = sdf;

    rigidBodies[id].particleCount++;
}

__global__ void initRigidBodyParticles(Saiga::ArrayView<Particle> particles, int particleCountRB, RigidBody *rigidBodies) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particleCountRB)
        return;
    
    particles[ti.thread_id].relative = particles[ti.thread_id].predicted - rigidBodies[particles[ti.thread_id].rbID].originOfMass;
}

// 4.4
int ParticleSystem::loadObj(int rigidBodyCount, int particleCountRB, vec3 pos, vec3 rot, vec4 color) {
    Saiga::UnifiedModel model("objs/teapot.obj");
    Saiga::UnifiedMesh mesh = model.CombinedMesh().first;
    std::vector<Triangle> triangles = mesh.TriangleSoup();
    // 1
    Saiga::AABB bb = mesh.BoundingBox(); // TODO model.BoundingBox() ?
    vec3 min = bb.min;
    vec3 max = bb.max;
    // 2
    // Schnittstellen
    float maxObjParticleCount = 40;
    float maxSize = bb.maxSize();
    //float sampleDistance = 0.1;
    float sampleDistance = maxSize / maxObjParticleCount;
    int count = 0;
    Saiga::AccelerationStructure::ObjectMedianBVH omBVH(triangles);

    if (useSDF) {
        // 3d voxel grid
        vec3 size = bb.Size() / sampleDistance;
        const int xDim = ceil(size.x());
        const int yDim = ceil(size.y());
        const int zDim = ceil(size.z());

        auto ***voxel = new std::pair<int, vec3>**[zDim];
        for(int i = 0; i < zDim; ++i) {
            voxel[i] = new std::pair<int, vec3>*[yDim];
            for(int j = 0; j < yDim; ++j) {
                voxel[i][j] = new std::pair<int, vec3>[xDim];
            }
        }
        // init voxels
        for (int x = 0; x < xDim; x++) {
            for (int y = 0; y < yDim; y++) {
                for (int z = 0; z < zDim; z++) {
                    vec3 ori = min + sampleDistance * vec3{(float)x, (float)y, (float)z}; // TODO cast
                    bool isInside = true;
                    for (float dx = -1; dx < 2; dx += 2) {
                        for (float dy = -1; dy < 2; dy += 2) {
                            for (float dz = -1; dz < 2; dz += 2) {
                                vec3 dir = {dx,dy,dz};
                                Saiga::Ray ray(dir, ori);
                                Saiga::Intersection::RayTriangleIntersection rti = omBVH.getClosest(ray);
                                if (!rti.valid)
                                    isInside = false;
                            }
                        }
                    }
                    if (isInside) {
                        count++;
                        voxel[z][y][x].first = 1;
                    } else {
                        voxel[z][y][x].first = 0;
                    }
                    // init border sdf
                    voxel[z][y][x].second = {0,0,0};
                    if (x == 0)
                        voxel[z][y][x].second[0] = +1;
                    else if (x == xDim-1)
                        voxel[z][y][x].second[0] = -1;
                    
                    if (y == 0)
                        voxel[z][y][x].second[1] = +1;
                    else if (y == yDim-1)
                        voxel[z][y][x].second[1] = -1;
                    
                    if (z == 0)
                        voxel[z][y][x].second[2] = +1;
                    else if (z == zDim-1)
                        voxel[z][y][x].second[2] = -1;
                }
            }
        }
        // calc distance field
        int i = 0;
        int changed = 1;
        while (changed) {
            i++;
            changed = 0;
            for (int x = 1; x < xDim-1; x++) {
                for (int y = 1; y < yDim-1; y++) {
                    for (int z = 1; z < zDim-1; z++) {
                        if (voxel[z][y][x].first == i) {
                            if (    voxel[z+1][y][x].first < i
                                ||  voxel[z-1][y][x].first < i
                                ||  voxel[z][y+1][x].first < i
                                ||  voxel[z][y-1][x].first < i
                                ||  voxel[z][y][x+1].first < i
                                ||  voxel[z][y][x-1].first < i)
                                continue;
                            voxel[z][y][x].first++;
                            changed++;
                        }
                    }
                }
            }
        }
        // calc derivative (normal)
        for (int x = 1; x < xDim-1; x++) {
            for (int y = 1; y < yDim-1; y++) {
                for (int z = 1; z < zDim-1; z++) {
                    if (voxel[z][y][x].first) {
                        float dz = voxel[z+1][y][x].first - voxel[z-1][y][x].first;
                        float dy = voxel[z][y+1][x].first - voxel[z][y-1][x].first;
                        float dx = voxel[z][y][x+1].first - voxel[z][y][x-1].first;
                        voxel[z][y][x].second = {dx, dy, dz};
                    }
                }
            }
        }

        for (int x = 0; x < xDim; x++) {
            for (int y = 0; y < yDim; y++) {
                for (int z = 0; z < zDim; z++) {
                    vec3 ori = min + sampleDistance * vec3{(float)x, (float)y, (float)z}; // TODO cast
                    if (voxel[z][y][x].first) {
                        count++;
                        float scaling = 1.0f;
                        vec3 position = pos + ori*(scaling / sampleDistance);
                        vec3 sdf = (float)voxel[z][y][x].first * normalize(voxel[z][y][x].second);
                        initSingleRigidBodyParticle<<<1, 32>>>(d_particles, rigidBodyCount, position, sdf, color, particleCountRB++, d_rigidBodies);
                    }
                }
            }
        }


        for(int i = 0; i < zDim; ++i) {
            for(int j = 0; j < yDim; ++j) {
                delete [] voxel[i][j];
            }
            delete [] voxel[i];
        }
        delete [] voxel;

    } else {
        
        for (float x = min.x(); x < max.x(); x += sampleDistance) {
            for (float y = min.y(); y < max.y(); y += sampleDistance) {
                for (float z = min.z(); z < max.z(); z += sampleDistance) {
                    vec3 ori = {x,y,z};
                    bool isInside = true;
                    for (float dx = -1; dx < 2; dx += 2) {
                        for (float dy = -1; dy < 2; dy += 2) {
                            for (float dz = -1; dz < 2; dz += 2) {
                                vec3 dir = {dx,dy,dz};
                                Saiga::Ray ray(dir, ori);
                                Saiga::Intersection::RayTriangleIntersection rti = omBVH.getClosest(ray);
                                if (!rti.valid)
                                    isInside = false;
                            }
                        }
                    }
                    if (isInside) {
                        count++;
                        float scaling = 1.0f;
                        vec3 position = pos + ori*(scaling / sampleDistance); // TODO 
                        initSingleRigidBodyParticle<<<1, 32>>>(d_particles, rigidBodyCount, position, vec3{0.f,0.f,0.f}, color, particleCountRB++, d_rigidBodies);
                    }
                }
            }
        }

    }
    return count;
}

// 4.4
int ParticleSystem::loadBox(int rigidBodyCount, int particleCountRB, ivec3 dim, vec3 pos, vec3 rot, vec4 color, bool fixed=false, float mass=1.0) {    
    vec3 min = {0,0,0};
    int count = 0;
    float sampleDistance = 1.0;

        // 3d voxel grid
        //vec3 size = bb.Size() / sampleDistance;
        const int xDim = dim.x();
        const int yDim = dim.y();
        const int zDim = dim.z();

        auto ***voxel = new std::pair<int, vec3>**[zDim];
        for(int i = 0; i < zDim; ++i) {
            voxel[i] = new std::pair<int, vec3>*[yDim];
            for(int j = 0; j < yDim; ++j) {
                voxel[i][j] = new std::pair<int, vec3>[xDim];
            }
        }
        // init voxels
        for (int x = 0; x < xDim; x++) {
            for (int y = 0; y < yDim; y++) {
                for (int z = 0; z < zDim; z++) {
                    
                    voxel[z][y][x].first = 1;

                    // init border sdf
                    voxel[z][y][x].second = {0,0,0};
                    if (x == 0)
                        voxel[z][y][x].second[0] = +1;
                    else if (x == xDim-1)
                        voxel[z][y][x].second[0] = -1;
                    
                    if (y == 0)
                        voxel[z][y][x].second[1] = +1;
                    else if (y == yDim-1)
                        voxel[z][y][x].second[1] = -1;
                    
                    if (z == 0)
                        voxel[z][y][x].second[2] = +1;
                    else if (z == zDim-1)
                        voxel[z][y][x].second[2] = -1;
                }
            }
        }
        // calc distance field
        int i = 0;
        int changed = 1;
        while (changed) {
            i++;
            changed = 0;
            for (int x = 1; x < xDim-1; x++) {
                for (int y = 1; y < yDim-1; y++) {
                    for (int z = 1; z < zDim-1; z++) {
                        if (voxel[z][y][x].first == i) {
                            if (    voxel[z+1][y][x].first < i
                                ||  voxel[z-1][y][x].first < i
                                ||  voxel[z][y+1][x].first < i
                                ||  voxel[z][y-1][x].first < i
                                ||  voxel[z][y][x+1].first < i
                                ||  voxel[z][y][x-1].first < i)
                                continue;
                            voxel[z][y][x].first++;
                            changed++;
                        }
                    }
                }
            }
        }
        // calc derivative (normal)
        for (int x = 1; x < xDim-1; x++) {
            for (int y = 1; y < yDim-1; y++) {
                for (int z = 1; z < zDim-1; z++) {
                    if (voxel[z][y][x].first) {
                        float dz = voxel[z+1][y][x].first - voxel[z-1][y][x].first;
                        float dy = voxel[z][y+1][x].first - voxel[z][y-1][x].first;
                        float dx = voxel[z][y][x+1].first - voxel[z][y][x-1].first;
                        voxel[z][y][x].second = {dx, dy, dz};
                    }
                }
            }
        }

        for (int x = 0; x < xDim; x++) {
            for (int y = 0; y < yDim; y++) {
                for (int z = 0; z < zDim; z++) {
                    vec3 ori = min + sampleDistance * vec3{(float)x, (float)y, (float)z}; // TODO cast
                    if (voxel[z][y][x].first) {
                        count++;
                        float scaling = 1.0f;
                        vec3 position = pos + ori*(scaling / sampleDistance);
                        vec3 sdf = (float)voxel[z][y][x].first * normalize(voxel[z][y][x].second);
                        initSingleRigidBodyParticle<<<1, 32>>>(d_particles, rigidBodyCount, position, sdf, color, particleCountRB++, d_rigidBodies, fixed, mass);
                    }
                }
            }
        }


        for(int i = 0; i < zDim; ++i) {
            for(int j = 0; j < yDim; ++j) {
                delete [] voxel[i][j];
            }
            delete [] voxel[i];
        }
        delete [] voxel;

    return count;
}

__global__ void caclulateRigidBodyOriginOfMass(Saiga::ArrayView<Particle> particles, int particleCountRB, RigidBody *rigidBodies) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particleCountRB)
        return;
    Particle &p = particles[ti.thread_id];
    if (p.rbID >= 0) {
        vec3 d_originOfMass = p.predicted / (float)rigidBodies[p.rbID].particleCount; // TODO position vs predicted
        atomicAdd(&rigidBodies[p.rbID].originOfMass[0], d_originOfMass[0]);
        atomicAdd(&rigidBodies[p.rbID].originOfMass[1], d_originOfMass[1]);
        atomicAdd(&rigidBodies[p.rbID].originOfMass[2], d_originOfMass[2]);
    }
}

__global__ void covariance(Saiga::ArrayView<Particle> particles, int particleCountRB, RigidBody *rigidBodies) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particleCountRB)
        return;
    Particle &p = particles[ti.thread_id];
    if (p.rbID >= 0) {
        //vec3 pc = p.position - rigidBodies[p.rbID].originOfMass;
        mat3 pcr = (p.predicted - rigidBodies[p.rbID].originOfMass) * p.relative.transpose(); // TODO position vs predicted

        //printf("%i, %f, %f, %f\n", p.rbID, rigidBodies[p.rbID].originOfMass[0], rigidBodies[p.rbID].originOfMass[1], rigidBodies[p.rbID].originOfMass[2]);

        atomicAdd(&rigidBodies[p.rbID].A(0,0), pcr(0,0));
        atomicAdd(&rigidBodies[p.rbID].A(0,1), pcr(0,1));
        atomicAdd(&rigidBodies[p.rbID].A(0,2), pcr(0,2));
        atomicAdd(&rigidBodies[p.rbID].A(1,0), pcr(1,0));
        atomicAdd(&rigidBodies[p.rbID].A(1,1), pcr(1,1));
        atomicAdd(&rigidBodies[p.rbID].A(1,2), pcr(1,2));
        atomicAdd(&rigidBodies[p.rbID].A(2,0), pcr(2,0));
        atomicAdd(&rigidBodies[p.rbID].A(2,1), pcr(2,1));
        atomicAdd(&rigidBodies[p.rbID].A(2,2), pcr(2,2));
    }
}
// TODO test if random atomicAdd order is faster

// TODO rename A to AQ used for both
__global__ void SVD(RigidBody *rigidBodies, int rigidBodyCount) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= rigidBodyCount)
        return;
    RigidBody &rb = rigidBodies[ti.thread_id];
    rb.A = svd3_cuda::pd(rb.A);

    // reset
    //if (ti.thread_id == 0)
        //printf("%i: %f, %f, %f\n", ti.thread_id, rb.originOfMass[0], rb.originOfMass[1], rb.originOfMass[2]);
}

__global__ void resolveRigidBodyConstraints(Saiga::ArrayView<Particle> particles, int particleCountRB, RigidBody *rigidBodies) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particleCountRB)
        return;
    Particle &p = particles[ti.thread_id];
    if (p.rbID >= 0) {
        // dx = (Q*r + c) - p
        //if (ti.thread_id == 0)
            //printf("%f, %f, %f, %f, %f, %f, %f, %f, %f\n", p.relative[0], p.relative[1], p.relative[2], rigidBodies[p.rbID].A(0,0), rigidBodies[p.rbID].A(0,1), rigidBodies[p.rbID].A(1,1), rigidBodies[p.rbID].originOfMass[0], rigidBodies[p.rbID].originOfMass[1], rigidBodies[p.rbID].originOfMass[2]);
        p.predicted += (rigidBodies[p.rbID].A * p.relative + rigidBodies[p.rbID].originOfMass) - p.predicted; // TODO position vs predicted
    }
}

__global__ void resetRigidBody(RigidBody *rigidBodies, int rigidBodyCount) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= rigidBodyCount)
        return;
    RigidBody &rb = rigidBodies[ti.thread_id];
    // reset
    rb.originOfMass = {0,0,0};
    rb.A = mat3::Zero().cast<float>();
}

__global__ void resetRigidBodyComplete(RigidBody *rigidBodies, int maxRigidBodyCount) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= maxRigidBodyCount)
        return;
    RigidBody &rb = rigidBodies[ti.thread_id];
    // reset
    rb.particleCount = 0;
    rb.originOfMass = {0,0,0};
    rb.A = mat3::Zero().cast<float>();
}

void ParticleSystem::constraintsShapeMatchingRB() {
    updateRigidBodies();

    resolveRigidBodyConstraints<<<BLOCKS, BLOCK_SIZE>>>(d_particles, particleCount, d_rigidBodies);
    CUDA_SYNC_CHECK_ERROR();    
}

void ParticleSystem::updateRigidBodies() {
    const unsigned int BLOCKS_RB = Saiga::CUDA::getBlockCount(rigidBodyCount, BLOCK_SIZE);

    resetRigidBody<<<BLOCKS_RB, BLOCK_SIZE>>>(d_rigidBodies, rigidBodyCount);
    CUDA_SYNC_CHECK_ERROR();

    caclulateRigidBodyOriginOfMass<<<BLOCKS, BLOCK_SIZE>>>(d_particles, particleCount, d_rigidBodies);
    CUDA_SYNC_CHECK_ERROR();
    covariance<<<BLOCKS, BLOCK_SIZE>>>(d_particles, particleCount, d_rigidBodies);
    CUDA_SYNC_CHECK_ERROR();
    SVD<<<BLOCKS_RB, BLOCK_SIZE>>>(d_rigidBodies, rigidBodyCount);
    CUDA_SYNC_CHECK_ERROR();
    
}

// TODO sehr haesslich
__global__ void deactivateNonRB(Saiga::ArrayView<Particle> particles) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    
    Particle &p = particles[ti.thread_id];
    if (p.rbID < 0) {
        p.position[1] += 1000000.0f;
        p.predicted[1] = p.position[1];
    }
}

void ParticleSystem::reset(int x, int z, vec3 corner, float distance, float randInitMul, int scenario) {
    int rbID = -1; // free particles
    vec4 color = {0.0f, 1.0f, 0.0f, 1.f};
    if (scenario >= 7) {
        color = {0.1f, 0.2f, 0.8f, 1.f};
        rbID = -2; // fluid
    }
    resetParticles<<<BLOCKS, BLOCK_SIZE>>>(x, z, corner, distance, d_particles, randInitMul, particleRenderRadius, rbID, color);
    CUDA_SYNC_CHECK_ERROR();

    if (scenario == 9) {
        initParticles<<<BLOCKS, BLOCK_SIZE>>>(0, 20*20*20, 20, 20, {-20, 0, -20}, distance, d_particles, randInitMul, particleRenderRadius, -2, {0.f, 0.f, 1.f, .1f}, false);
        initParticles<<<BLOCKS, BLOCK_SIZE>>>(20*20*20, 40*40*40, 40, 40, {0, 0, 0}, distance, d_particles, randInitMul, particleRenderRadius, -2, {0.f, .2f, .7f, .3f}, false);
        CUDA_SYNC_CHECK_ERROR();
    }
    
    const unsigned int BLOCKS_RB = Saiga::CUDA::getBlockCount(maxRigidBodyCount, BLOCK_SIZE);
    resetRigidBodyComplete<<<BLOCKS_RB, BLOCK_SIZE>>>(d_rigidBodies, maxRigidBodyCount);

    particleCountRB = 0;
    rigidBodyCount = 0;

    if (scenario == 7) {
        initParticles<<<BLOCKS, BLOCK_SIZE>>>(10000, 10*5*10, 10, 5, {-2, 0, -2}, 1, d_particles, randInitMul, particleRenderRadius, -1, {.0f, .0f, .0f, 1.f}, true);
        initParticles<<<BLOCKS, BLOCK_SIZE>>>(10000+10*5*10, 64, 4, 4, {0, 0, 10}, 1, d_particles, randInitMul, particleRenderRadius, -1, {.0f, .9f, .0f, 1.f}, false);
        CUDA_SYNC_CHECK_ERROR();
    }

    if (scenario == 8) {
        //initParticles<<<BLOCKS, BLOCK_SIZE>>>(10000, 19*2*10, 2, 19, {-1, 0, -20}, 1, d_particles, 0, particleRenderRadius, -1, {.0f, .0f, .0f, 1.f}, true);
        //initParticles<<<BLOCKS, BLOCK_SIZE>>>(11000, 19*2*10, 2, 19, {-1, 0, 1.5}, 1, d_particles, 0, particleRenderRadius, -1, {.0f, .0f, .0f, 1.f}, true);
    }

    if (scenario == 10) { // cloth
        rbID = -3; // free particles
        vec4 color = {1.0f, 1.0f, 1.0f, 1.f};
        resetParticles<<<BLOCKS, BLOCK_SIZE>>>(x, z, corner, distance, d_particles, randInitMul, particleRenderRadius, rbID, color);
        CUDA_SYNC_CHECK_ERROR();


        std::vector<ClothConstraint> clothConstraints(0);

        std::vector<ClothBendingConstraint> clothBendingConstraints(0);

        int dimX = 50;
        int dimZ = 50;

        for (int j = 0; j < dimZ; j++) {
            for (int i = 0; i < dimX; i++) {
                int idx = j * dimX + i;
                if (i < dimX - 1) {
                    clothConstraints.push_back({idx, idx+1, 1.0f * distance});
                }
                if (j < dimZ - 1) {
                    clothConstraints.push_back({idx, idx+dimX, 1.0f * distance});
                }
                if (j < dimZ - 1 && i < dimX - 1) {
                    if (i+j % 2)
                        clothConstraints.push_back({idx, idx+dimX+1, 1.4142f*distance});
                    else
                        clothConstraints.push_back({idx+dimX, idx+1, 1.4142f*distance});

                    clothBendingConstraints.push_back({idx+dimX+1, idx, idx+dimX, idx+1});
                }
            }
        }

        size_t clothConstraintSize = sizeof(clothConstraints[0]) * clothConstraints.size();
        size_t clothBendingConstraintSize = sizeof(clothBendingConstraints[0]) * clothBendingConstraints.size();

        int distanceConstraintCount = clothConstraints.size();
        int bendingConstraintCount = clothBendingConstraints.size();

        cudaMemcpy(d_constraintListCloth, clothConstraints.data(), clothConstraintSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_constraintListClothBending, clothBendingConstraints.data(), clothBendingConstraintSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_constraintCounterCloth, &distanceConstraintCount, sizeof(int) * 1, cudaMemcpyHostToDevice);
        cudaMemcpy(d_constraintCounterClothBending, &bendingConstraintCount, sizeof(int) * 1, cudaMemcpyHostToDevice);

        // box
        color = {1, 0, 0, 1};
        vec3 rot = {0,0,0};
        ivec3 dim = {10,10,10};
        vec3 pos = {-5, 0, -5};

        particleCountRB = dimX*dimZ;
        int objParticleCount = loadBox(rigidBodyCount++, particleCountRB, dim, pos, rot, color, false, 5);
        particleCountRB += dim.x() * dim.y() * dim.z();
    }

    if (scenario > 2 && scenario < 8)
        initRigidBodies(distance, scenario);

    if (scenario > 2 && scenario != 6 && scenario != 7 && scenario < 7)
        deactivateNonRB<<<BLOCKS, BLOCK_SIZE>>>(d_particles);
    CUDA_SYNC_CHECK_ERROR();
    // TODO
    resetRigidBody<<<BLOCKS, BLOCK_SIZE>>>(d_rigidBodies, rigidBodyCount);
    CUDA_SYNC_CHECK_ERROR();

    caclulateRigidBodyOriginOfMass<<<BLOCKS, BLOCK_SIZE>>>(d_particles, particleCountRB, d_rigidBodies);
    CUDA_SYNC_CHECK_ERROR();
    initRigidBodyParticles<<<BLOCKS, BLOCK_SIZE>>>(d_particles, particleCountRB, d_rigidBodies);
    CUDA_SYNC_CHECK_ERROR();

    resetRigidBody<<<BLOCKS, BLOCK_SIZE>>>(d_rigidBodies, rigidBodyCount);
    CUDA_SYNC_CHECK_ERROR();
}

// 4.0
void ParticleSystem::initRigidBodies(float distance, int scenario) {
    // spawn
    ivec3 dim;
    vec3 pos;
    vec3 rot;
    vec4 color;
    int objParticleCount;

    if (scenario != 3 && scenario != 5 && scenario != 7) {
        color = {.8, .6, .5, 1};

        pos = linearRand(vec3(-40, 20, -40), vec3(40, 30, 40));
        rot = {0,0,0};
        objParticleCount = loadObj(rigidBodyCount++, particleCountRB, pos, rot, color);
        particleCountRB += objParticleCount;
        printf("%i\n", objParticleCount);

        pos = {0, 70, 0};
        rot = {0,0,0};
        objParticleCount = loadObj(rigidBodyCount++, particleCountRB, pos, rot, color);
        particleCountRB += objParticleCount;
    }

    color = {1.0, 0., .0, 1};

    if (scenario == 5) {
        rot = {0,0,0};
        dim = {5,5,5};

        pos = {0, 30, 0};
        objParticleCount = loadBox(rigidBodyCount++, particleCountRB, dim, pos, rot, color);
        particleCountRB += dim.x() * dim.y() * dim.z();

        pos = {0, 20, 0};
        objParticleCount = loadBox(rigidBodyCount++, particleCountRB, dim, pos, rot, color, false, 10);
        particleCountRB += dim.x() * dim.y() * dim.z();

        pos = {0, 10, 0};
        objParticleCount = loadBox(rigidBodyCount++, particleCountRB, dim, pos, rot, color);
        particleCountRB += dim.x() * dim.y() * dim.z();

    } else {

        for (int i = 0; i < 20; i++) {
            ivec3 dim = linearRand(ivec3(3,3,3), ivec3(5,5,5));
            vec3 pos = linearRand(vec3(-30, 10, -30), vec3(30, 40, 30));
            vec3 rot = linearRand(vec3(0, 0, 0), vec3(M_PI_2, M_PI_2, M_PI_2));
            //initCuboidParticles<<<1, 32>>>(d_particles, rigidBodyCount++, pos, dim, rot, color, particleCountRB, d_rigidBodies);
            //CUDA_SYNC_CHECK_ERROR();
            objParticleCount = loadBox(rigidBodyCount++, particleCountRB, dim, pos, rot, color);
            particleCountRB += dim.x() * dim.y() * dim.z();
        }

    }

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
    // TODO refactor momentum replaced by velocity
    bool alive = particle.velocity.dot(plane.normal) * particle.massinv < 0;
    vec3 dp1e = {0, 0, 0};
    if (alive)
        dp1e = elasticCollision(plane.normal, elast_const, particle.velocity, {0,0,0}, particle.massinv, 0.0);
    vec3 dp1s = springCollision(d0, dt, spring_const, plane.normal);
    vec3 p1 = particle.velocity;
    vec3 p2 = {0, 0, 0};
    vec3 dp1f = frictionCollision(p1, p2, plane.normal, frict_const);

    particle.d_momentum += dp1e + dp1s + dp1f;
}

// particle particle
__device__ vec3 resolveCollision(Particle &particleA, Particle &particleB, float d0, float dt, float elast_const, float spring_const, float frict_const) {
    // TODO refactor momentum replaced by velocity
    vec3 n1 = (particleA.position - particleB.position).normalized();
    bool alive = particleA.velocity.dot(n1) * particleA.massinv - particleB.velocity.dot(n1) * particleB.massinv < 0;
    vec3 dp1e = {0, 0, 0};
    if (alive)
        dp1e = elasticCollision(n1, elast_const, particleA.velocity, particleB.velocity, particleA.massinv, particleB.massinv);
    vec3 dp1s = springCollision(d0, dt, spring_const, n1);
    vec3 p1 = particleA.velocity;
    vec3 p2 = particleB.velocity;
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
    // TODO refactor momentum replaced by velocity
    vec3 n1 = (particleA.position - particleB.position).normalized();
    bool alive = particleA.velocity.dot(n1) * particleA.massinv - particleB.velocity.dot(n1) * particleB.massinv < 0;
    vec3 dp1e = {0, 0, 0};
    if (alive)
        dp1e = elasticCollision(n1, elast_const, particleA.velocity, particleB.velocity, particleA.massinv, particleB.massinv);
    vec3 dp1s = springCollision(d0, dt, spring_const, n1);
    vec3 p1 = particleA.velocity;
    vec3 p2 = particleB.velocity;
    vec3 dp1f = frictionCollision(p1, p2, n1, frict_const);


    vec3 d_momentum = dp1e + dp1s + dp1f;
    atomicAdd(&particleA.velocity[0], d_momentum[0]);
    atomicAdd(&particleA.velocity[1], d_momentum[1]);
    atomicAdd(&particleA.velocity[2], d_momentum[2]);

    atomicAdd(&particleB.velocity[0], -d_momentum[0]);
    atomicAdd(&particleB.velocity[1], -d_momentum[1]);
    atomicAdd(&particleB.velocity[2], -d_momentum[2]);
}

// 2.1 just like resolveCollision but directly changes BOTH particles with atomic funcitons
__device__ void resolveConstraint(Particle &particle, Saiga::Plane &plane, float d0, float dt, float elast_const, float spring_const, float frict_const) {
    // TODO refactor momentum replaced by velocity
    bool alive = particle.velocity.dot(plane.normal) * particle.massinv < 0;
    vec3 dp1e = {0, 0, 0};
    if (alive)
        dp1e = elasticCollision(plane.normal, elast_const, particle.velocity, {0,0,0}, particle.massinv, 0.0);
    vec3 dp1s = springCollision(d0, dt, spring_const, plane.normal);
    vec3 p1 = particle.velocity;
    vec3 p2 = {0, 0, 0};
    vec3 dp1f = frictionCollision(p1, p2, plane.normal, frict_const);

    vec3 d_momentum = dp1e + dp1s + dp1f;
    atomicAdd(&particle.velocity[0], d_momentum[0]);
    atomicAdd(&particle.velocity[1], d_momentum[1]);
    atomicAdd(&particle.velocity[2], d_momentum[2]);
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
/*__device__ vec3 resolvePBD(Particle &particleA, Particle &particleB) {
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
}*/

__global__ void solverPBDParticles(Saiga::ArrayView<Particle> particles, int *constraints, int *constraintCounter, int maxConstraintNum, float relaxP, bool jacobi) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= *constraintCounter || ti.thread_id >= maxConstraintNum)
        return;
    int idxA = constraints[ti.thread_id*2 + 0];
    int idxB = constraints[ti.thread_id*2 + 1];
    Particle &pa = particles[idxA];
    Particle &pb = particles[idxB];

    if (pa.rbID == -2 && pb.rbID == -2) // deactivate for fluid
        return;


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
    vec3 dx2 = - (1.0f - m) * d * n;

    if (pa.fixed)
        dx2 *= 2.0;
    if (pb.fixed)
        dx1 *= 2.0;

    // jacobi integration mode: set predicted directly without using d_predicted and apply relax here to dx1
    if (jacobi) {
        if (!pa.fixed) {
            atomicAdd(&pa.d_predicted[0], dx1[0]);
            atomicAdd(&pa.d_predicted[1], dx1[1]);
            atomicAdd(&pa.d_predicted[2], dx1[2]);
        }
        if (!pb.fixed) {
            atomicAdd(&pb.d_predicted[0], dx2[0]);
            atomicAdd(&pb.d_predicted[1], dx2[1]);
            atomicAdd(&pb.d_predicted[2], dx2[2]);
        }
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

__global__ void solverPBDParticlesSDF(Saiga::ArrayView<Particle> particles, int *constraints, int *constraintCounter, int maxConstraintNum, float relaxP, bool jacobi, RigidBody *rigidBodies, float mu_k=0, float mu_s=0, float mu_f=0) {
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
 
    if (pa.rbID == -2 && pb.rbID == -2) // deactivate for fluid
        return;
    
    if (pa.rbID >= 0 || pb.rbID >= 0) {
        vec3 sdf1 = pa.sdf;
        vec3 sdf2 = pb.sdf;
        mat3 R;
        if (pa.rbID >= 0 && pb.rbID >= 0) {
            Particle pi;
            Particle pj;
            if (sdf1.norm() <= sdf2.norm()) {
                d = sdf1.norm();
                n = normalize(sdf1);
                R = rigidBodies[pa.rbID].A;
            } else {
                d = sdf2.norm();
                n = -normalize(sdf2);
                R = rigidBodies[pb.rbID].A;
            }
        } else if (pa.rbID >= 0) {
            d = sdf1.norm();
            n = normalize(sdf1);
            R = rigidBodies[pa.rbID].A;
        } else if (pb.rbID >= 0) {
            d = sdf2.norm();
            n = -normalize(sdf2);
            R = rigidBodies[pb.rbID].A;
        }
        n = R * -n;
        if (d <= 1.0) {
            // border particle
            d = collideSphereSphere(pa_copy.radius, pb_copy.radius, pa_copy.predicted, pb_copy.predicted); // TODO redundant?
            vec3 xij = -(pa_copy.predicted - pb_copy.predicted).normalized();
            if (xij.dot(n) < 0.f) {
                n = xij - 2.0f*(xij.dot(n))*n;
            } else {
                n = xij;
            }
        }
        n = -n;
    }

    float m = (m1 / (m1 + m2));
    vec3 dx1 = m * d * n; //resolvePBD(pa_copy, pb_copy);
    vec3 dx2 = - (1.0f - m) * d * n;

    // Friction
    // TODO put friction after fixed check and also set other dx to 0?
    //vec3 a = ((pa.predicted - pa.position) - (pb.predicted - pb.position));
    vec3 a = ((pa.position - pa.predicted) - (pb.position - pb.predicted));
    vec3 dx_orthogonal = a - (a.dot(n))*n; // a_orthogonal_n

    if (!dx_orthogonal.norm() < mu_s * d) {
        float min = mu_k * d / dx_orthogonal.norm();
        min = min <= 1.0 ? min : 1.0;
        dx_orthogonal *= min;
    }

    vec3 dx1_f = m * dx_orthogonal;
    vec3 dx2_f = - (1.0f - m) * dx_orthogonal;
    
    dx1 += dx1_f * mu_f;
    dx2 += dx2_f * mu_f;
    // END Friction

    if (pa.fixed)
        dx2 *= 2.0;
    if (pb.fixed)
        dx1 *= 2.0;


    // jacobi integration mode: set predicted directly without using d_predicted and apply relax here to dx1
    if (jacobi) {
        if (!pa.fixed) {
            atomicAdd(&pa.d_predicted[0], dx1[0]);
            atomicAdd(&pa.d_predicted[1], dx1[1]);
            atomicAdd(&pa.d_predicted[2], dx1[2]);
        }
        if (!pb.fixed) {
            atomicAdd(&pb.d_predicted[0], dx2[0]);
            atomicAdd(&pb.d_predicted[1], dx2[1]);
            atomicAdd(&pb.d_predicted[2], dx2[2]);
        }
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

__global__ void solverPBDWalls(Saiga::ArrayView<Particle> particles, Saiga::ArrayView<Saiga::Plane> walls, int *constraints, int *constraintCounter, int maxConstraintNum, float relaxP, bool jacobi, float mu_k=0, float mu_s=0, float mu_f=0) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= *constraintCounter || ti.thread_id >= maxConstraintNum)
        return;
    int idxP = constraints[ti.thread_id*2 + 0];
    int idxW = constraints[ti.thread_id*2 + 1];
    Particle &p = particles[idxP];
    Saiga::Plane &w = walls[idxW];

    if (p.fixed)
        return;

    //vec3 dx1 = resolvePBD(p, w);

    float m1 = p.massinv;
    float m2 = 0;
    float d = -collideSpherePlane(p.radius, p.predicted, w);
    //float d = -wall.sphereOverlap(particle.predicted, particle.radius);
    vec3 n = w.normal;
    float m = (m1 / (m1 + m2));
    vec3 dx1 = - m * d * n;

    // Friction
    //vec3 a = ((pa.predicted - pa.position) - (pb.predicted - pb.position));
    vec3 a = (p.position - p.predicted);
    vec3 dx_orthogonal = a - (a.dot(n))*n; // a_orthogonal_n

    if (!dx_orthogonal.norm() < mu_s * d) {
        float min = mu_k * d / dx_orthogonal.norm();
        min = min <= 1.0 ? min : 1.0;
        dx_orthogonal *= min;
    }

    vec3 dx1_f = m * dx_orthogonal;
    vec3 dx2_f = - (1.0f - m) * dx_orthogonal;
    
    dx1 += dx1_f * mu_f;
    // END Friction

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

__global__ void updateLookupTable(Saiga::ArrayView<Particle> particles, int *particleIdLookup) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    particleIdLookup[particles[ti.thread_id].id] = ti.thread_id;
}

__global__ void solverPBDCloth(Saiga::ArrayView<Particle> particles, ClothConstraint *constraints, int *constraintCounter, int maxConstraintNum, int *particleIdLookup) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= *constraintCounter || ti.thread_id >= maxConstraintNum)
        return;
    int idxA_ = constraints[ti.thread_id].first;
    int idxB_ = constraints[ti.thread_id].second;
    int idxA = particleIdLookup[idxA_];
    int idxB = particleIdLookup[idxB_];
    Particle &pa = particles[idxA];
    Particle &pb = particles[idxB];

    //if (pa.rbID != -3 || pb.rbID != -3)
    //    return;

    ParticleCalc pa_copy;
    Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[idxA]), &pa_copy);
    ParticleCalc pb_copy;
    Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[idxB]), &pb_copy);

    float m1 = pa.massinv;
    float m2 = pb.massinv;

    float d = collideSphereSphere(constraints[ti.thread_id].dist, 0, pa_copy.predicted, pb_copy.predicted);
    vec3 n = (pa_copy.predicted - pb_copy.predicted).normalized();
    float m = (m1 / (m1 + m2));
    vec3 dx1 = m * d * n; //resolvePBD(pa_copy, pb_copy);
    vec3 dx2 = - (1.0f - m) * d * n;

    if (pa.fixed)
        dx2 *= 2.0;
    if (pb.fixed)
        dx1 *= 2.0;

    // jacobi integration mode: set predicted directly without using d_predicted and apply relax here to dx1
    if (true) {
        if (!pa.fixed) {
            atomicAdd(&pa.d_predicted[0], dx1[0]);
            atomicAdd(&pa.d_predicted[1], dx1[1]);
            atomicAdd(&pa.d_predicted[2], dx1[2]);
        }
        if (!pb.fixed) {
            atomicAdd(&pb.d_predicted[0], dx2[0]);
            atomicAdd(&pb.d_predicted[1], dx2[1]);
            atomicAdd(&pb.d_predicted[2], dx2[2]);
        }
    }
}

__device__ void changePredicted(Particle &p, vec3 dx) {
    if (!p.fixed) {
        atomicAdd(&p.d_predicted[0], dx[0]);
        atomicAdd(&p.d_predicted[1], dx[1]);
        atomicAdd(&p.d_predicted[2], dx[2]);
    }
}

__global__ void solverPBDClothBending(Saiga::ArrayView<Particle> particles, ClothBendingConstraint *constraints, int *constraintCounter, int maxConstraintNum, int *particleIdLookup, float testFloat) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= *constraintCounter || ti.thread_id >= maxConstraintNum)
        return;
    int idx_[4] = { constraints[ti.thread_id].id1,
                    constraints[ti.thread_id].id2,
                    constraints[ti.thread_id].id3,
                    constraints[ti.thread_id].id4 };
    int idx[4] = {  particleIdLookup[idx_[0]],
                    particleIdLookup[idx_[1]],
                    particleIdLookup[idx_[2]],
                    particleIdLookup[idx_[3]] };

    vec3 p12 = (particles[idx[0]].predicted + particles[idx[1]].predicted) / 2.0f;
    //vec3 p12 = particles[idx[0]].predicted;

    vec3 p1 = particles[idx[0]].predicted - p12;
    vec3 p2 = particles[idx[1]].predicted - p12;
    vec3 p3 = particles[idx[2]].predicted - p12;
    vec3 p4 = particles[idx[3]].predicted - p12;
    //printf("%i %i; %i %i; %f %f %f, %f %f %f, %f %f %f, %f %f %f\n", idx_[1], idx_[2], idx[1], idx[2], particles[idx[1]].predicted.x(), particles[idx[1]].predicted.y(), particles[idx[1]].predicted.z(), particles[idx[2]].predicted.x(), particles[idx[2]].predicted.y(), particles[idx[2]].predicted.z(),
    //    p12.x(), p12.y(), p12.z(), p2.x(), p2.y(), p2.z());

    vec3 n1 = (p2.cross(p3)).normalized();
    vec3 n2 = (p2.cross(p4)).normalized();

    float epsilon = 1e-5;

    if (n1.norm() < epsilon || n2.norm() < epsilon)
        return;

    float d = n1.dot(n2);
    d = d > 1.0f ? 1.0f : d;
    d = d < -1.0f ? -1.0f : d;


    vec3 q3 = (p2.cross(n2) + n1.cross(p2)*d) / (p2.cross(p3).norm());
    vec3 q4 = (p2.cross(n1) + n2.cross(p2)*d) / (p2.cross(p4).norm());
    vec3 q2 = - (p3.cross(n2) + n1.cross(p3)*d) / (p2.cross(p3).norm()) - (p4.cross(n1) + n2.cross(p4)*d) / (p2.cross(p4).norm());
    vec3 q1 = -q2-q3-q4;

    //if (q1.norm() < epsilon || q2.norm() < epsilon || q3.norm() < epsilon || q4.norm() < epsilon)
    //    return;

    float norm2_1 = q1.norm() * q1.norm();
    float norm2_2 = q2.norm() * q2.norm();
    float norm2_3 = q3.norm() * q3.norm();
    float norm2_4 = q4.norm() * q4.norm();

    const float omega1 = 1.0f;
    float angle0 = M_PI;
    float sqrt_d2 = sqrtf(1.0f-d*d);

    float sum_omega_q = norm2_1 + norm2_2 + norm2_3 + norm2_4;
    sum_omega_q *= omega1;

    if (sum_omega_q < epsilon)
        return;

    /*float dp1 = -(omega1 * sqrt_d2 * (acosf(d) - angle0)) / (norm2_2 + norm2_3 + norm2_4);
    float dp2 = -(omega1 * sqrt_d2 * (acosf(d) - angle0)) / (norm2_1 + norm2_3 + norm2_4);
    float dp3 = -(omega1 * sqrt_d2 * (acosf(d) - angle0)) / (norm2_2 + norm2_1 + norm2_4);
    float dp4 = -(omega1 * sqrt_d2 * (acosf(d) - angle0)) / (norm2_2 + norm2_3 + norm2_1);*/

    float dp = - (omega1 * sqrt_d2 * (acosf(d) - angle0)) / (sum_omega_q);
    dp *= testFloat;

    float dp1 = -(omega1 * sqrt_d2 * (acosf(d) - angle0)) / (sum_omega_q);
    float dp2 = -(omega1 * sqrt_d2 * (acosf(d) - angle0)) / (sum_omega_q);
    float dp3 = -(omega1 * sqrt_d2 * (acosf(d) - angle0)) / (sum_omega_q);
    float dp4 = -(omega1 * sqrt_d2 * (acosf(d) - angle0)) / (sum_omega_q);

    //printf("%f %f %f, %f %f %f, %f %f %f, %f %f %f\n", n1.x(), n1.y(), n1.z(), q2.x(), q2.y(), q2.z(), q3.x(), q3.y(), q3.z(), q4.x(), q4.y(), q4.z());

    changePredicted(particles[idx[0]], dp * q1);
    changePredicted(particles[idx[1]], dp * q2);
    changePredicted(particles[idx[2]], dp * q3);
    changePredicted(particles[idx[3]], dp * q4);
}

__global__ void reset_cell_list(std::pair<int, int>* cell_list, int cellCount) {
    Saiga::CUDA::ThreadInfo<> ti;

    if (ti.thread_id < cellCount) {
        cell_list[ti.thread_id].first = -1;
    }
}

__global__ void reset_cell_list_opti(std::pair<int, int>* cell_list, int cellCount, int particleCount) {
    Saiga::CUDA::ThreadInfo<> ti;

    if (ti.thread_id < cellCount) {
        cell_list[ti.thread_id].first = particleCount;
        cell_list[ti.thread_id].second = 0;
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

__global__ void createLinkedCells(Saiga::ArrayView<Particle> particles, std::pair<int, int>* cell_list, int* particle_list, ivec3 cell_dims, int cellCount, float cellSize, int hashFunction) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id < particles.size()) {
        ivec3 cell_idx = calculate_cell_idx(particles[ti.thread_id].position, cellSize);
        int flat_cell_idx = calculate_hash_idx(cell_idx, cell_dims, cellCount, hashFunction);
        particle_list[ti.thread_id] = atomicExch(&cell_list[flat_cell_idx].first, ti.thread_id);
    }
}

__global__ void calculateHash(Saiga::ArrayView<Particle> particles, int* particle_hash, std::pair<int, int>* cell_list, int* particle_list, ivec3 cell_dims, int cellCount, float cellSize, int hashFunction) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id < particles.size()) {
        ivec3 cell_idx = calculate_cell_idx(particles[ti.thread_id].predicted, cellSize); // TODO predicted or position ??? others only use predicted
        int flat_cell_idx = calculate_hash_idx(cell_idx, cell_dims, cellCount, hashFunction);
        particle_hash[ti.thread_id] = flat_cell_idx;
    }
}

__global__ void createLinkedCellsOpti(Saiga::ArrayView<Particle> particles, int* particle_hash, std::pair<int, int>* cell_list, int* particle_list, ivec3 cell_dims, int cellCount, float cellSize, int hashFunction) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id < particles.size()) {
        int flat_cell_idx = particle_hash[ti.thread_id];
        //atomicMin(&cell_list[flat_cell_idx].first, ti.thread_id);
        //atomicAdd(&cell_list[flat_cell_idx].second, 1);

        // replace every
        //int end_idx = cell_list[neighbor_flat_idx].second + neighbor_particle_idx;
        // with
        //int end_idx = cell_list[neighbor_flat_idx].second + 1;

        if (ti.thread_id > 0) {
            if (flat_cell_idx != particle_hash[ti.thread_id - 1])
                cell_list[flat_cell_idx].first = ti.thread_id;
        } else {
            cell_list[flat_cell_idx].first = ti.thread_id;
        }

        if (ti.thread_id < particles.size() - 1) {
            if (flat_cell_idx != particle_hash[ti.thread_id + 1])
                cell_list[flat_cell_idx].second = ti.thread_id;
        } else {
            cell_list[flat_cell_idx].second = ti.thread_id;
        }
    }
}

//template<typename T>
//struct less : public thrust::binary_function<T,T,bool>
/*struct custom_hash_less : public thrust::binary_function
{
    __host__ __device__ bool operator()(const Particle &pa, const Particle &pb) const {
        //return lhs < rhs;
        ivec3 cell_idx1 = calculate_cell_idx(pa.predicted, cellSize);
        int hash1 = calculate_hash_idx(cell_idx1, cell_dims, cellCount, hashFunction);
        ivec3 cell_idx1 = calculate_cell_idx(pb.predicted, cellSize);
        int hash2 = calculate_hash_idx(cell_idx1, cell_dims, cellCount, hashFunction);
        return hash1 < hash2;
    }
};*/

// NOTE: cell_dims is irrelevant if hashFunction = 0 (random Hasing) is used, only the cellCount N is relevant in this case

__global__ void createConstraintParticlesLinkedCells(Saiga::ArrayView<Particle> particles, std::pair<int, int>* cell_list, int* particle_list, int *constraints, int *constraintCounter, int maxConstraintNum, ivec3 cell_dims, int cellCount, float cellSize, int hashFunction) {
    Saiga::CUDA::ThreadInfo<> ti;

    if (ti.thread_id < particles.size()) {
        //Particle pa = particles[ti.thread_id];
        ParticleCalc pa;
        ParticleCalc pb;
        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[ti.thread_id]), &pa);

        ivec3 cell_idx = calculate_cell_idx(pa.predicted, cellSize); // actually pa.position but we only load predicted and its identical here
        
        /*for (int x = -1; x <= 0; x++) {
            for (int y = -1; y <= 1; y++) {
                //    if (x == 0 && y > 0)
                //        break;
                for (int z = -1; z <= 1; z++) {
                //    if (x == 0 && y == 0 && z > 0)
                //        break;
                    ivec3 neighbor_cell_idx = cell_idx + ivec3(x, y, z);
                    int neighbor_flat_idx = calculate_hash_idx(neighbor_cell_idx, cell_dims, cellCount, hashFunction);
                    int neighbor_particle_idx = cell_list[neighbor_flat_idx].first;
                    while (neighbor_particle_idx != -1) {
                        //Particle pb = particles[neighbor_particle_idx];
                        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[neighbor_particle_idx]), &pb);
                        // Exclude current particle (r = 0) from force calculation
                        //if (!(x == 0 && y == 0 && z == 0) || neighbor_particle_idx > ti.thread_id) {
                        if (neighbor_particle_idx > ti.thread_id) {
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
            int neighbor_particle_idx = cell_list[neighbor_flat_idx].first;
            while (neighbor_particle_idx != -1) {

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

// 4.0
__global__ void createConstraintParticlesLinkedCellsRigidBodies(Saiga::ArrayView<Particle> particles, std::pair<int, int>* cell_list, int* particle_list, int *constraints, int *constraintCounter, int maxConstraintNum, ivec3 cell_dims, int cellCount, float cellSize, int hashFunction) {
    Saiga::CUDA::ThreadInfo<> ti;

    if (ti.thread_id < particles.size()) {
        //Particle pa = particles[ti.thread_id];
        ParticleCalc pa;
        ParticleCalc pb;
        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[ti.thread_id]), &pa);
        int rbIDa = particles[ti.thread_id].rbID;

        ivec3 cell_idx = calculate_cell_idx(pa.predicted, cellSize); // actually pa.position but we only load predicted and its identical here

        
        static const int X_CONSTS[14] = {-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0};
        static const int Y_CONSTS[14] = {-1,-1,-1, 0, 0, 0, 1, 1, 1,-1,-1,-1, 0, 0};
        static const int Z_CONSTS[14] = {-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0};


        for (int i = 0; i < 14; i++) {
            int x = X_CONSTS[i];
            int y = Y_CONSTS[i];
            int z = Z_CONSTS[i];
            
            ivec3 neighbor_cell_idx = cell_idx + ivec3(x, y, z);
            int neighbor_flat_idx = calculate_hash_idx(neighbor_cell_idx, cell_dims, cellCount, hashFunction);
            int neighbor_particle_idx = cell_list[neighbor_flat_idx].first;
            while (neighbor_particle_idx != -1) {

                int rbIDb = particles[neighbor_particle_idx].rbID;
                if ( (rbIDa == -1 || rbIDb == -1 || rbIDa != rbIDb) &&
                        (i != 13 || neighbor_particle_idx > ti.thread_id) ) {
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

__global__ void createConstraintParticlesLinkedCellsRigidBodiesFluid(Saiga::ArrayView<Particle> particles, std::pair<int, int>* cell_list, int* particle_list, int *constraints, int *constraintCounter, int maxConstraintNum, ivec3 cell_dims, int cellCount, float cellSize, int hashFunction) {
    Saiga::CUDA::ThreadInfo<> ti;

    if (ti.thread_id < particles.size()) {
        //Particle pa = particles[ti.thread_id];
        ParticleCalc pa;
        ParticleCalc pb;
        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[ti.thread_id]), &pa);
        int rbIDa = particles[ti.thread_id].rbID;

        ivec3 cell_idx = calculate_cell_idx(pa.predicted, cellSize); // actually pa.position but we only load predicted and its identical here


        /*for (int x = -1; x <= 1; x++) {
        //for (int x = -1; x <= 0; x++) {
            for (int y = -1; y <= 1; y++) {
                //    if (x == 0 && y > 0)
                //        break;
                for (int z = -1; z <= 1; z++) {
                //    if (x == 0 && y == 0 && z > 0)
                //        break;
                    ivec3 neighbor_cell_idx = cell_idx + ivec3(x, y, z);
                    int neighbor_flat_idx = calculate_hash_idx(neighbor_cell_idx, cell_dims, cellCount, hashFunction);
                    int neighbor_particle_idx = cell_list[neighbor_flat_idx].first;
                    int end_idx = cell_list[neighbor_flat_idx].second + 1;
                    for (; neighbor_particle_idx < end_idx; neighbor_particle_idx++) {
                        //Particle pb = particles[neighbor_particle_idx];
                        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[neighbor_particle_idx]), &pb);
                        // Exclude current particle (r = 0) from force calculation
                        //if (!(x == 0 && y == 0 && z == 0) || neighbor_particle_idx > ti.thread_id) {
                        int rbIDb = particles[neighbor_particle_idx].rbID;
                        if ( (rbIDa == -1 || rbIDb == -1 || rbIDa != rbIDb) && (neighbor_particle_idx > ti.thread_id) ) {
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
                        //neighbor_particle_idx = particle_list[neighbor_particle_idx];
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
            int neighbor_particle_idx = cell_list[neighbor_flat_idx].first;
            int end_idx = cell_list[neighbor_flat_idx].second + 1;
            for (; neighbor_particle_idx < end_idx; neighbor_particle_idx++) {

                int rbIDb = particles[neighbor_particle_idx].rbID;
                if ( (rbIDa == -1 || rbIDb == -1 || rbIDa != rbIDb) &&
                        (i != 13 || neighbor_particle_idx > ti.thread_id) ) {
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


// 6.2

__device__ __host__ float W_poly6(float r, float h) {
    if (r > h)
        return 0;
    float hd = h * h - r * r;
    float hd3 = hd * hd * hd;
    float h3 = h * h * h;
    float h9 = h3 * h3 * h3;
    return 315.f / (64.f * M_PI * h9) * hd3;
}

__device__ __host__ vec3 W_spiky(vec3 r, float h, float epsilon) {
    float d = r.norm();
    if (d <= epsilon || d > h)
        return {0, 0, 0};
    float hd = h - d;
    vec3 hd2 = r.normalized() * hd * hd;
    float h3 = h * h * h;
    float h6 = h3 * h3;
    return -45.f / (M_PI * h6) * hd2;
}

__global__ void computeDensityAndLambda(Saiga::ArrayView<Particle> particles, std::pair<int, int>* cell_list, int* particle_list, int *constraints, int *constraintCounter, int maxConstraintNum, ivec3 cell_dims, int cellCount, float cellSize, int hashFunction, float h, float epsilon_spiky, float omega_lambda_relax, float particleRadius) {
    Saiga::CUDA::ThreadInfo<> ti;

    const float m = 1.0;

    if (ti.thread_id < particles.size()) {
        //Particle pa = particles[ti.thread_id];
        int rbIDa = particles[ti.thread_id].rbID;
        if (rbIDa != -2)
            return;
        const float rho0inv = (8.0 * particleRadius * particleRadius * particleRadius);
        ParticleCalc pa;
        ParticleCalc pb;
        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[ti.thread_id]), &pa);

        ivec3 cell_idx = calculate_cell_idx(pa.predicted, cellSize); // actually pa.position but we only load predicted and its identical here

        float rho = 0;
        vec3 spiky_sum = {0, 0, 0};
        float lambda2 = 0;

        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                for (int z = -1; z <= 1; z++) {
                    ivec3 neighbor_cell_idx = cell_idx + ivec3(x, y, z);
                    int neighbor_flat_idx = calculate_hash_idx(neighbor_cell_idx, cell_dims, cellCount, hashFunction);
                    int neighbor_particle_idx = cell_list[neighbor_flat_idx].first;
                    int end_idx = cell_list[neighbor_flat_idx].second + 1;
                    for (; neighbor_particle_idx < end_idx; neighbor_particle_idx++) {
                        //Particle pb = particles[neighbor_particle_idx];
                        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[neighbor_particle_idx]), &pb);
                        // Exclude current particle (r = 0) from force calculation
                        //if (!(x == 0 && y == 0 && z == 0) || neighbor_particle_idx > ti.thread_id) {
                        int rbIDb = particles[neighbor_particle_idx].rbID;
                        //if (rbIDa == -2) { // TODO an anfang der for loop verschieben
                            //float d0 = collideSphereSphere(h, 0, pa.predicted, pb.predicted);
                            //if (d0 > 0) {
                                vec3 d_p = pa.predicted - pb.predicted;

                                float d_rho = m * W_poly6((d_p).norm(), h);
                                rho += d_rho;

                                vec3 spiky = W_spiky(d_p, h, epsilon_spiky) * rho0inv;
                                float spiky_norm = spiky.norm();
                                spiky_sum += spiky;
                                lambda2 += spiky_norm * spiky_norm;
                            //}
                        //}
                        // Follow linked list
                        //neighbor_particle_idx = particle_list[neighbor_particle_idx];
                    }
                }
            }
        }

        float C_density = rho * rho0inv - 1.0;
        float lambda1 = spiky_sum.norm();
        lambda1 *= lambda1;
        float lambda = -C_density / (lambda1 + lambda2 + omega_lambda_relax);

        //printf("%f, %f\n", lambda1, lambda2);

        particles[ti.thread_id].lambda = lambda;
    }
}

__global__ void updateParticlesPBD2IteratorFluid(Saiga::ArrayView<Particle> particles, std::pair<int, int>* cell_list, int* particle_list, int *constraints, int *constraintCounter, int maxConstraintNum, ivec3 cell_dims, int cellCount, float cellSize, int hashFunction, float h, float epsilon_spiky, float particleRadius, float artificial_pressure_k, int artificial_pressure_n, float w_poly_d_q) {
    Saiga::CUDA::ThreadInfo<> ti;


    if (ti.thread_id < particles.size()) {
        int rbIDa = particles[ti.thread_id].rbID;
        if (rbIDa != -2)
            return;
        const float rho0inv = (8.0 * particleRadius * particleRadius * particleRadius);
        //Particle pa = particles[ti.thread_id];
        ParticleCalc pa;
        ParticleCalc pb;
        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[ti.thread_id]), &pa);

        ivec3 cell_idx = calculate_cell_idx(pa.predicted, cellSize); // actually pa.position but we only load predicted and its identical here

        vec3 lambda_spiky = {0, 0, 0};

        //float w_poly_d_q = W_poly6(delta_q * h, h);

        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                for (int z = -1; z <= 1; z++) {
                    ivec3 neighbor_cell_idx = cell_idx + ivec3(x, y, z);
                    int neighbor_flat_idx = calculate_hash_idx(neighbor_cell_idx, cell_dims, cellCount, hashFunction);
                    int neighbor_particle_idx = cell_list[neighbor_flat_idx].first;
                    int end_idx = cell_list[neighbor_flat_idx].second + 1;
                    for (; neighbor_particle_idx < end_idx; neighbor_particle_idx++) {
                        int rbIDb = particles[neighbor_particle_idx].rbID;
                        if (rbIDb != -2)
                            continue;
                        //Particle pb = particles[neighbor_particle_idx];
                        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[neighbor_particle_idx]), &pb);
                        // Exclude current particle (r = 0) from force calculation
                        //if (!(x == 0 && y == 0 && z == 0) || neighbor_particle_idx > ti.thread_id) {
                        //if (rbIDa == -2 && rbIDb == -2) {
                            //float d0 = collideSphereSphere(h, 0, pa.predicted, pb.predicted);
                            //if (d0 > 0) {
                                // 6 d
                                float lambda1 = particles[ti.thread_id].lambda;
                                float lambda2 = particles[neighbor_particle_idx].lambda;
                                
                                vec3 d_p = pa.predicted - pb.predicted;
                                vec3 spiky = W_spiky(d_p, h, epsilon_spiky);

                                // 6 e surface
                                float d_poly = W_poly6((d_p).norm(), h) / w_poly_d_q;// W_poly6(delta_q * h, h); // TODO 2. W const ?
                                float poly = d_poly;
                                for (int i = 0; i < artificial_pressure_n - 1; i++) {
                                    poly *= d_poly;
                                }
                                float s_corr = -artificial_pressure_k * poly;

                                // 6 d, e
                                vec3 d_lambda_spiky = (lambda1 + lambda2 + s_corr) * spiky;
                                lambda_spiky += d_lambda_spiky;
                            //}
                        //}
                        // Follow linked list
                        //neighbor_particle_idx = particle_list[neighbor_particle_idx];
                    }
                }
            }
        }

        particles[ti.thread_id].d_predicted += lambda_spiky * rho0inv;
    }
}

__global__ void computeVorticityAndViscosity(float dt, Saiga::ArrayView<Particle> particles, std::pair<int, int>* cell_list, int* particle_list, int *constraints, int *constraintCounter, int maxConstraintNum, ivec3 cell_dims, int cellCount, float cellSize, int hashFunction, float h, float epsilon_spiky, float c_viscosity) {
    Saiga::CUDA::ThreadInfo<> ti;

    if (ti.thread_id < particles.size()) {
        //Particle& pa = particles[ti.thread_id];
        ParticleCalc1 pa;
        ParticleCalc1 pb;
        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc1*>(&particles[ti.thread_id].velocity), &pa);
        int rbIDa = pa.rbID;
        if (rbIDa != -2)
            return;

        ivec3 cell_idx = calculate_cell_idx(pa.position, cellSize); // actually pa.position but we only load predicted and its identical here

        vec3 curl = {0, 0, 0};

        vec3 viscosity = {0, 0, 0};

        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                for (int z = -1; z <= 1; z++) {
                    ivec3 neighbor_cell_idx = cell_idx + ivec3(x, y, z);
                    int neighbor_flat_idx = calculate_hash_idx(neighbor_cell_idx, cell_dims, cellCount, hashFunction);
                    int neighbor_particle_idx = cell_list[neighbor_flat_idx].first;
                    int end_idx = cell_list[neighbor_flat_idx].second + 1;
                    for (; neighbor_particle_idx < end_idx; neighbor_particle_idx++) {
                        //Particle& pb = particles[neighbor_particle_idx];
                        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc1*>(&particles[neighbor_particle_idx].velocity), &pb);
                        // Exclude current particle (r = 0) from force calculation
                        //if (!(x == 0 && y == 0 && z == 0) || neighbor_particle_idx > ti.thread_id) {
                        int rbIDb = pb.rbID;
                        if (rbIDb != -2)
                            continue;
                            //float d0 = collideSphereSphere(h, 0, pa.predicted, pb.predicted);
                            //if (d0 > 0) {
                                // 6 f 1
                                // vorticity
                                //vec3 v_i = (pa.predicted - pa.position) / dt;
                                //vec3 v_j = (pb.predicted - pb.position) / dt;

                                //vec3 v_i = pa.velocity;
                                //vec3 v_j = pb.velocity;

                                vec3 d_velocity = pb.velocity - pa.velocity;

                                vec3 d_p = pa.position - pb.position;
                                
                                vec3 spiky = W_spiky(d_p, h, epsilon_spiky);
                                curl += d_velocity.cross(spiky);

                                // 6 g
                                // viscosity
                                float poly = W_poly6((d_p).norm(), h);
                                viscosity += d_velocity * poly;

                            //}
                        // Follow linked list
                        //neighbor_particle_idx = particle_list[neighbor_particle_idx];
                    }
                }
            }
        }

        particles[ti.thread_id].sdf = curl;

        //particles[ti.thread_id].velocity += c_viscosity * viscosity;
        particles[ti.thread_id].d_momentum = c_viscosity * viscosity; // TODO
    }
}

__global__ void applyVorticityAndViscosity(float dt, Saiga::ArrayView<Particle> particles, std::pair<int, int>* cell_list, int* particle_list, int *constraints, int *constraintCounter, int maxConstraintNum, ivec3 cell_dims, int cellCount, float cellSize, int hashFunction, float h, float epsilon_spiky, float epsilon_vorticity) { // TODO no h needed
    Saiga::CUDA::ThreadInfo<> ti;

    if (ti.thread_id < particles.size()) {
        //Particle& pa = particles[ti.thread_id];
        ParticleCalc2 pa;
        ParticleCalc3 pb;
        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc2*>(&particles[ti.thread_id].position), &pa);
        int rbIDa = pa.rbID;

        if (rbIDa != -2)
            return;

        ivec3 cell_idx = calculate_cell_idx(pa.position, cellSize); // actually pa.position but we only load predicted and its identical here

        vec3 curl_gradient = {0, 0, 0};

        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                for (int z = -1; z <= 1; z++) {
                    ivec3 neighbor_cell_idx = cell_idx + ivec3(x, y, z);
                    int neighbor_flat_idx = calculate_hash_idx(neighbor_cell_idx, cell_dims, cellCount, hashFunction);
                    int neighbor_particle_idx = cell_list[neighbor_flat_idx].first;
                    int end_idx = cell_list[neighbor_flat_idx].second + 1;
                    for (; neighbor_particle_idx < end_idx; neighbor_particle_idx++) {
                        //Particle& pb = particles[neighbor_particle_idx];
                        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc3*>(&particles[neighbor_particle_idx].position), &pb);
                        // Exclude current particle (r = 0) from force calculation
                        //if (!(x == 0 && y == 0 && z == 0) || neighbor_particle_idx > ti.thread_id) {
                        int rbIDb = pb.rbID;
                        if (rbIDb != -2)
                            continue;
                            //float d0 = collideSphereSphere(h, 0, pa.predicted, pb.predicted);
                            //if (d0 > 0) {
                                // 6 f
                                // vorticity
                                curl_gradient += pa.sdf.norm() * W_spiky(pa.position - pb.position, h, epsilon_spiky);
                            //}
                        // Follow linked list
                        //neighbor_particle_idx = particle_list[neighbor_particle_idx];
                    }
                }
            }
        }

        vec3 force = epsilon_vorticity * curl_gradient.normalized().cross(pa.sdf);
        // apply vorticity force
        particles[ti.thread_id].velocity += force * pa.massinv;

        // TODO
        // apply viscosity
        particles[ti.thread_id].velocity += pa.d_momentum;

        // reset curl for sdf
        particles[ti.thread_id].sdf = {0,0,0};
    }
}

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
    } else if (physicsMode == 5) { // 4.0 Rigid Body
        resetConstraintCounter<<<1, 32>>>(d_constraintCounter, d_constraintCounterWalls);
        //resetConstraints<<<Saiga::CUDA::getBlockCount(maxConstraintNum, BLOCK_SIZE), BLOCK_SIZE>>>(d_constraintList, maxConstraintNum, d_constraintCounter, d_constraintCounterWalls);
        //CUDA_SYNC_CHECK_ERROR();

        const unsigned int BLOCKS_CELLS = Saiga::CUDA::getBlockCount(cellCount, BLOCK_SIZE);
        reset_cell_list<<<BLOCKS_CELLS, BLOCK_SIZE>>>(d_cell_list, cellCount);
        //CUDA_SYNC_CHECK_ERROR();
        createLinkedCells<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_cell_list, d_particle_list, cellDim, cellCount, cellSize, hashFunction);
        //CUDA_SYNC_CHECK_ERROR();

    // 4.0 einziger unterschied
        createConstraintParticlesLinkedCellsRigidBodies<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, hashFunction);

        createConstraintWalls<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_walls, d_constraintListWalls, d_constraintCounterWalls, maxConstraintNumWalls);
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
            if (useSDF) {
                updateRigidBodies();
                solverPBDParticlesSDF<<<Saiga::CUDA::getBlockCount(maxConstraintNum, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_constraintList, d_constraintCounter, maxConstraintNum, relaxP, jacobi, d_rigidBodies);
            } else {
                solverPBDParticles<<<Saiga::CUDA::getBlockCount(maxConstraintNum, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_constraintList, d_constraintCounter, maxConstraintNum, relaxP, jacobi);
            }
            solverPBDWalls<<<Saiga::CUDA::getBlockCount(maxConstraintNumWalls, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_walls, d_constraintListWalls, d_constraintCounterWalls, maxConstraintNumWalls, relaxP, jacobi);
            CUDA_SYNC_CHECK_ERROR();

            updateParticlesPBD2Iterator<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, calculatedRelaxP);
            CUDA_SYNC_CHECK_ERROR();
        }

        // 4.0 TODO hier rein!
        constraintsShapeMatchingRB();

        updateParticlesPBD2<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, relaxP);
        //CUDA_SYNC_CHECK_ERROR();
        cudaDeviceSynchronize();
    } else if (physicsMode == 6) { // 5.0 Fluid        
        resetConstraintCounter<<<1, 32>>>(d_constraintCounter, d_constraintCounterWalls);
        //resetConstraints<<<Saiga::CUDA::getBlockCount(maxConstraintNum, BLOCK_SIZE), BLOCK_SIZE>>>(d_constraintList, maxConstraintNum, d_constraintCounter, d_constraintCounterWalls);
        //CUDA_SYNC_CHECK_ERROR();

        const unsigned int BLOCKS_CELLS = Saiga::CUDA::getBlockCount(cellCount, BLOCK_SIZE);
        reset_cell_list_opti<<<BLOCKS_CELLS, BLOCK_SIZE>>>(d_cell_list, cellCount, particleCount);
        CUDA_SYNC_CHECK_ERROR();

        // moved up from previously after createConstraintWalls before iteration loop
        updateParticlesPBD1<<<BLOCKS, BLOCK_SIZE>>>(dt, gravity, d_particles, dampV);

        calculateHash<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_particle_hash, d_cell_list, d_particle_list, cellDim, cellCount, cellSize, hashFunction);
        CUDA_SYNC_CHECK_ERROR();
        thrust::sort_by_key(thrust::device_pointer_cast(d_particle_hash), thrust::device_pointer_cast(d_particle_hash) + particleCount, d_particles.device_begin());

        createLinkedCellsOpti<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_particle_hash, d_cell_list, d_particle_list, cellDim, cellCount, cellSize, hashFunction);
        //createLinkedCells<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_cell_list, d_particle_list, cellDim, cellCount, cellSize, hashFunction);
        CUDA_SYNC_CHECK_ERROR();

    // 4.0 einziger unterschied
        createConstraintParticlesLinkedCellsRigidBodiesFluid<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, hashFunction);

        createConstraintWalls<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_walls, d_constraintListWalls, d_constraintCounterWalls, maxConstraintNumWalls);
        CUDA_SYNC_CHECK_ERROR();

        
        // TODO constraints
        //thrust::device_ptr<int> d = thrust::device_pointer_cast(d_constraintList);  
        
        //thrust::fill(d, d+N, 2);
        //int N = thrust::remove_if(d, d + maxConstraintNum, remove_predicate_constraints()) - d;

        CUDA_SYNC_CHECK_ERROR();
        // solver Iterations: project Constraints

        float w_poly_d_q = W_poly6(delta_q * h, h);

        float calculatedRelaxP = relaxP;
        for (int i = 0; i < solverIterations; i++) {
            // 6
            // b, c
            computeDensityAndLambda<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, hashFunction, h, epsilon_spiky, omega_lambda_relax, particleRadiusRestDensity);
            CUDA_SYNC_CHECK_ERROR();
            // d, e
            updateParticlesPBD2IteratorFluid<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, hashFunction, h, epsilon_spiky, particleRadiusRestDensity, artificial_pressure_k, artificial_pressure_n, w_poly_d_q);
            CUDA_SYNC_CHECK_ERROR();
            
            // old:

            if (useCalculatedRelaxP) {
                calculatedRelaxP = 1 - pow(1 - calculatedRelaxP, 1.0/(i+1));
            }
            // TODO N -> maxConstraintNum
            if (useSDF) {
                updateRigidBodies();
                solverPBDParticlesSDF<<<Saiga::CUDA::getBlockCount(maxConstraintNum, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_constraintList, d_constraintCounter, maxConstraintNum, relaxP, jacobi, d_rigidBodies, mu_k, mu_s, mu_f);
            } else {
                solverPBDParticles<<<Saiga::CUDA::getBlockCount(maxConstraintNum, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_constraintList, d_constraintCounter, maxConstraintNum, relaxP, jacobi);
            }
            solverPBDWalls<<<Saiga::CUDA::getBlockCount(maxConstraintNumWalls, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_walls, d_constraintListWalls, d_constraintCounterWalls, maxConstraintNumWalls, relaxP, jacobi, mu_k, mu_s, mu_f);
            CUDA_SYNC_CHECK_ERROR();

            updateParticlesPBD2Iterator<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, calculatedRelaxP);
            CUDA_SYNC_CHECK_ERROR();
        }

        // 4.0 TODO hier rein!
        constraintsShapeMatchingRB();

        updateParticlesPBD2<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, relaxP);
        CUDA_SYNC_CHECK_ERROR();

        computeVorticityAndViscosity<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, hashFunction, h, epsilon_spiky, c_viscosity);
        CUDA_SYNC_CHECK_ERROR();
        applyVorticityAndViscosity<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, hashFunction, h, epsilon_spiky, epsilon_vorticity);
        CUDA_SYNC_CHECK_ERROR();


        cudaDeviceSynchronize();
    } else if (physicsMode == 7) { // cloth physics        
        resetConstraintCounter<<<1, 32>>>(d_constraintCounter, d_constraintCounterWalls);
        //resetConstraints<<<Saiga::CUDA::getBlockCount(maxConstraintNum, BLOCK_SIZE), BLOCK_SIZE>>>(d_constraintList, maxConstraintNum, d_constraintCounter, d_constraintCounterWalls);
        //CUDA_SYNC_CHECK_ERROR();

        const unsigned int BLOCKS_CELLS = Saiga::CUDA::getBlockCount(cellCount, BLOCK_SIZE);
        reset_cell_list_opti<<<BLOCKS_CELLS, BLOCK_SIZE>>>(d_cell_list, cellCount, particleCount);
        CUDA_SYNC_CHECK_ERROR();

        // moved up from previously after createConstraintWalls before iteration loop
        updateParticlesPBD1_radius<<<BLOCKS, BLOCK_SIZE>>>(dt, gravity, d_particles, dampV, particleRadiusWater, particleRadiusCloth);

        calculateHash<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_particle_hash, d_cell_list, d_particle_list, cellDim, cellCount, cellSize, hashFunction);
        CUDA_SYNC_CHECK_ERROR();
        thrust::sort_by_key(thrust::device_pointer_cast(d_particle_hash), thrust::device_pointer_cast(d_particle_hash) + particleCount, d_particles.device_begin());

        createLinkedCellsOpti<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_particle_hash, d_cell_list, d_particle_list, cellDim, cellCount, cellSize, hashFunction);
        //createLinkedCells<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_cell_list, d_particle_list, cellDim, cellCount, cellSize, hashFunction);
        CUDA_SYNC_CHECK_ERROR();

    // 4.0 einziger unterschied
        createConstraintParticlesLinkedCellsRigidBodiesFluid<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, hashFunction);

        createConstraintWalls<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_walls, d_constraintListWalls, d_constraintCounterWalls, maxConstraintNumWalls);
        CUDA_SYNC_CHECK_ERROR();

        
        // TODO constraints
        //thrust::device_ptr<int> d = thrust::device_pointer_cast(d_constraintList);  
        
        //thrust::fill(d, d+N, 2);
        //int N = thrust::remove_if(d, d + maxConstraintNum, remove_predicate_constraints()) - d;

        CUDA_SYNC_CHECK_ERROR();
        // solver Iterations: project Constraints

        float w_poly_d_q = W_poly6(delta_q * h, h);

        updateLookupTable<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_particleIdLookup);

        float calculatedRelaxP = relaxP;
        for (int i = 0; i < solverIterations; i++) {
            // 6
            // b, c
            computeDensityAndLambda<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, hashFunction, h, epsilon_spiky, omega_lambda_relax, particleRadiusRestDensity);
            CUDA_SYNC_CHECK_ERROR();
            // d, e
            updateParticlesPBD2IteratorFluid<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, hashFunction, h, epsilon_spiky, particleRadiusRestDensity, artificial_pressure_k, artificial_pressure_n, w_poly_d_q);
            CUDA_SYNC_CHECK_ERROR();
            
            // old:

            if (useCalculatedRelaxP) {
                calculatedRelaxP = 1 - pow(1 - calculatedRelaxP, 1.0/(i+1));
            }
            // TODO N -> maxConstraintNum
            if (useSDF) {
                updateRigidBodies();
                solverPBDParticlesSDF<<<Saiga::CUDA::getBlockCount(maxConstraintNum, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_constraintList, d_constraintCounter, maxConstraintNum, relaxP, jacobi, d_rigidBodies, mu_k, mu_s, mu_f);
            } else {
                solverPBDParticles<<<Saiga::CUDA::getBlockCount(maxConstraintNum, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_constraintList, d_constraintCounter, maxConstraintNum, relaxP, jacobi);
            }
            solverPBDWalls<<<Saiga::CUDA::getBlockCount(maxConstraintNumWalls, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_walls, d_constraintListWalls, d_constraintCounterWalls, maxConstraintNumWalls, relaxP, jacobi, mu_k, mu_s, mu_f);
            CUDA_SYNC_CHECK_ERROR();

            solverPBDCloth<<<Saiga::CUDA::getBlockCount(maxConstraintNumCloth, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_constraintListCloth, d_constraintCounterCloth, maxConstraintNumCloth, d_particleIdLookup);
            if (testBool)
                solverPBDClothBending<<<Saiga::CUDA::getBlockCount(maxConstraintNumClothBending, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_constraintListClothBending, d_constraintCounterClothBending, maxConstraintNumClothBending, d_particleIdLookup, testFloat);

            updateParticlesPBD2Iterator<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, calculatedRelaxP);
            CUDA_SYNC_CHECK_ERROR();
        }

        // 4.0 TODO hier rein!
        constraintsShapeMatchingRB();

        updateParticlesPBD2<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, relaxP);
        CUDA_SYNC_CHECK_ERROR();

        computeVorticityAndViscosity<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, hashFunction, h, epsilon_spiky, c_viscosity);
        CUDA_SYNC_CHECK_ERROR();
        applyVorticityAndViscosity<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, hashFunction, h, epsilon_spiky, epsilon_vorticity);
        CUDA_SYNC_CHECK_ERROR();


        cudaDeviceSynchronize();
    }

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
        // TODO refactor momentum replaced by velocity
        particles[idx].velocity = {0,0,0};
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

    // TODO fix for fixed and different kinds of particles
    
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
            // TODO refactor momentum replaced by velocity
            int x = p.position[0] * 10.0 + p.velocity[0] * 100.0 + p.velocity[0] * 100.0;
            int y = p.position[1] * 10.0 + p.velocity[1] * 100.0 + p.velocity[1] * 100.0;
            int z = p.position[2] * 10.0 + p.velocity[2] * 100.0 + p.velocity[2] * 100.0;
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
            // TODO refactor momentum replaced by velocity
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
    return x.second <= 1e-5;
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