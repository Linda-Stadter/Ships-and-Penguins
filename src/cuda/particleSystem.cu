﻿#define _USE_MATH_DEFINES
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
#include <unordered_map>

#ifdef __windows__
    #include <crt\math_functions.h>
#endif

__host__ void checkError(cudaError_t err);
// time
float t = 0;
std::unordered_map<std::string, int> objects{};

void ParticleSystem::setDevicePtr(void* particleVbo) {
    d_particles = ArrayView<Particle>((Particle*) particleVbo, particleCount);
}

__global__ void updateParticlesPBD1_radius(float dt, vec3 gravity, Saiga::ArrayView<Particle>particles, float damp_v, float particleRadiusWater, float particleRadiusCloth, int cannonId, int playerPenguinId) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    Particle &p = particles[ti.thread_id];

    if (p.fixed || p.rbID == cannonId || p.rbID == playerPenguinId)
        return;

    /*
    // quite expensive memory access
    if (p.rbID == -2)
        p.radius = particleRadiusWater;
    else if (p.rbID == -3)
        p.radius = particleRadiusCloth;
    */

    vec3 newVelocity = p.velocity + dt * gravity;
    // dampVelocities
    newVelocity *= damp_v;

    p.predicted = p.position + dt * newVelocity;
}

__global__ void updateParticlesPBD2Iterator(float dt, Saiga::ArrayView<Particle>particles, float relax_p) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    Particle &p = particles[ti.thread_id];

    if (!p.fixed) {
        p.predicted += relax_p * p.d_predicted;
    }
    // reset
    p.d_predicted = {0, 0, 0};
}
__global__ void updateParticlesPBD2(float dt, Saiga::ArrayView<Particle>particles, float relax_p) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    Particle &p = particles[ti.thread_id];

    if (p.rbID == -4)
        return;

    if (!p.fixed) {
        //p.predicted += relax_p * p.d_predicted;
        p.velocity = (p.predicted - p.position) / dt;
        p.position = p.predicted;
    }
    // reset
    p.d_predicted = {0, 0, 0};

    // 6.2
    p.lambda = 0;
}

__global__ void resetOcean(Saiga::ArrayView<Particle> d_particles, int startId, int endId, int xMax, int zMax, vec3 corner, vec4 color, vec3 fluidDim, int particleCountRB) {
    Saiga::CUDA::ThreadInfo<> ti;
    int id = ti.thread_id;

    if (id >= startId && id < endId) {
        if (id > particleCountRB) {
            id -= particleCountRB;
        }
        int y = (id - startId) / (xMax * zMax);
        int z = ((id - startId) - (y * xMax * zMax)) / xMax;
        int x = ((id - startId) - (y * xMax * zMax)) % xMax;

        Particle &p = d_particles[ti.thread_id];
        // for fluids
        int matId = -2;
        float distance = 0.5;
        float particleRenderRadius = 0.3;


        vec3 random_offset = vec3((id % 3) * 0.01, (id % 7) * 0.01, (id % 11) * 0.01);

        vec3 position = vec3(x, y, z) * distance + corner;
        // for trochoidals
        if ((position[0] < -fluidDim[0]/2) || (position[2] < -fluidDim[2]/2) || (position[0] > fluidDim[0]/2) || (position[2] > fluidDim[2]/2)) {
            matId = -4;
        }
        p.position = position + random_offset;
        p.velocity ={0, 0, 0};
        p.massinv = 1.0/1.0;
        p.predicted = p.position;
        // 2.3
        p.color = color;
        p.radius = particleRenderRadius;

        p.fixed = false;

        // 4.0
        p.rbID = matId;
        p.relative ={0, 0, 0};
        p.sdf ={0, 0, 0};


        // 6.0
        p.lambda = 0;

        p.id = ti.thread_id; // cloth

        if (matId == -4) {
            p.relative = p.position;
        }
    }

}

__global__ void resetHits(int* d_particleHits, int particleCount, int* d_rbHits, int rigidBodyCount) {
    Saiga::CUDA::ThreadInfo<> ti;
    int id = ti.thread_id;

    if (id < particleCount) {
        d_particleHits[id] = 0;
    }

    if (id < rigidBodyCount) {
        d_rbHits[id] = 0;
    }

}

__global__ void countHits(int* d_particleHits, int particleCount, int* d_rbHits, int rigidBodyCount, int* d_score, ShipInfo* d_shipInfos, int* d_shipInfosCounter) {
    Saiga::CUDA::ThreadInfo<> ti;
    int id = ti.thread_id;

    if (id < particleCount) {
        if (d_particleHits[id] != 0 && d_particleHits[id] != -1) {
            atomicAdd(d_score, 1);
            d_particleHits[id] = -1;
        }
    }

    if (id < rigidBodyCount) {
        if (d_rbHits[id] != 0 && d_rbHits[id] != -1) {
            for (int shipIdx = 0; shipIdx < *d_shipInfosCounter; shipIdx++) {
                ShipInfo &shipInfo = d_shipInfos[shipIdx];
                if (id == shipInfo.rbID) {
                    atomicAdd(d_score, 5);
                    break;
                }
                if (id == shipInfo.penguinID) {
                    atomicAdd(d_score, 50);
                    break;
                }
            }
            d_rbHits[id] = -1;
        }
    }

}

// a bit redundant
__global__ void resetParticlesStartEnd(Saiga::ArrayView<Particle> d_particles, Saiga::ArrayView<vec3> d_gradient, int startId, int endId, int xMax, int zMax, vec3 corner, float distance, int matId, vec4 color, float particleRenderRadius) {
    Saiga::CUDA::ThreadInfo<> ti;
    int id = ti.thread_id;

    if (id < d_particles.size() && id >= startId && id < endId) {
        Particle &p = d_particles[id];

        int y = (id - startId) / (xMax * zMax);
        int z = ((id - startId) - (y * xMax * zMax)) / xMax;
        int x = ((id - startId) - (y * xMax * zMax)) % xMax;
        float offset = d_particles[id].radius;

        vec3 random_offset = vec3((id % 3) * 0.01, (id % 7) * 0.01, (id % 11) * 0.01);

        p.position = vec3(x, y, z) * distance + corner + random_offset;

        p.velocity = {0, 0, 0};
        p.massinv = 1.0/1.0;
        p.predicted = p.position;
        // 2.3
        p.color = color;
        p.radius = particleRenderRadius;

        p.fixed = false;

        // 4.0
        p.rbID = matId;
        p.relative = {0, 0, 0};
        p.sdf = {0, 0, 0};
        if (matId == -4) {
            p.sdf = d_gradient[id - startId];
        } 

        // 6.0
        p.lambda = 0;

        p.id = ti.thread_id; // cloth

        if (matId == -4) {
            p.relative = p.position;
        }
    }

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

    // pseudo random position offset
    int rand = ti.thread_id + p.position[0];
    p.position = corner + pos * distance + vec3{rand % 11, rand % 17, rand % 13} * randInitMul;

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

    if (rbID == -4) {
        p.relative = p.position;
    }
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

    // pseudo random position offset
    int rand = ti.thread_id + p.position[0];
    p.position = corner + pos * distance + vec3{rand % 11, rand % 17, rand % 13} * randInitMul;

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

// 4.0
__global__ void initCuboidParticles(Saiga::ArrayView<Particle> particles, int id, vec3 pos, ivec3 dim, vec3 rot, vec4 color, int particleCountRB, RigidBody *rigidBodies) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id > 0)
        return;
    
    mat3 rotMat;
    rotMat = Eigen::AngleAxisf(rot.x(), vec3::UnitX())
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

__global__ void initSingleRigidBodyParticle(Saiga::ArrayView<Particle> particles, int id, vec3 pos, vec3 sdf, vec4 color, int particleCountRB, RigidBody *rigidBodies, bool fixed=false, float mass=1.0, float particleRadius=0.5, bool stripes=true) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id > 0)
        return;
    
    particles[particleCountRB].position = pos;
    particles[particleCountRB].predicted = pos;
    particles[particleCountRB].rbID = id;

    // random wood stripes
    if (stripes)
        color = color * (1- (((int)((pos[0]+pos[1])*30) % 7) * 0.07));

    particles[particleCountRB].color = color;

    particles[particleCountRB].fixed = fixed;
    particles[particleCountRB].massinv = 1.0f/mass;

    particles[particleCountRB].radius = particleRadius;

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
int ParticleSystem::loadObj(int rigidBodyCount, int particleCountRB, vec3 pos, vec3 rot, vec4 color, Saiga::UnifiedModel model, float scaling, float particleMass = 1, float maxObjParticleCount = 30, bool stripes = true, bool fixed = false) {
    Saiga::UnifiedMesh mesh = model.CombinedMesh().first;
    std::vector<Triangle> triangles = mesh.TriangleSoup();
    // 1
    Saiga::AABB bb = model.BoundingBox(); // mesh. or model.BoundingBox()
    vec3 min = bb.min;
    vec3 max = bb.max;
    // 2
    // Schnittstellen
    float maxSize = bb.maxSize();
    //float sampleDistance = 0.1;
    float sampleDistance = maxSize / maxObjParticleCount;
    int count = 0;
    Saiga::AccelerationStructure::ObjectMedianBVH omBVH(triangles);

    if (true) {
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
                    vec3 ori = min + sampleDistance * ivec3{x, y, z}.cast<float>();
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
                    vec3 ori = min + sampleDistance * ivec3{x, y, z}.cast<float>();
                    if (voxel[z][y][x].first) {
                        count++;
                        vec3 position = pos + ori*(scaling / sampleDistance);
                        vec3 sdf = (float)voxel[z][y][x].first * normalize(voxel[z][y][x].second);
                        initSingleRigidBodyParticle<<<1, 32>>>(d_particles, rigidBodyCount, position, sdf, color, particleCountRB++, d_rigidBodies, fixed, particleMass, scaling, stripes);
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
                        vec3 position = pos + ori * (scaling / sampleDistance);
                        initSingleRigidBodyParticle<<<1, 32>>>(d_particles, rigidBodyCount, position, vec3{0.f,0.f,0.f}, color, particleCountRB++, d_rigidBodies);
                    }
                }
            }
        }

    }
    return count;
}

// 4.4
int ParticleSystem::loadBox(int rigidBodyCount, int particleCountRB, ivec3 dim, vec3 pos, vec3 rot, vec4 color, bool fixed=false, float mass=1.0, float scaling=1.0, float particleRadius=0.5, bool noSDF = false) {    
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
                    vec3 ori = min + sampleDistance * ivec3{x, y, z}.cast<float>();
                    if (voxel[z][y][x].first) {
                        count++;
                        //float scaling = 0.5f;
                        vec3 position = pos + ori*(scaling / sampleDistance);
                        vec3 sdf = (float)voxel[z][y][x].first * normalize(voxel[z][y][x].second);
                        if (noSDF)
                            sdf = {0, 0, 0};
                        initSingleRigidBodyParticle<<<1, 32>>>(d_particles, rigidBodyCount, position, sdf, color, particleCountRB++, d_rigidBodies, fixed, mass, particleRadius, false);
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
        vec3 d_originOfMass = p.predicted / (float)rigidBodies[p.rbID].particleCount;
        atomicAdd(&rigidBodies[p.rbID].originOfMass[0], d_originOfMass[0]);
        atomicAdd(&rigidBodies[p.rbID].originOfMass[1], d_originOfMass[1]);
        atomicAdd(&rigidBodies[p.rbID].originOfMass[2], d_originOfMass[2]);
    }
}

__global__ void covariance(Saiga::ArrayView<Particle> particles, int particleCountRB, RigidBody *rigidBodies, int cannonId, int playerPenguinId) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particleCountRB)
        return;
    Particle &p = particles[ti.thread_id];
    if (p.rbID >= 0 && p.rbID != cannonId && p.rbID != playerPenguinId) {
        //vec3 pc = p.position - rigidBodies[p.rbID].originOfMass;
        mat3 pcr = (p.predicted - rigidBodies[p.rbID].originOfMass) * p.relative.transpose();

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

__global__ void SVD(RigidBody *rigidBodies, int rigidBodyCount) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= rigidBodyCount)
        return;
    RigidBody &rb = rigidBodies[ti.thread_id];
    rb.A = svd3_cuda::pd(rb.A);
}

__global__ void resolveRigidBodyConstraints(Saiga::ArrayView<Particle> particles, int particleCountRB, RigidBody *rigidBodies, int playerPenguinId) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particleCountRB)
        return;
    Particle &p = particles[ti.thread_id];
    if (p.rbID >= 0) {
        // dx = (Q*r + c) - p
        if (p.rbID == playerPenguinId) {
            p.predicted += (rigidBodies[0].A * p.relative + rigidBodies[p.rbID].originOfMass) - p.predicted;
        }
        else {
            p.predicted += (rigidBodies[p.rbID].A * p.relative + rigidBodies[p.rbID].originOfMass) - p.predicted;
        }
    }
}

__global__ void updateRigidBodySpeed(RigidBody *rigidBodies, int maxRigidBodyCount, float dt = 0) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= maxRigidBodyCount)
        return;
    RigidBody &rb = rigidBodies[ti.thread_id];
    // reset
    rb.speed = (rb.lastOriginOfMass - rb.originOfMass).norm() / dt;
    rb.lastOriginOfMass = rb.originOfMass;
}

__global__ void resetRigidBody(RigidBody *rigidBodies, int maxRigidBodyCount) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= maxRigidBodyCount)
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
    rb.lastOriginOfMass = {0,0,0};
    rb.speed = 0;
    rb.A = mat3::Zero().cast<float>();
}

__global__ void initRigidBodiesRotation(RigidBody *rigidBodies, int rigidBodyCount) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= rigidBodyCount)
        return;
    RigidBody &rb = rigidBodies[ti.thread_id];
    rb.initA = rb.A;
}

void ParticleSystem::constraintsShapeMatchingRB() {
    updateRigidBodies();

    resolveRigidBodyConstraints<<<BLOCKS, BLOCK_SIZE>>>(d_particles, particleCount, d_rigidBodies, objects["player_penguin"]);
    CUDA_SYNC_CHECK_ERROR();    
}

void ParticleSystem::updateRigidBodies() {
    const unsigned int BLOCKS_RB = Saiga::CUDA::getBlockCount(rigidBodyCount, BLOCK_SIZE);

    resetRigidBody<<<BLOCKS_RB, BLOCK_SIZE>>>(d_rigidBodies, maxRigidBodyCount);
    CUDA_SYNC_CHECK_ERROR();

    caclulateRigidBodyOriginOfMass<<<BLOCKS, BLOCK_SIZE>>>(d_particles, particleCount, d_rigidBodies);
    CUDA_SYNC_CHECK_ERROR();
    covariance<<<BLOCKS, BLOCK_SIZE>>>(d_particles, particleCount, d_rigidBodies, objects["cannon"], objects["player_penguin"]);
    CUDA_SYNC_CHECK_ERROR();
    SVD<<<BLOCKS_RB, BLOCK_SIZE>>>(d_rigidBodies, rigidBodyCount);
    CUDA_SYNC_CHECK_ERROR();
    
}

// sehr haesslich
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

std::vector<vec3> computeGradient(vec3 voxelGridEnd, int*** grid) {
    std::vector<vec3> gradients;
    std::vector<int> magnitudes;

    // This order must be consistent with the spawn order
    // (Otherwise output gradient in a 3d array)
    for (int y = 0; y < voxelGridEnd[1]; y+= 1) {
        for (int z = 0; z < voxelGridEnd[2]; z+= 1) {
            for (int x = 0; x < voxelGridEnd[0]; x += 1) {

                if (grid[x][y][z] == 0) {
                    continue;
                }
                vec3 grad = vec3(0, 0, 0);

                int left = 0;
                if (x > 0) {
                    left = grid[x-1][y][z];
                }
                int top = 0;
                if (y > 0) {
                    top = grid[x][y-1][z];
                }
                int back = 0;
                if (z > 0) {
                    back = grid[x][y][z-1];
                }

                int right = 0;
                if (x < voxelGridEnd[0]-1) {
                    right = grid[x+1][y][z];
                }
                int bottom = 0;
                if (y < voxelGridEnd[1]-1) {
                    bottom = grid[x][y+1][z];
                }
                int front = 0;
                if (z < voxelGridEnd[2]-1) {
                    front = grid[x][y][z+1];
                }

                // using central differencing
                grad[0] += (left-right);
                grad[1] += (top-bottom);
                grad[2] += (back-front);

                grad = normalize(grad);
                if (grad == vec3(0, 0, 0)) {
                    grad = vec3(0, 1, 0);
                }
                grad *= grid[x][y][z];

                // gradient output as a normal
                gradients.push_back(grad * (-1));
            }
        }
    }


    return gradients;
}

void computeSDF(vec3 voxelGridEnd, int*** grid) {

    // dynamic programming algorithm

    // go forward
    for (int x = 0; x < voxelGridEnd[0]; x += 1) {
        for (int y = 0; y < voxelGridEnd[1]; y+= 1) {
            for (int z = 0; z < voxelGridEnd[2]; z+= 1) {
                if (grid[x][y][z] == 0) {
                    continue;
                }
                if (x == 0 || y == 0 || z == 0) {
                    continue;
                }

                // look at -1 neighbors (already updated)
                int minNeighbor = std::min({grid[x-1][y][z], grid[x][y-1][z], grid[x][y][z-1]});
                grid[x][y][z] = minNeighbor + 1;
            }
        }
    }

    // go backward and overwrite wrong values in-place
    for (int x = voxelGridEnd[0]-1; x >= 0; x -= 1) {
        for (int y = voxelGridEnd[1]-1; y >= 0; y -= 1) {
            for (int z = voxelGridEnd[2]-1; z >= 0; z -= 1) {
                if (grid[x][y][z] == 0) {
                    continue;
                }
                if (x == voxelGridEnd[0]-1 || y == voxelGridEnd[1]-1 || z == voxelGridEnd[2]-1) {
                    grid[x][y][z] = 1;
                    continue;
                }

                // look at +1 neighbors (already updated)
                int minNeighbor =  std::min({grid[x+1][y][z], grid[x][y+1][z], grid[x][y][z+1]});
                grid[x][y][z] = min(grid[x][y][z], minNeighbor + 1);
            }
        }
    }
}

__global__ void resetEnemyGrid(int *d_enemyGridWeight, int enemyGridDim) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= enemyGridDim * enemyGridDim)
        return;

    d_enemyGridWeight[ti.thread_id] = 0;
}

__device__ int getPrintChar(const char c, int x, int y) {
    const int width = 16;
    const int height = 10;
    char l_char[height][width] = {
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0},
        {0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0},
        {0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0},
        {0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0},
        {0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0},
        {0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0},
        {0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0},
        {0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    };

    char g_char[height][width] = {
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0},
        {0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,0},
        {0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0},
        {0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0},
        {0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0},
        {0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,0},
        {0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,0},
        {0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    };

    char d_char[height][width] = {
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0},
        {0,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0},
        {0,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0},
        {0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0},
        {0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0},
        {0,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0},
        {0,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0},
        {0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    };

    char v_char[height][width] = {
        {0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1},
        {1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,0},
        {0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0},
        {0,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0},
        {0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0},
        {0,0,1,1,1,1,0,0,0,1,1,1,1,0,0,0},
        {0,0,0,1,1,1,1,0,0,1,1,1,1,0,0,0},
        {0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0},
        {0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    };

    switch (c) {
        case 'L':
            return l_char[y][x];
        case 'G':
            return g_char[y][x];
        case 'D':
            return d_char[y][x];
        case 'V':
            return v_char[y][x];
        default:
            return 0;
    }
}

__global__ void printCharOnSail(Saiga::ArrayView<Particle> d_particles, int start_id, int width, int height, const char printChar) {
    Saiga::CUDA::ThreadInfo<> ti;

    if (ti.thread_id > 0)
        return;

    vec4 red = vec4(239.0/256, 52.0/256, 39.0/256, 1);
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            if (getPrintChar(printChar, x, height - y - 1))
                d_particles[start_id + y * width + x].color = red;
        }
    }

}

void ParticleSystem::spawnShip(vec3 spawnPos, vec4 ship_color, Saiga::UnifiedModel shipModel, float scaling, float particleMass = 1, float maxObjParticleCount = 30, char printChar='X') {
    float randInitMul = 0;

    vec3 rot = {0,0,0};
    color = {0.36, 0.23, 0.10, 1};
    int objParticleCount;

    int particleShipStart = particleCountRB;

    objParticleCount = loadObj(rigidBodyCount, particleCountRB, spawnPos, rot, ship_color, shipModel, scaling, particleMass, maxObjParticleCount);
    particleCountRB += objParticleCount;

    vec3 pos;
    ivec3 dim;

    spawnPos += vec3{-0.8, -0.5, -3};

    int upperMastStartId;
    int lowerMastStartId;


    float mastThickness = 0.2;
    float fixtureThickness = 0.1;

    // cloth
    float sailThickness = 0.2;
    float clothDistance = 0.25;
    int dimX = 16;
    int dimZ = 10;

    // vertical mast
    dim = {1,24,1};
    pos = {0.75, 0.5, 1.5};
    objParticleCount = loadBox(rigidBodyCount, particleCountRB, dim, pos + spawnPos, rot, color, false, 0.1, 0.25, mastThickness, true);
    particleCountRB += dim.x() * dim.y() * dim.z();

    // horizontal upper mast
    dim = {16,1,1};
    pos = {-1, 6, 1.5};
    objParticleCount = loadBox(rigidBodyCount, particleCountRB, dim, pos + spawnPos, rot, color, false, 0.1, 0.25, mastThickness, true);
    particleCountRB += dim.x() * dim.y() * dim.z();

    // sail upper fixture
    upperMastStartId = particleCountRB;
    dim = {dimX,1,1};
    pos = {-1, 6, 2};
    objParticleCount = loadBox(rigidBodyCount, particleCountRB, dim, pos + spawnPos, rot, color, false, 0.1, clothDistance, fixtureThickness, true);
    particleCountRB += dim.x() * dim.y() * dim.z();


    // horizontal lower mast
    dim = {16,1,1};
    pos = {-1, 2, 1.5};
    objParticleCount = loadBox(rigidBodyCount, particleCountRB, dim, pos + spawnPos, rot, color, false, 0.1, 0.25, mastThickness, true);
    particleCountRB += dim.x() * dim.y() * dim.z();

    // sail lower fixture
    lowerMastStartId = particleCountRB;
    dim = {dimX,1,1};
    pos = {-1, 2, 2};
    objParticleCount = loadBox(rigidBodyCount, particleCountRB, dim, pos + spawnPos, rot, color, false, 0.1, clothDistance, fixtureThickness, true);
    particleCountRB += dim.x() * dim.y() * dim.z();

    rigidBodyCount++;

    int particlePenguinStart = particleCountRB;


    // penguin
    pos = {0.8, 0.7, 2.5};
    std::unordered_map<std::string, float> penguin_paramters{{"scaling", 0.1}, {"mass", 0.002}, {"particleCount", 12}};
    Saiga::UnifiedModel penguinModel("objs/penguin.obj");
    vec4 penguin_color = {0.11, 0.10, 0.07, 1};


    objParticleCount = loadObj(rigidBodyCount++, particleCountRB, pos + spawnPos, rot, penguin_color, penguinModel, penguin_paramters["scaling"], penguin_paramters["mass"], penguin_paramters["particleCount"], false);
    particleCountRB += objParticleCount;

    int constraintsStart = clothConstraints.size();

    /*clothConstraints.push_back({particleShipStart + 535, particlePenguinStart + 84, 0, 0});
    clothConstraints.push_back({particleShipStart + 534, particlePenguinStart + 83, 0, 0});
    clothConstraints.push_back({particleShipStart + 536, particlePenguinStart + 85, 0, 0});*/

    clothConstraints.push_back({particleShipStart + 640+4+1, particlePenguinStart + 94+1, .5, 1}); //374
    clothConstraints.push_back({particleShipStart + 640+4+2, particlePenguinStart + 94+2, .5, 1});
    clothConstraints.push_back({particleShipStart + 640+4+3, particlePenguinStart + 94+3, .5, 1});

    int particleClothStart = particleCountRB;

    // ship cloth
    int rbID = -3;
    color = {1.0f, 1.0f, 1.0f, 1.f};
    
    vec3 clothCorner = {-1, 2.75, 2.5};
    int clothParticleCount = dimX * dimZ;
    initParticles<<<BLOCKS, BLOCK_SIZE>>>(particleCountRB, clothParticleCount, dimX, 1, clothCorner + spawnPos, clothDistance, d_particles, randInitMul, sailThickness, rbID, color, false, 0.01);

    // print char
    if (print_LGDV_logo && printChar != 'X')
        printCharOnSail<<<1, 32>>>(d_particles, particleClothStart, dimX, dimZ, printChar);

    // fix upper row
    //initParticles<<<BLOCKS, BLOCK_SIZE>>>(particleCountRB, dimX, dimX, 1, clothCorner + spawnPos, clothDistance, d_particles, randInitMul, 0.35, rbID, color, true, 0.1);
    CUDA_SYNC_CHECK_ERROR();

    //std::vector<ClothConstraint> clothConstraints(0);
    //std::vector<ClothBendingConstraint> clothBendingConstraints(0);

    int initActiveState = 150; // 150 frames ~ 5sec * 30 fps

    // lower fixture
    /*for (int j = 0; j < dimZ; j++) {
        for (int i = dimX-1; i < dimX; i++) {
            int idx = particleCountRB + j * dimX + i;
            int idx2 = lowerMastStartId + j;
            clothConstraints.push_back({idx, idx2, 1.1f * clothDistance, initActiveState});
        }
    }

    // upper fixture
    for (int j = 0; j < dimZ; j++) {
        for (int i = 0; i < 1; i++) {
            int idx = particleCountRB + j * dimX + i;
            int idx2 = upperMastStartId + j;
            clothConstraints.push_back({idx, idx2, 1.1f * clothDistance, initActiveState});
        }
    }*/

    // lower fixture
    //for (int j = 0; j < dimZ; j++) {
    int j = 0;
        for (int i = 0; i < dimX; i++) {
            int idx = particleCountRB + j * dimX + i;
            int idx2 = lowerMastStartId + i;
            clothConstraints.push_back({idx, idx2, 1.1f * clothDistance, initActiveState});
        }
    //}

    // upper fixture
    //for (int j = 0; j < dimZ; j++) {
    j = dimZ-1;
        for (int i = 0; i < dimX; i++) {
            int idx = particleCountRB + j * dimX + i;
            int idx2 = upperMastStartId + i;
            clothConstraints.push_back({idx, idx2, 1.1f * clothDistance, initActiveState});
        }
    //}

    for (int j = 0; j < dimZ; j++) {
        for (int i = 0; i < dimX; i++) {
            int idx = particleCountRB + j * dimX + i;
            if (i < dimX - 1) {
                clothConstraints.push_back({idx, idx+1, 1.0f * clothDistance, initActiveState});
            }
            if (j < dimZ - 1) {
                clothConstraints.push_back({idx, idx+dimX, 1.0f * clothDistance, initActiveState});
            }
            if (j < dimZ - 1 && i < dimX - 1) {
                if (i+j % 2)
                    clothConstraints.push_back({idx, idx+dimX+1, 1.4142f*clothDistance, initActiveState});
                else
                    clothConstraints.push_back({idx+dimX, idx+1, 1.4142f*clothDistance, initActiveState});

                //clothBendingConstraints.push_back({idx+dimX+1, idx, idx+dimX, idx+1});
            }
        }
    }

    shipInfos.push_back({0,
        rigidBodyCount - 2,
        particleShipStart,
        particlePenguinStart,
        particleClothStart,
        particleClothStart + clothParticleCount,
        constraintsStart,
        (int)clothConstraints.size(),
        rigidBodyCount - 1}
    );

    particleCountRB += clothParticleCount;
}

__global__ void colorObjects(Saiga::ArrayView<Particle> d_particles, RigidBody *rigidBodies, ShipInfo* d_shipInfos, int* d_shipInfosCounter, int fishStart, int swordfishStart, int cannonStart, int playerPenguinStart) {
    Saiga::CUDA::ThreadInfo<> ti;

    vec4 white = vec4(0.95, 0.93, 0.93, 1);
    vec4 orange = vec4(0.93, 0.67, 0.27, 1);
    vec4 black ={0.11, 0.10, 0.07, 1};
    vec4 wood ={0.36, 0.23, 0.10, 1};
    vec4 brown = vec4(0.54, 0.45, 0.33, 1);
    vec4 light_brown = vec4(0.60, 0.51, 0.39, 1);
    vec4 blue = vec4(0.51, 0.57, 0.58, 1);

    // color penguin
    for (int shipIdx = 0; shipIdx < *d_shipInfosCounter; shipIdx++) {
        ShipInfo &shipInfo = d_shipInfos[shipIdx];
        if (ti.thread_id == shipInfo.rbID) {
            // nose
            d_particles[shipInfo.penguinStart + 123].color = orange;

            // feet
            d_particles[shipInfo.penguinStart + 45].color = orange;
            d_particles[shipInfo.penguinStart + 135].color = orange;

            // belly
            d_particles[shipInfo.penguinStart + 67].color = white;
            d_particles[shipInfo.penguinStart + 112].color = white;
            d_particles[shipInfo.penguinStart + 158].color = white;
            d_particles[shipInfo.penguinStart + 29].color = white;
            d_particles[shipInfo.penguinStart + 109].color = white;
            d_particles[shipInfo.penguinStart + 192].color = white;
            d_particles[shipInfo.penguinStart + 60].color = white;
            d_particles[shipInfo.penguinStart + 105].color = white;
            d_particles[shipInfo.penguinStart + 151].color = white;
            d_particles[shipInfo.penguinStart + 64].color = white;
            d_particles[shipInfo.penguinStart + 188].color = white;
            d_particles[shipInfo.penguinStart + 184].color = white;
            d_particles[shipInfo.penguinStart + 146].color = white;
            d_particles[shipInfo.penguinStart + 99].color = white;
            d_particles[shipInfo.penguinStart + 55].color = white;
            d_particles[shipInfo.penguinStart + 21].color = white;
            d_particles[shipInfo.penguinStart + 17].color = white;
            d_particles[shipInfo.penguinStart + 50].color = white;
            d_particles[shipInfo.penguinStart + 93].color = white;
            d_particles[shipInfo.penguinStart + 140].color = white;
            d_particles[shipInfo.penguinStart + 180].color = white;
            d_particles[shipInfo.penguinStart + 155].color = white;
            d_particles[shipInfo.penguinStart + 25].color = white;
            
            break;
        }
    }

    if (ti.thread_id == 0) {
        // color player penguin
        // nose
        d_particles[playerPenguinStart + 123].color = orange;

        // feet
        d_particles[playerPenguinStart + 45].color = orange;
        d_particles[playerPenguinStart + 135].color = orange;

        // belly
        d_particles[playerPenguinStart + 67].color = white;
        d_particles[playerPenguinStart + 112].color = white;
        d_particles[playerPenguinStart + 158].color = white;
        d_particles[playerPenguinStart + 29].color = white;
        d_particles[playerPenguinStart + 109].color = white;
        d_particles[playerPenguinStart + 192].color = white;
        d_particles[playerPenguinStart + 60].color = white;
        d_particles[playerPenguinStart + 105].color = white;
        d_particles[playerPenguinStart + 151].color = white;
        d_particles[playerPenguinStart + 64].color = white;
        d_particles[playerPenguinStart + 188].color = white;
        d_particles[playerPenguinStart + 184].color = white;
        d_particles[playerPenguinStart + 146].color = white;
        d_particles[playerPenguinStart + 99].color = white;
        d_particles[playerPenguinStart + 55].color = white;
        d_particles[playerPenguinStart + 21].color = white;
        d_particles[playerPenguinStart + 17].color = white;
        d_particles[playerPenguinStart + 50].color = white;
        d_particles[playerPenguinStart + 93].color = white;
        d_particles[playerPenguinStart + 140].color = white;
        d_particles[playerPenguinStart + 180].color = white;
        d_particles[playerPenguinStart + 155].color = white;
        d_particles[playerPenguinStart + 25].color = white;



        // fish head
        d_particles[fishStart + 269].color = brown;
        d_particles[fishStart + 173].color = brown;
        d_particles[fishStart + 174].color = brown;
        d_particles[fishStart + 175].color = brown;
        d_particles[fishStart + 176].color = brown;
        d_particles[fishStart + 273].color = brown;
        d_particles[fishStart + 374].color = brown;
        d_particles[fishStart + 373].color = brown;
        d_particles[fishStart + 466].color = brown;
        d_particles[fishStart + 465].color = brown;
        d_particles[fishStart + 372].color = brown;
        d_particles[fishStart + 272].color = brown;
        d_particles[fishStart + 271].color = brown;
        d_particles[fishStart + 371].color = brown;
        d_particles[fishStart + 464].color = brown;
        d_particles[fishStart + 463].color = brown;
        d_particles[fishStart + 370].color = brown;
        d_particles[fishStart + 270].color = brown;
        d_particles[fishStart + 369].color = brown;
        d_particles[fishStart + 269].color = brown;

        d_particles[fishStart + 257].color = light_brown;
        d_particles[fishStart + 356].color = light_brown;
        d_particles[fishStart + 453].color = light_brown;
        d_particles[fishStart + 454].color = light_brown;
        d_particles[fishStart + 523].color = light_brown;
        d_particles[fishStart + 524].color = light_brown;
        d_particles[fishStart + 525].color = light_brown;
        d_particles[fishStart + 526].color = light_brown;
        d_particles[fishStart + 527].color = light_brown;
        d_particles[fishStart + 528].color = light_brown;
        d_particles[fishStart + 562].color = light_brown;
        d_particles[fishStart + 165].color = light_brown;
        d_particles[fishStart + 94].color = light_brown;
        d_particles[fishStart + 95].color = light_brown;
        d_particles[fishStart + 96].color = light_brown;
        d_particles[fishStart + 97].color = light_brown;
        d_particles[fishStart + 170].color = light_brown;
        d_particles[fishStart + 263].color = light_brown;
        d_particles[fishStart + 264].color = light_brown;
        d_particles[fishStart + 97].color = light_brown;
        d_particles[fishStart + 363].color = light_brown;
        d_particles[fishStart + 459].color = light_brown;
        d_particles[fishStart + 514].color = light_brown;
        d_particles[fishStart + 88].color = light_brown;
        d_particles[fishStart + 154].color = light_brown;

        // fish eyes
        // left
        d_particles[fishStart + 427].color = white;
        d_particles[fishStart + 414].color = black;
        d_particles[fishStart + 401].color = white;

        //right
        d_particles[fishStart + 142].color = white;
        d_particles[fishStart + 130].color = black;
        d_particles[fishStart + 119].color = white;

        // mouth
        d_particles[fishStart + 301].color = brown;
        d_particles[fishStart + 202].color = brown;

        d_particles[swordfishStart + 92].color = black;
        d_particles[swordfishStart + 91].color = white;
        d_particles[swordfishStart + 62].color = white;
        d_particles[swordfishStart + 84].color = white;
        d_particles[swordfishStart + 122].color = white;
        d_particles[swordfishStart + 171].color = white;
        // lid
        d_particles[swordfishStart + 138].color = blue;
        d_particles[swordfishStart + 100].color = blue;

        d_particles[swordfishStart + 107].color = blue;
        d_particles[swordfishStart + 145].color = blue;
        d_particles[swordfishStart + 178].color = white;
        d_particles[swordfishStart + 129].color = white;
        d_particles[swordfishStart + 68].color = white;
        d_particles[swordfishStart + 99].color = black;
        d_particles[swordfishStart + 137].color = white;
        d_particles[swordfishStart + 130].color = white;
        //d_particles[swordfishStart + 131].color = black;

        d_particles[swordfishStart + 151].color = blue;
        d_particles[swordfishStart + 150].color = blue;
        d_particles[swordfishStart + 191].color = blue;
        d_particles[swordfishStart + 192].color = blue;
        d_particles[swordfishStart + 235].color = blue;
        d_particles[swordfishStart + 234].color = blue;
        d_particles[swordfishStart + 233].color = blue;
        d_particles[swordfishStart + 278].color = blue;
        d_particles[swordfishStart + 279].color = blue;
        d_particles[swordfishStart + 280].color = blue;
        d_particles[swordfishStart + 281].color = blue;
        d_particles[swordfishStart + 328].color = blue;
        d_particles[swordfishStart + 327].color = blue;
        d_particles[swordfishStart + 326].color = blue;
        d_particles[swordfishStart + 325].color = blue;
        d_particles[swordfishStart + 372].color = blue;
        d_particles[swordfishStart + 374].color = blue;
        d_particles[swordfishStart + 373].color = blue;
        d_particles[swordfishStart + 375].color = blue;
        d_particles[swordfishStart + 419].color = blue;
        d_particles[swordfishStart + 418].color = blue;
        d_particles[swordfishStart + 417].color = blue;
        d_particles[swordfishStart + 416].color = blue;
        d_particles[swordfishStart + 459].color = blue;
        d_particles[swordfishStart + 460].color = blue;
        d_particles[swordfishStart + 494].color = blue;
        d_particles[swordfishStart + 493].color = blue;

        d_particles[swordfishStart + 186].color = blue;
        //d_particles[swordfishStart + 73].color = blue;
        //d_particles[swordfishStart + 74].color = blue;
        d_particles[swordfishStart + 110].color = blue;
        d_particles[swordfishStart + 109].color = blue;
        d_particles[swordfishStart + 149].color = blue;
        d_particles[swordfishStart + 189].color = blue;
        d_particles[swordfishStart + 231].color = blue;
        d_particles[swordfishStart + 457].color = blue;
        d_particles[swordfishStart + 454].color = blue;
        d_particles[swordfishStart + 524].color = blue;
        d_particles[swordfishStart + 525].color = blue;

        // left cannon
        d_particles[cannonStart + 1051].color = wood;
        d_particles[cannonStart + 1049].color = wood;
        d_particles[cannonStart + 1046].color = wood;
        d_particles[cannonStart + 1043].color = wood;
        d_particles[cannonStart + 1106].color = wood;
        d_particles[cannonStart + 1049].color = wood;
        d_particles[cannonStart + 1124].color = wood;
        d_particles[cannonStart + 1122].color = wood;
        d_particles[cannonStart + 1124].color = wood;
        d_particles[cannonStart + 1118].color = wood;
        d_particles[cannonStart + 1079].color = wood;
        d_particles[cannonStart + 1089].color = wood;
        d_particles[cannonStart + 1090].color = wood;
        d_particles[cannonStart + 1080].color = wood;
        d_particles[cannonStart + 1103].color = wood;
        d_particles[cannonStart + 1097].color = wood;
        d_particles[cannonStart + 1052].color = wood;
        d_particles[cannonStart + 1050].color = wood;
        d_particles[cannonStart + 1049].color = wood;
        d_particles[cannonStart + 1049].color = wood;
        d_particles[cannonStart + 1047].color = wood;
        d_particles[cannonStart + 1088].color = wood;
        d_particles[cannonStart + 1045].color = wood;
        d_particles[cannonStart + 1086].color = wood;
        d_particles[cannonStart + 1043].color = wood;
        d_particles[cannonStart + 1085].color = wood;
        d_particles[cannonStart + 1086].color = wood;
        d_particles[cannonStart + 1087].color = wood;
        d_particles[cannonStart + 1088].color = wood;
        d_particles[cannonStart + 1089].color = wood;
        d_particles[cannonStart + 1113].color = wood;
        d_particles[cannonStart + 1109].color = wood;
        d_particles[cannonStart + 1111].color = wood;
        d_particles[cannonStart + 1109].color = wood;
        d_particles[cannonStart + 1110].color = wood;
        d_particles[cannonStart + 1112].color = wood;
        d_particles[cannonStart + 1113].color = wood;
        d_particles[cannonStart + 1103].color = wood;
        d_particles[cannonStart + 1104].color = wood;
        d_particles[cannonStart + 1121].color = wood;
        d_particles[cannonStart + 1105].color = wood;
        d_particles[cannonStart + 1112].color = wood;
        d_particles[cannonStart + 1107].color = wood;
        d_particles[cannonStart + 1108].color = wood;
        d_particles[cannonStart + 1127].color = wood;
        d_particles[cannonStart + 1126].color = wood;
        d_particles[cannonStart + 1126].color = wood;
        d_particles[cannonStart + 1125].color = wood;
        d_particles[cannonStart + 1129].color = wood;
        d_particles[cannonStart + 1122].color = wood;
        d_particles[cannonStart + 1121].color = wood;
        d_particles[cannonStart + 1114].color = wood;
        d_particles[cannonStart + 1115].color = wood;
        d_particles[cannonStart + 1116].color = wood;
        d_particles[cannonStart + 1119].color = wood;
        d_particles[cannonStart + 1120].color = wood;
        d_particles[cannonStart + 1096].color = wood;
        d_particles[cannonStart + 1102].color = wood;
        d_particles[cannonStart + 1108].color = wood;
        d_particles[cannonStart + 1089].color = wood;
        d_particles[cannonStart + 1041].color = wood;
        d_particles[cannonStart + 1114].color = wood;
        d_particles[cannonStart + 1121].color = wood;
        d_particles[cannonStart + 1109].color = wood;
        d_particles[cannonStart + 1109].color = wood;
        d_particles[cannonStart + 1085].color = wood;
        d_particles[cannonStart + 1123].color = wood;
        d_particles[cannonStart + 1049].color = wood;
        d_particles[cannonStart + 1044].color = wood;
        d_particles[cannonStart + 1074].color = wood;
        d_particles[cannonStart + 1091].color = wood;
        d_particles[cannonStart + 1037].color = wood;
        d_particles[cannonStart + 1095].color = wood;
        d_particles[cannonStart + 1094].color = wood;
        d_particles[cannonStart + 1101].color = wood;
        d_particles[cannonStart + 1095].color = wood;
        d_particles[cannonStart + 1094].color = wood;
        d_particles[cannonStart + 1100].color = wood;
        d_particles[cannonStart + 1101].color = wood;
        d_particles[cannonStart + 1078].color = wood;
        d_particles[cannonStart + 1078].color = wood;
        d_particles[cannonStart + 1076].color = wood;
        d_particles[cannonStart + 1075].color = wood;
        d_particles[cannonStart + 1081].color = wood;
        d_particles[cannonStart + 1082].color = wood;
        d_particles[cannonStart + 1077].color = wood;
        d_particles[cannonStart + 1078].color = wood;
        d_particles[cannonStart + 1084].color = wood;
        d_particles[cannonStart + 1091].color = wood;
        d_particles[cannonStart + 1092].color = wood;


        // right cannon
        d_particles[cannonStart + 43].color = wood;
        d_particles[cannonStart + 55].color = wood;
        d_particles[cannonStart + 37].color = wood;
        d_particles[cannonStart + 23].color = wood;
        d_particles[cannonStart + 72].color = wood;
        d_particles[cannonStart + 73].color = wood;
        d_particles[cannonStart + 74].color = wood;
        d_particles[cannonStart + 75].color = wood;
        d_particles[cannonStart + 75].color = wood;
        d_particles[cannonStart + 74].color = wood;
        d_particles[cannonStart + 73].color = wood;
        d_particles[cannonStart + 72].color = wood;
        d_particles[cannonStart + 66].color = wood;
        d_particles[cannonStart + 48].color = wood;
        d_particles[cannonStart + 49].color = wood;
        d_particles[cannonStart + 50].color = wood;
        d_particles[cannonStart + 51].color = wood;
        d_particles[cannonStart + 52].color = wood;
        d_particles[cannonStart + 47].color = wood;
        d_particles[cannonStart + 51].color = wood;
        d_particles[cannonStart + 50].color = wood;
        d_particles[cannonStart + 49].color = wood;
        d_particles[cannonStart + 48].color = wood;
        d_particles[cannonStart + 29].color = wood;
        d_particles[cannonStart + 30].color = wood;
        d_particles[cannonStart + 31].color = wood;
        d_particles[cannonStart + 32].color = wood;
        d_particles[cannonStart + 33].color = wood;
        d_particles[cannonStart + 34].color = wood;
        d_particles[cannonStart + 15].color = wood;
        d_particles[cannonStart + 13].color = wood;
        d_particles[cannonStart + 13].color = wood;
        d_particles[cannonStart + 6].color = wood;
        d_particles[cannonStart + 7].color = wood;
        d_particles[cannonStart + 8].color = wood;
        d_particles[cannonStart + 11].color = wood;
        d_particles[cannonStart + 10].color = wood;
        d_particles[cannonStart + 3].color = wood;
        d_particles[cannonStart + 2].color = wood;
        d_particles[cannonStart + 9].color = wood;
        d_particles[cannonStart + 6].color = wood;
        d_particles[cannonStart + 55].color = wood;
        d_particles[cannonStart + 22].color = wood;
        d_particles[cannonStart + 21].color = wood;
        d_particles[cannonStart + 26].color = wood;
        d_particles[cannonStart + 27].color = wood;
        d_particles[cannonStart + 41].color = wood;
        d_particles[cannonStart + 40].color = wood;
        d_particles[cannonStart + 58].color = wood;
        d_particles[cannonStart + 58].color = wood;
        d_particles[cannonStart + 58].color = wood;
        d_particles[cannonStart + 72].color = wood;
        d_particles[cannonStart + 67].color = wood;
        d_particles[cannonStart + 68].color = wood;
        d_particles[cannonStart + 69].color = wood;
        d_particles[cannonStart + 75].color = wood;
        d_particles[cannonStart + 70].color = wood;
        d_particles[cannonStart + 51].color = wood;
        d_particles[cannonStart + 50].color = wood;
        d_particles[cannonStart + 49].color = wood;
        d_particles[cannonStart + 48].color = wood;
        d_particles[cannonStart + 14].color = wood;
        d_particles[cannonStart + 12].color = wood;
        d_particles[cannonStart + 4].color = wood;
        d_particles[cannonStart + 16].color = wood;
        d_particles[cannonStart + 22].color = wood;
        d_particles[cannonStart + 28].color = wood;
        d_particles[cannonStart + 42].color = wood;
        d_particles[cannonStart + 59].color = wood;
        d_particles[cannonStart + 70].color = wood;
        d_particles[cannonStart + 16].color = wood;
        d_particles[cannonStart + 46].color = wood;
        d_particles[cannonStart + 45].color = wood;
        d_particles[cannonStart + 44].color = wood;
        d_particles[cannonStart + 43].color = wood;
        d_particles[cannonStart + 64].color = wood;
        d_particles[cannonStart + 17].color = wood;
        d_particles[cannonStart + 18].color = wood;
        d_particles[cannonStart + 19].color = wood;
        d_particles[cannonStart + 20].color = wood;
        d_particles[cannonStart + 48].color = wood;

        
        d_particles[cannonStart + 1131].color = black;
        d_particles[cannonStart + 1129].color = black;
        d_particles[cannonStart + 1131].color = black;
        d_particles[cannonStart + 1130].color = black;
        d_particles[cannonStart + 1128].color = black;
        d_particles[cannonStart + 1036].color = black;
        d_particles[cannonStart + 1035].color = black;
        d_particles[cannonStart + 921].color = black;
        d_particles[cannonStart + 922].color = black;
        d_particles[cannonStart + 786].color = black;
        d_particles[cannonStart + 785].color = black;
        d_particles[cannonStart + 634].color = black;
        d_particles[cannonStart + 635].color = black;
        d_particles[cannonStart + 635].color = black;
        d_particles[cannonStart + 476].color = black;
        d_particles[cannonStart + 328].color = black;
        d_particles[cannonStart + 329].color = black;
        d_particles[cannonStart + 634].color = black;
        d_particles[cannonStart + 92].color = black;
        d_particles[cannonStart + 93].color = black;
        d_particles[cannonStart + 54].color = black;
        d_particles[cannonStart + 55].color = wood;
        d_particles[cannonStart + 53].color = black;
        d_particles[cannonStart + 1].color = black;
        d_particles[cannonStart + 0].color = black;
        d_particles[cannonStart + 5].color = black;
        d_particles[cannonStart + 1093].color = black;
        d_particles[cannonStart + 477].color = black;
        d_particles[cannonStart + 200].color = black;
        d_particles[cannonStart + 199].color = black;
        d_particles[cannonStart + 199].color = black;
        d_particles[cannonStart + 35].color = black;
        d_particles[cannonStart + 1072].color = black;
        d_particles[cannonStart + 1073].color = black;

        // first ring
        d_particles[cannonStart + 78].color = wood;
        d_particles[cannonStart + 77].color = wood;
        d_particles[cannonStart + 81].color = wood;
        d_particles[cannonStart + 80].color = wood;
        d_particles[cannonStart + 85].color = wood;
        d_particles[cannonStart + 84].color = wood;
        d_particles[cannonStart + 88].color = wood;
        d_particles[cannonStart + 194].color = wood;
        d_particles[cannonStart + 194].color = wood;
        d_particles[cannonStart + 318].color = wood;
        d_particles[cannonStart + 317].color = wood;
        d_particles[cannonStart + 469].color = wood;
        d_particles[cannonStart + 468].color = wood;
        d_particles[cannonStart + 625].color = wood;
        d_particles[cannonStart + 626].color = wood;
        d_particles[cannonStart + 778].color = wood;
        d_particles[cannonStart + 777].color = wood;
        d_particles[cannonStart + 915].color = wood;
        d_particles[cannonStart + 908].color = wood;
        d_particles[cannonStart + 1028].color = wood;
        d_particles[cannonStart + 1029].color = wood;
        d_particles[cannonStart + 1068].color = wood;
        d_particles[cannonStart + 1065].color = wood;
        d_particles[cannonStart + 1062].color = wood;
        d_particles[cannonStart + 1057].color = wood;
        d_particles[cannonStart + 1058].color = wood;
        d_particles[cannonStart + 1055].color = wood;
        d_particles[cannonStart + 1054].color = wood;
        d_particles[cannonStart + 947].color = wood;
        d_particles[cannonStart + 934].color = wood;
        d_particles[cannonStart + 809].color = wood;
        d_particles[cannonStart + 186].color = wood;
        d_particles[cannonStart + 119].color = wood;
        d_particles[cannonStart + 219].color = wood;
        d_particles[cannonStart + 342].color = wood;
        d_particles[cannonStart + 492].color = wood;
        d_particles[cannonStart + 649].color = wood;

        // second ring
        d_particles[cannonStart + 1042].color = wood;
        d_particles[cannonStart + 1053].color = wood;
        d_particles[cannonStart + 1053].color = wood;
        d_particles[cannonStart + 1056].color = wood;
        d_particles[cannonStart + 1061].color = wood;
        d_particles[cannonStart + 1060].color = wood;
        d_particles[cannonStart + 1004].color = wood;
        d_particles[cannonStart + 1005].color = wood;
        d_particles[cannonStart + 898].color = wood;
        d_particles[cannonStart + 758].color = wood;
        d_particles[cannonStart + 604].color = wood;
        d_particles[cannonStart + 449].color = wood;
        d_particles[cannonStart + 307].color = wood;
        d_particles[cannonStart + 172].color = wood;
        d_particles[cannonStart + 173].color = wood;
        d_particles[cannonStart + 82].color = wood;
        d_particles[cannonStart + 82].color = wood;
        d_particles[cannonStart + 79].color = wood;
        d_particles[cannonStart + 76].color = wood;
        d_particles[cannonStart + 71].color = wood;
        d_particles[cannonStart + 65].color = wood;
        d_particles[cannonStart + 83].color = wood;
        d_particles[cannonStart + 1048].color = wood;
        d_particles[cannonStart + 925].color = wood;
        d_particles[cannonStart + 788].color = wood;
        d_particles[cannonStart + 639].color = wood;
        d_particles[cannonStart + 333].color = wood;
        d_particles[cannonStart + 201].color = wood;
        d_particles[cannonStart + 94].color = wood;
        d_particles[cannonStart + 482].color = wood;

    }
}

void ParticleSystem::reset(int x, int z, vec3 corner, float distance, float randInitMul, int scenario, vec3 fluidDim, vec3 trochoidal1Dim, vec3 trochoidal2Dim, ivec2 layers) {
    int rbID = -1; // free particles
    vec4 color = {0.0f, 1.0f, 0.0f, 1.f};

    std::unordered_map<std::string, float> ship_paramters{{"scaling", 0.17}, {"mass", 0.02}, {"particleCount", 21}};
    vec4 ship_color = {0.36, 0.23, 0.10, 1};
    Saiga::UnifiedModel shipModel("objs/ship.obj");

    std::unordered_map<std::string, float> penguin_paramters{{"scaling", 0.1}, {"mass", 0.002}, {"particleCount", 12}};
    Saiga::UnifiedModel penguinModel("objs/penguin.obj");
    vec4 penguin_color ={0.11, 0.10, 0.07, 1};

    int cannonStart;
    int playerPenguinStart;

    if (scenario >= 7) {
        color = {0, 0, 0.8, 1};
        rbID = -2; // fluid
    }
    if (scenario == 12) { // trochoidal test scenario
        color ={0.1f, 0.2f, 0.8f, 1.f};
        rbID = -4; // trochoidal particles
    }

    // reset cloth constraints and ship info
    clothConstraints.clear();
    clothBendingConstraints.clear();
    shipInfos.clear();


    if (scenario == 13) {
        // scene parameters
        wave_number = 5;
        steepness = 0.2;
        wind_speed = 0;

        ivec3 trochDim = ivec3(20, 8, 20);
        int startId = 0;
        int endId = trochDim[0] * trochDim[1] * trochDim[2];
        vec3 voxelGridEnd = vec3(trochDim[0], trochDim[1], trochDim[2]);
        int ***grid3D = new int**[endId];
        for (int i = 0; i < trochDim[0]; i++) {
            grid3D[i] = new int*[trochDim[1]];
            for (int j = 0; j < trochDim[1]; j++) {
                grid3D[i][j] = new int[trochDim[2]];
                for (int h = 0; h < trochDim[2]; h++) {
                    grid3D[i][j][h] = 1;
                }
            }
        }

        // compute sdf for trochoidal ocean "block"
        computeSDF(voxelGridEnd, grid3D);
        std::vector<vec3> gradients = computeGradient(voxelGridEnd, grid3D);

        vec3* gradPtr;
        cudaMalloc((void **)&gradPtr, sizeof(vec3) * trochDim[0] * trochDim[1] * trochDim[2]);
        cudaMemcpy(gradPtr, gradients.data(), sizeof(vec3) * trochDim[0] * trochDim[1] * trochDim[2], cudaMemcpyHostToDevice);
        ArrayView<vec3> d_gradient = make_ArrayView(gradPtr, trochDim[0] * trochDim[1] * trochDim[2]);

        // adds trochoidal particles
        resetParticlesStartEnd<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_gradient, startId, endId, x, z, vec3(-10, 0, -4), 0.4, -4, color, 0.3);
        CUDA_SYNC_CHECK_ERROR();

        startId = endId;
        endId = 20 * 20 * 40 * 2;

        // adds fluid particles
        resetParticlesStartEnd<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_gradient, startId, endId, x, z, corner + vec3(x, 0, 0), distance, -2, color, 0.5);
        CUDA_SYNC_CHECK_ERROR();
    }
    else if (scenario == 14) {
        // scene parameters
        wave_number = 3.5;
        steepness = 0.47;
        wind_direction = {-1.0, 0.0, -1.0};
        wind_speed = 0.90;
        solver_iterations = 1;
        c_viscosity = 0.02;
        epsilon_vorticity = 0.001;
        float distance = 0.5;

        // adds trochoidal particles
        // generate first layers of fluids and trochoidals
        resetOcean<<<BLOCKS, BLOCK_SIZE>>>(d_particles, 0, layers[0], x, z, corner, color, fluidDim, 22480);
        CUDA_SYNC_CHECK_ERROR();

        // generate top layer of fluids and trochoidals
        corner -= vec3(trochoidal2Dim[0], 0, trochoidal2Dim[2]);
        float height = trochoidal1Dim[1] - trochoidal2Dim[1];
        corner += vec3(0, height, 0);
        x += 1/distance * trochoidal2Dim[0] * 2;
        z += 1/distance * trochoidal2Dim[2] * 2;

        mapDim = vec3(x * distance, 80, z * distance);
        this->fluidDim = fluidDim;
        enemyGridCell = fluidDim[0] / enemyGridDim;

        resetOcean<<<BLOCKS, BLOCK_SIZE>>>(d_particles, layers[0], layers[1], x, z, corner, color, fluidDim, 0);
        CUDA_SYNC_CHECK_ERROR();

        resetEnemyGrid<<<BLOCKS, BLOCK_SIZE>>>(d_enemyGridWeight, enemyGridDim);
    }
    else {
        resetParticles<<<BLOCKS, BLOCK_SIZE>>>(x, z, corner, distance, d_particles, randInitMul, particleRenderRadius, rbID, color);
        CUDA_SYNC_CHECK_ERROR();
    }


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

        int dimX = 50;
        int dimZ = 50;

        for (int j = 0; j < dimZ; j++) {
            for (int i = 0; i < dimX; i++) {
                int idx = j * dimX + i;
                if (i < dimX - 1) {
                    clothConstraints.push_back({idx, idx+1, 1.0f * distance, 1});
                }
                if (j < dimZ - 1) {
                    clothConstraints.push_back({idx, idx+dimX, 1.0f * distance, 1});
                }
                if (j < dimZ - 1 && i < dimX - 1) {
                    if (i+j % 2)
                        clothConstraints.push_back({idx, idx+dimX+1, 1.4142f*distance, 1});
                    else
                        clothConstraints.push_back({idx+dimX, idx+1, 1.4142f*distance, 1});

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

        particleCountRB = dimX * dimZ;
        int objParticleCount = loadBox(rigidBodyCount++, particleCountRB, dim, pos, rot, color, false, 5);
        particleCountRB += dim.x() * dim.y() * dim.z();
    }

    if (scenario > 2 && scenario < 8)
        initRigidBodies(distance, scenario);

    if (scenario == 11 || scenario == 14) {
        vec3 rot = {0,0,0};
        ivec3 dim = {5,5,5};

        vec3 pos = {0, 3, 0};

        objects["player"] = rigidBodyCount;
        int objParticleCount = loadObj(rigidBodyCount++, particleCountRB, pos, rot, ship_color, shipModel, ship_paramters["scaling"], ship_paramters["mass"], ship_paramters["particleCount"]);
        particleCountRB += objParticleCount; 
    }

    if (scenario == 14) {
        // spawn player ship
        vec3 spawnPos = {5, 2, 5};
        //spawnShip(spawnPos, 0);

        vec3 rot ={0, 0, 0};
        ivec3 dim ={3, 3, 3};

        color ={.5, .5, .5, 0.5};
        vec3 pos ={0, 1, 0};

        // spawns cannonball
        particleFishStart = particleCountRB;
        fishID = rigidBodyCount;
        Saiga::UnifiedModel fishModel("objs/fish.obj");
        objects["ball_1"] = rigidBodyCount;
        vec4 fish_color ={0.77, 0.65, 0.46, 1};
        int objParticleCount = loadObj(rigidBodyCount++, particleCountRB, pos, rot, fish_color, fishModel, 0.05, 0.03, 15, false);
        particleCountRB += objParticleCount;

        pos ={2, 1, 0};
        particleSwordfishStart = particleCountRB;
        Saiga::UnifiedModel swordfishModel("objs/swordfish.obj");
        objects["ball_2"] = rigidBodyCount;
        fish_color ={0.64, 0.68, 0.69, 1};
        objParticleCount = loadObj(rigidBodyCount++, particleCountRB, pos, rot, fish_color, swordfishModel, 0.05, 0.01, 37, false);
        particleCountRB += objParticleCount;

        // spawns enemies
        pos ={-10, 3, 45};
        objects["enemy_1"] = rigidBodyCount;
        spawnShip(pos, ship_color, shipModel, ship_paramters["scaling"], ship_paramters["mass"], ship_paramters["particleCount"], 'L');

        pos ={-30, 3, 25};
        objects["enemy_2"] = rigidBodyCount;
        spawnShip(pos, ship_color, shipModel, ship_paramters["scaling"], ship_paramters["mass"], ship_paramters["particleCount"], 'G');

        pos ={10, 3, 10};
        objects["enemy_3"] = rigidBodyCount;
        spawnShip(pos, ship_color, shipModel, ship_paramters["scaling"], ship_paramters["mass"], ship_paramters["particleCount"], 'D');

        pos ={25, 3, -10};
        objects["enemy_4"] = rigidBodyCount;
        spawnShip(pos, ship_color, shipModel, ship_paramters["scaling"], ship_paramters["mass"], ship_paramters["particleCount"], 'V');

        pos ={5, 3, -20};
        objects["enemy_5"] = rigidBodyCount;
        spawnShip(pos, ship_color, shipModel, ship_paramters["scaling"], ship_paramters["mass"], ship_paramters["particleCount"]);

        pos ={-15, 3, -35};
        objects["enemy_6"] = rigidBodyCount;
        spawnShip(pos, ship_color, shipModel, ship_paramters["scaling"], ship_paramters["mass"], ship_paramters["particleCount"]);

        cannonStart = particleCountRB;
        Saiga::UnifiedModel cannonModel("objs/cannon.obj");
        pos ={0, 30.3, 0.};
        vec4 cannon_color ={0.68, 0.45, 0.24, 1.0};
        objects["cannon"] = rigidBodyCount;
        objParticleCount = loadObj(rigidBodyCount++, particleCountRB, pos, rot, cannon_color, cannonModel, 0.06, 0.001, 23, false);
        particleCountRB += objParticleCount;

        playerPenguinStart = particleCountRB;
        objects["player_penguin"] = rigidBodyCount;
        objParticleCount = loadObj(rigidBodyCount++, particleCountRB, pos + spawnPos, rot, penguin_color, penguinModel, penguin_paramters["scaling"], penguin_paramters["mass"], penguin_paramters["particleCount"], false);
        particleCountRB += objParticleCount;

        // ice
        Saiga::UnifiedModel iceModel("objs/ice.obj");
        Saiga::UnifiedModel iceModel2("objs/ice2.obj");
        vec4 ice_color = {.95, .95, .95, 1};
        float ice_mass = 0.15;

        objects["ice_start"] = rigidBodyCount;
        vec3 icePos = {20, 3, 20};
        objParticleCount = loadObj(rigidBodyCount++, particleCountRB, icePos, rot, ice_color, iceModel, 0.2, ice_mass, 20, false);
        particleCountRB += objParticleCount;

        // spawn ice
        icePos = {-20, 3, 10};
        objParticleCount = loadObj(rigidBodyCount++, particleCountRB, icePos, rot, ice_color, iceModel2, 0.2, ice_mass, 20, false);
        particleCountRB += objParticleCount;

        icePos = {-20, 3, -30};
        objParticleCount = loadObj(rigidBodyCount++, particleCountRB, icePos, rot, ice_color, iceModel2, 0.3, ice_mass, 20, false);
        particleCountRB += objParticleCount;

        objects["ice_end"] = rigidBodyCount;
        icePos = {20, 3, -30};
        objParticleCount = loadObj(rigidBodyCount++, particleCountRB, icePos, rot, ice_color, iceModel, 0.3, ice_mass, 20, false);
        particleCountRB += objParticleCount;

        // fixed ice
        float ice_hight = 1.5;
        icePos = {-60, ice_hight, 10};
        objParticleCount = loadObj(rigidBodyCount++, particleCountRB, icePos, rot, ice_color, iceModel, 0.2, ice_mass, 15, false, true);
        particleCountRB += objParticleCount;

        icePos = {70, ice_hight, 20};
        objParticleCount = loadObj(rigidBodyCount++, particleCountRB, icePos, rot, ice_color, iceModel, 0.3, ice_mass, 20, false, true);
        particleCountRB += objParticleCount;

        icePos = {65, ice_hight, -60};
        objParticleCount = loadObj(rigidBodyCount++, particleCountRB, icePos, rot, ice_color, iceModel, 0.2, ice_mass, 20, false, true);
        particleCountRB += objParticleCount;

        icePos = {-75, ice_hight, -20};
        objParticleCount = loadObj(rigidBodyCount++, particleCountRB, icePos, rot, ice_color, iceModel, 0.3, ice_mass, 15, false, true);
        particleCountRB += objParticleCount;

        icePos = {-65, ice_hight, 60};
        objParticleCount = loadObj(rigidBodyCount++, particleCountRB, icePos, rot, ice_color, iceModel, 0.2, ice_mass, 20, false, true);
        particleCountRB += objParticleCount;

        icePos = {70, ice_hight, -70};
        objParticleCount = loadObj(rigidBodyCount++, particleCountRB, icePos, rot, ice_color, iceModel, 0.3, ice_mass, 20, false, true);
        particleCountRB += objParticleCount;
    }

    if (scenario == 14) {
        // spawn enemy ship
        //spawnShip({-10, 2, -10}, 1);

        // copy constraints

        size_t clothConstraintSize = sizeof(clothConstraints[0]) * clothConstraints.size();
        size_t clothBendingConstraintSize = sizeof(clothBendingConstraints[0]) * clothBendingConstraints.size();

        int distanceConstraintCount = clothConstraints.size();
        int bendingConstraintCount = clothBendingConstraints.size();

        cudaMemcpy(d_constraintListCloth, clothConstraints.data(), clothConstraintSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_constraintListClothBending, clothBendingConstraints.data(), clothBendingConstraintSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_constraintCounterCloth, &distanceConstraintCount, sizeof(int) * 1, cudaMemcpyHostToDevice);
        cudaMemcpy(d_constraintCounterClothBending, &bendingConstraintCount, sizeof(int) * 1, cudaMemcpyHostToDevice);

        // copy ship info

        size_t shipInfosSize = sizeof(shipInfos[0]) * shipInfos.size();
        int shipInfosCount = shipInfos.size();
        cudaMemcpy(d_shipInfos, shipInfos.data(), shipInfosSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_shipInfosCounter, &shipInfosCount, sizeof(int) * 1, cudaMemcpyHostToDevice);
    }

    if (scenario > 2 && scenario != 6 && scenario != 7 && scenario < 7)
        deactivateNonRB<<<BLOCKS, BLOCK_SIZE>>>(d_particles);
    CUDA_SYNC_CHECK_ERROR();
    
    resetRigidBody<<<BLOCKS, BLOCK_SIZE>>>(d_rigidBodies, maxRigidBodyCount);
    CUDA_SYNC_CHECK_ERROR();

    caclulateRigidBodyOriginOfMass<<<BLOCKS, BLOCK_SIZE>>>(d_particles, particleCountRB, d_rigidBodies);
    CUDA_SYNC_CHECK_ERROR();
    initRigidBodyParticles<<<BLOCKS, BLOCK_SIZE>>>(d_particles, particleCountRB, d_rigidBodies);
    CUDA_SYNC_CHECK_ERROR();

    updateRigidBodies();
    initRigidBodiesRotation<<<BLOCKS_RB, BLOCK_SIZE>>>(d_rigidBodies, rigidBodyCount);

    resetRigidBody<<<BLOCKS, BLOCK_SIZE>>>(d_rigidBodies, maxRigidBodyCount);
    CUDA_SYNC_CHECK_ERROR();

    checkError(cudaMalloc((void **)&d_rbHits, sizeof(int) * rigidBodyCount));
    resetHits<<<BLOCKS, BLOCK_SIZE>>>(d_particleHits, particleCount, d_rbHits, rigidBodyCount);
    CUDA_SYNC_CHECK_ERROR();
    int reset = 0;
    checkError(cudaMemcpy(d_score, &reset, sizeof(int), cudaMemcpyHostToDevice));

    printf("rb particles: %i", particleCountRB);
    // reset game values    
    passed_time = 0;
    bonus_flag = false;
    ammo_left = 3;
    game_over = false;

    if (score >= high_score)
        high_score = score;
    score = 0;

    colorObjects<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_rigidBodies, d_shipInfos, d_shipInfosCounter, particleFishStart, particleSwordfishStart, cannonStart, playerPenguinStart);
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
        Saiga::UnifiedModel teapot("objs/teapot.obj");
        pos = linearRand(vec3(-40, 20, -40), vec3(40, 30, 40));
        rot = {0,0,0};
        objParticleCount = loadObj(rigidBodyCount++, particleCountRB, pos, rot, color, teapot, 1);
        particleCountRB += objParticleCount;
        printf("%i\n", objParticleCount);

        pos = {0, 70, 0};
        rot = {0,0,0};
        objParticleCount = loadObj(rigidBodyCount++, particleCountRB, pos, rot, color, teapot, 1);
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

__global__ void createConstraintWalls(Saiga::ArrayView<Particle> particles, Saiga::ArrayView<Saiga::Plane> walls, int *constraints, int *constraintCounter, int maxConstraintNum, int exception_start, int exception_end) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    Particle p = particles[ti.thread_id];

    for (int i = 0; i < walls.size(); i++) {
        if (p.rbID >= exception_start && p.rbID <= exception_end && i > 0 || p.rbID == -3) {
            // only consider ground plane for rigid body execptions
            return;
        }
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

__global__ void solverPBDParticlesSDF(Saiga::ArrayView<Particle> particles, int *constraints, int *constraintCounter, int maxConstraintNum, float relax_p, RigidBody *rigidBodies, float mu_k=0, float mu_s=0, float mu_f=0) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= *constraintCounter || ti.thread_id >= maxConstraintNum)
        return;
    int idxA = constraints[ti.thread_id*2 + 0];
    int idxB = constraints[ti.thread_id*2 + 1];
    Particle &pa = particles[idxA];
    Particle &pb = particles[idxB];

    if (pa.rbID == -2 && pb.rbID == -2) // deactivate for fluid
        return;

    if (pa.rbID == -4 && pb.rbID == -4) // deactivate for trochoidal particles
        return;

    ParticleCalc pa_copy;
    Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[idxA]), &pa_copy);
    ParticleCalc pb_copy;
    Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[idxB]), &pb_copy);

    // TODO mass von material abhaengig machen nicht aus particle lesen
    float m1 = pa.massinv;
    float m2 = pb.massinv;

    float d = collideSphereSphere(pa_copy.radius, pb_copy.radius, pa_copy.predicted, pb_copy.predicted);
    vec3 n = (pa_copy.predicted - pb_copy.predicted).normalized();
    
    vec3 sdf1 = pa.sdf;
    vec3 sdf2 = pb.sdf;
    if ((pa.rbID >= 0 && sdf1.norm() > 0) || (pb.rbID >= 0 && sdf2.norm() > 0)) {
        mat3 R;
        if (pa.rbID >= 0 && sdf1.norm() > 0 && pb.rbID >= 0 && sdf2.norm() > 0) {
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
        } else if (pa.rbID >= 0 && sdf1.norm() > 0) {
            d = sdf1.norm();
            n = normalize(sdf1);
            R = rigidBodies[pa.rbID].A;
        } else if (pb.rbID >= 0 && sdf2.norm() > 0) {
            d = sdf2.norm();
            n = -normalize(sdf2);
            R = rigidBodies[pb.rbID].A;
        }
        n = R * -n;
        if (d <= 1.0) {
            // border particle
            d = collideSphereSphere(pa_copy.radius, pb_copy.radius, pa_copy.predicted, pb_copy.predicted);
            vec3 xij = -(pa_copy.predicted - pb_copy.predicted).normalized();
            if (xij.dot(n) < 0.f) {
                n = xij - 2.0f*(xij.dot(n))*n;
            } else {
                n = xij;
            }
        } else {
            d *= pa_copy.radius + pb_copy.radius;
        }
        n = -n;
    }

    float m = (m1 / (m1 + m2));
    vec3 dx1 = m * d * n;
    vec3 dx2 = - (1.0f - m) * d * n;

    // Friction
    if (mu_f) {
        vec3 a = ((pa.position - pa.predicted) - (pb.position - pb.predicted)); //vec3 a = ((pa.predicted - pa.position) - (pb.predicted - pb.position));
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
    }
    // END Friction

    if (pa.fixed)
        dx2 *= 2.0;
    if (pb.fixed)
        dx1 *= 2.0;

    // jacobi integration
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

__global__ void solverPBDWalls(Saiga::ArrayView<Particle> particles, Saiga::ArrayView<Saiga::Plane> walls, int *constraints, int *constraintCounter, int maxConstraintNum, float relax_p, float mu_k=0, float mu_s=0, float mu_f=0) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= *constraintCounter || ti.thread_id >= maxConstraintNum)
        return;
    int idxP = constraints[ti.thread_id*2 + 0];
    int idxW = constraints[ti.thread_id*2 + 1];
    Particle &p = particles[idxP];
    Saiga::Plane &w = walls[idxW];

    if (p.fixed)
        return;

    // TODO mass von material abhaengig machen nicht aus particle lesen
    float m1 = p.massinv;
    float m2 = 0;
    float d = -collideSpherePlane(p.radius, p.predicted, w);
    //float d = -wall.sphereOverlap(particle.predicted, particle.radius);
    vec3 n = w.normal;
    float m = (m1 / (m1 + m2));
    vec3 dx1 = - m * d * n;

    // Friction
    if (mu_f) {
        vec3 a = (p.position - p.predicted); //vec3 a = ((pa.predicted - pa.position) - (pb.predicted - pb.position));
        vec3 dx_orthogonal = a - (a.dot(n))*n; // a_orthogonal_n

        if (!dx_orthogonal.norm() < mu_s * d) {
            float min = mu_k * d / dx_orthogonal.norm();
            min = min <= 1.0 ? min : 1.0;
            dx_orthogonal *= min;
        }

        vec3 dx1_f = m * dx_orthogonal;
        vec3 dx2_f = - (1.0f - m) * dx_orthogonal;
    
        dx1 += dx1_f * mu_f;
    }
    // END Friction

    atomicAdd(&p.d_predicted[0], dx1[0]);
    atomicAdd(&p.d_predicted[1], dx1[1]);
    atomicAdd(&p.d_predicted[2], dx1[2]);
}

__global__ void updateLookupTable(Saiga::ArrayView<Particle> particles, int *particleIdLookup) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    particleIdLookup[particles[ti.thread_id].id] = ti.thread_id;
}

__global__ void solverPBDCloth(Saiga::ArrayView<Particle> particles, ClothConstraint *constraints, int *constraintCounter, int maxConstraintNum, int *particleIdLookup, float breakDistance) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= *constraintCounter || ti.thread_id >= maxConstraintNum)
        return;
    int idxA_ = constraints[ti.thread_id].first;
    int idxB_ = constraints[ti.thread_id].second;
    int active = constraints[ti.thread_id].active;

    // ignore broken constraints
    if (active == -1) {
        return;
    }

    int idxA = particleIdLookup[idxA_];
    int idxB = particleIdLookup[idxB_];
    Particle &pa = particles[idxA];
    Particle &pb = particles[idxB];

    ParticleCalc pa_copy;
    Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[idxA]), &pa_copy);
    ParticleCalc pb_copy;
    Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[idxB]), &pb_copy);

    // TODO mass von material abhaengig machen nicht aus particle lesen
    float m1 = pa.massinv;
    float m2 = pb.massinv;

    float constraintDistance = constraints[ti.thread_id].dist;

    float d = collideSphereSphere(constraintDistance, 0, pa_copy.predicted, pb_copy.predicted);
    vec3 n = (pa_copy.predicted - pb_copy.predicted).normalized();
    float m = (m1 / (m1 + m2));
    vec3 dx1 = m * d * n;
    vec3 dx2 = - (1.0f - m) * d * n;

    if (pa.fixed)
        dx2 *= 2.0;
    if (pb.fixed)
        dx1 *= 2.0;

    // update constraint
    if (active > 1)
        constraints[ti.thread_id].active--;
    // break constraint
    if (active == 1 && breakDistance > 0 && abs(d) > constraintDistance * breakDistance) {
        constraints[ti.thread_id].active = -1;
    }

    // jacobi integration
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

__device__ void changePredicted(Particle &p, vec3 dx) {
    if (!p.fixed) {
        atomicAdd(&p.d_predicted[0], dx[0]);
        atomicAdd(&p.d_predicted[1], dx[1]);
        atomicAdd(&p.d_predicted[2], dx[2]);
    }
}

__global__ void solverPBDClothBending(Saiga::ArrayView<Particle> particles, ClothBendingConstraint *constraints, int *constraintCounter, int maxConstraintNum, int *particleIdLookup, float test_float) {
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

    float dp = - (omega1 * sqrt_d2 * (acosf(d) - angle0)) / (sum_omega_q);
    dp *= test_float;

    float dp1 = -(omega1 * sqrt_d2 * (acosf(d) - angle0)) / (sum_omega_q);
    float dp2 = -(omega1 * sqrt_d2 * (acosf(d) - angle0)) / (sum_omega_q);
    float dp3 = -(omega1 * sqrt_d2 * (acosf(d) - angle0)) / (sum_omega_q);
    float dp4 = -(omega1 * sqrt_d2 * (acosf(d) - angle0)) / (sum_omega_q);

    changePredicted(particles[idx[0]], dp * q1);
    changePredicted(particles[idx[1]], dp * q2);
    changePredicted(particles[idx[2]], dp * q3);
    changePredicted(particles[idx[3]], dp * q4);
}

__global__ void resetCellListOptimized(std::pair<int, int>* cell_list, int cellCount, int particleCount) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id < cellCount) {
        cell_list[ti.thread_id].first = particleCount;
        cell_list[ti.thread_id].second = 0;
    }
}

__device__ ivec3 calculateCellIdx(vec3 position, float cellSize) {
    return (position / cellSize).cast<int>(); // incorrect but faster
    /*vec3 idxf(position / cellSize);
    idxf = {floor(idxf[0]), floor(idxf[1]), floor(idxf[2])};
    return idxf.cast<int>();*/
}

__device__ int calculateHashIdx(ivec3 cell_idx, ivec3 cell_dims, int cellCount) {
    int i2 = ((cell_idx.x() % cell_dims.x()) + cell_dims.x()) % cell_dims.x();
    int j2 = ((cell_idx.y() % cell_dims.y()) + cell_dims.y()) % cell_dims.y();
    int k2 = ((cell_idx.z() % cell_dims.z()) + cell_dims.z()) % cell_dims.z();
    int flat_cell_idx = i2 * cell_dims.y() * cell_dims.z() + j2 * cell_dims.z() + k2;
    return flat_cell_idx;
}

__global__ void calculateHash(Saiga::ArrayView<Particle> particles, int* particle_hash, std::pair<int, int>* cell_list, int* particle_list, ivec3 cell_dims, int cellCount, float cellSize) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id < particles.size()) {
        ivec3 cell_idx = calculateCellIdx(particles[ti.thread_id].predicted, cellSize);
        int flat_cell_idx = calculateHashIdx(cell_idx, cell_dims, cellCount);
        particle_hash[ti.thread_id] = flat_cell_idx;
    }
}

__global__ void createLinkedCellsOptimized(Saiga::ArrayView<Particle> particles, int* particle_hash, std::pair<int, int>* cell_list, int* particle_list, ivec3 cell_dims, int cellCount, float cellSize) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id < particles.size()) {
        int flat_cell_idx = particle_hash[ti.thread_id];
        atomicMin(&cell_list[flat_cell_idx].first, ti.thread_id);
        atomicAdd(&cell_list[flat_cell_idx].second, 1);
    }
}

__device__ void registerHit(Saiga::ArrayView<Particle> particles, int score_particle, int score_rbID, int* d_particleHits, int* d_rbHits) {

    if (score_rbID != -1 && score_rbID != -2 && score_rbID != -4) {
        if (score_rbID == -3 && d_particleHits[score_particle] == 0 && particles[score_particle].lambda == 0) {
            d_particleHits[score_particle] = 1;
            particles[score_particle].lambda = 1;
        }

        if (score_rbID > 0 && d_rbHits[score_rbID] == 0) {
            d_rbHits[score_rbID] = 1;
        }
    }
}

__global__ void createConstraintParticlesLinkedCellsRigidBodiesFluid(Saiga::ArrayView<Particle> particles, ClothConstraint *d_constraintListCloth, std::pair<int, int>* cell_list, int* particle_list, int *constraints, int *constraintCounter, int maxConstraintNum, ivec3 cell_dims, int cellCount, float cellSize, int exception, int ball1Id, int ball2Id, int* d_particleHits, int* d_rbHits, bool regular_ball) {
    Saiga::CUDA::ThreadInfo<> ti;
    int ballId = (regular_ball) ? ball1Id : ball2Id;

    if (ti.thread_id < particles.size()) {
        ParticleCalc pa;
        ParticleCalc pb;
        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[ti.thread_id]), &pa);
        int rbIDa = particles[ti.thread_id].rbID;

        if (rbIDa == -4 || rbIDa == exception)
            return;

        ivec3 cell_idx = calculateCellIdx(pa.predicted, cellSize); // actually pa.position but we only load predicted and its identical here

        static const int X_CONSTS[14] ={-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0};
        static const int Y_CONSTS[14] ={-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0};
        static const int Z_CONSTS[14] ={-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0};

        for (int i = 0; i < 14; i++) {
            int x = X_CONSTS[i];
            int y = Y_CONSTS[i];
            int z = Z_CONSTS[i];

            ivec3 neighbor_cell_idx = cell_idx + ivec3(x, y, z);
            int neighbor_flat_idx = calculateHashIdx(neighbor_cell_idx, cell_dims, cellCount);
            int neighbor_particle_idx = cell_list[neighbor_flat_idx].first;
            int end_idx = cell_list[neighbor_flat_idx].second + neighbor_particle_idx;
            for (; neighbor_particle_idx < end_idx; neighbor_particle_idx++) {

                int rbIDb = particles[neighbor_particle_idx].rbID;
                if (rbIDb == -4 || rbIDb == exception)
                    continue;
                if ((rbIDa == -1 || rbIDb == -1 || rbIDa != rbIDb) &&
                        (i != 13 || neighbor_particle_idx > ti.thread_id)) {
                    Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[neighbor_particle_idx]), &pb);
                    float d0 = collideSphereSphere(pa.radius, pb.radius, pa.predicted, pb.predicted);
                    if (d0 > 0) {

                        // hit detection
                        if (rbIDa == ballId) {
                            registerHit(particles, neighbor_particle_idx, rbIDb, d_particleHits, d_rbHits);
                        }
                        else if (rbIDb == ballId) {
                            registerHit(particles, ti.thread_id, rbIDa, d_particleHits, d_rbHits);
                        }

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

// 6.2
__device__ __host__ float W_poly6(float r, float h) {
    if (r > h)
        return 0;
    float h2 = h * h;
    float hd = h2 - r * r;
    float hd3 = hd * hd * hd;
    float h3 = h2 * h;
    float h9 = h3 * h3 * h3;
    return 315.f / (64.f * M_PI * h9) * hd3;
}

__device__ __host__ vec3 W_spiky(vec3 r, float h, float epsilon) {
    float d = r.norm();
    if (d <= epsilon || d > h)
        return {0, 0, 0};
    float hd = h - d;
    vec3 hd2 = r/d * hd * hd; //vec3 hd2 = r.normalized() * hd * hd;
    float h3 = h * h * h;
    float h6 = h3 * h3;
    return -45.f / (M_PI * h6) * hd2;
}

inline __device__ __host__ float range(float value, float min, float max) {
    return value = value < min ? min : (value > max ? max : value);
}

__device__ float calculateSpray(float C_density, float rho0inv) {
    //float min_density = (1.0f * m) * rho0inv - 1.0;
    float min_density = 5 * rho0inv - 1.0; // 1 * W_poly(0, h) + 3 * W_poly(0.5, h)
    float max_density = 8 * rho0inv - 1.0; // 1 * W_poly(0, h) + 9 * W_poly(0.5, h) // 1.57 + x * 0.66; x= 3: 3.5, 6: 5.5, 9: 7.5
    float non_spray = (C_density - min_density) / (max_density - min_density);
    non_spray = range(non_spray, 0, 1);
    float spray = 1.0f - (non_spray * non_spray);
    return spray;
}

__global__ void computeDensityAndLambda(Saiga::ArrayView<Particle> particles, std::pair<int, int>* cell_list, int* particle_list, int *constraints, int *constraintCounter, int maxConstraintNum, ivec3 cell_dims, int cellCount, float cellSize, float h, float epsilon_spiky, float omega_lambda_relax, float particleRadius, vec3 fluidDim) {
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

        ivec3 cell_idx = calculateCellIdx(pa.predicted, cellSize); // actually pa.position but we only load predicted and its identical here
        float rho = 0;
        vec3 spiky_sum = {0, 0, 0};
        float lambda2 = 0;

        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                for (int z = -1; z <= 1; z++) {
                    ivec3 neighbor_cell_idx = cell_idx + ivec3(x, y, z);
                    int neighbor_flat_idx = calculateHashIdx(neighbor_cell_idx, cell_dims, cellCount);
                    int neighbor_particle_idx = cell_list[neighbor_flat_idx].first;
                    int end_idx = cell_list[neighbor_flat_idx].second + neighbor_particle_idx;
                    for (; neighbor_particle_idx < end_idx; neighbor_particle_idx++) {
                        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[neighbor_particle_idx]), &pb);
                        int rbIDb = particles[neighbor_particle_idx].rbID;
                        if (!(rbIDb == -2 || rbIDb == -4))
                            continue;
                        
                        vec3 d_p = pa.predicted - pb.predicted;

                        float d_rho = m * W_poly6((d_p).norm(), h);
                        rho += d_rho;

                        vec3 spiky = W_spiky(d_p, h, epsilon_spiky) * rho0inv;
                        float spiky_norm = spiky.norm();
                        spiky_sum += spiky;
                        lambda2 += spiky_norm * spiky_norm;
                    }
                }
            }
        }
        // compute density and lambda
        float C_density = rho * rho0inv - 1.0;
        float lambda1 = spiky_sum.norm();
        lambda1 *= lambda1;
        float lambda = -C_density / (lambda1 + lambda2 + omega_lambda_relax);
        particles[ti.thread_id].lambda = lambda;

        // no spray at fluid borders
        vec3 fluidDim_neg = -fluidDim/2 + vec3(2, 2, 2);
        vec3 fluidDim_pos = fluidDim/2 - vec3(2, 2, 2);
        if (pa.predicted[0] <= fluidDim_neg[0] || pa.predicted[2] <= fluidDim_neg[2] || pa.predicted[0] >= fluidDim_pos[0] || pa.predicted[2] >= fluidDim_pos[2]) {
            particles[ti.thread_id].color = vec4(0.1, 0.1, 0.8, 1);
            return;
        }

        // gischt (spray)
        float spray = calculateSpray(C_density, rho0inv);
        vec4 water_color = {0, 0, 0.8, 1};
        vec4 spray_color = {0.7, 0.7, 0.8, 1};
        float old_spray = particles[ti.thread_id].color[0];
        float new_spray = spray;
        float spray_cooldown = 0.9;
        if (new_spray < old_spray)
            new_spray = old_spray * spray_cooldown;
        particles[ti.thread_id].color = (1.0f - new_spray) * water_color + new_spray * spray_color;
    }
}

__global__ void updateParticlesPBD2IteratorFluid(Saiga::ArrayView<Particle> particles, std::pair<int, int>* cell_list, int* particle_list, int *constraints, int *constraintCounter, int maxConstraintNum, ivec3 cell_dims, int cellCount, float cellSize, float h, float epsilon_spiky, float particleRadius, float artificial_pressure_k, int artificial_pressure_n, float w_poly_d_q) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id < particles.size()) {
        int rbIDa = particles[ti.thread_id].rbID;
        if (rbIDa != -2)
            return;
        ParticleCalc pa;
        ParticleCalc pb;
        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[ti.thread_id]), &pa);

        const float rho0inv = (8.0 * particleRadius * particleRadius * particleRadius);
        float lambda1 = particles[ti.thread_id].lambda;
        vec3 lambda_spiky = {0, 0, 0};
        //float w_poly_d_q = W_poly6(delta_q * h, h);
        ivec3 cell_idx = calculateCellIdx(pa.predicted, cellSize); // actually pa.position but we only load predicted and its identical here

        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                for (int z = -1; z <= 1; z++) {
                    ivec3 neighbor_cell_idx = cell_idx + ivec3(x, y, z);
                    int neighbor_flat_idx = calculateHashIdx(neighbor_cell_idx, cell_dims, cellCount);
                    int neighbor_particle_idx = cell_list[neighbor_flat_idx].first;
                    int end_idx = cell_list[neighbor_flat_idx].second + neighbor_particle_idx;
                    for (; neighbor_particle_idx < end_idx; neighbor_particle_idx++) {
                        int rbIDb = particles[neighbor_particle_idx].rbID;
                        if (!(rbIDb == -2 || rbIDb == -4))
                            continue;
                        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[neighbor_particle_idx]), &pb);

                        // 6 d
                        float lambda2 = particles[neighbor_particle_idx].lambda;
                                
                        vec3 d_p = pa.predicted - pb.predicted;
                        vec3 spiky = W_spiky(d_p, h, epsilon_spiky);

                        // 6 e surface
                        float d_poly = W_poly6((d_p).norm(), h) / w_poly_d_q; // W_poly6(delta_q * h, h);
                        float poly = d_poly;
                        for (int i = 0; i < artificial_pressure_n - 1; i++) {
                            poly *= d_poly;
                        }
                        float s_corr = -artificial_pressure_k * poly;

                        // 6 d, e
                        vec3 d_lambda_spiky = (lambda1 + lambda2 + s_corr) * spiky;
                        lambda_spiky += d_lambda_spiky;
                    }
                }
            }
        }
        particles[ti.thread_id].d_predicted += lambda_spiky * rho0inv;
    }
}

__global__ void computeVorticityAndViscosity(float dt, Saiga::ArrayView<Particle> particles, std::pair<int, int>* cell_list, int* particle_list, int *constraints, int *constraintCounter, int maxConstraintNum, ivec3 cell_dims, int cellCount, float cellSize, float h, float epsilon_spiky, float c_viscosity) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id < particles.size()) {
        ParticleCalc1 pa;
        ParticleCalc1 pb;
        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc1*>(&particles[ti.thread_id].velocity), &pa);
        int rbIDa = pa.rbID;
        if (rbIDa != -2)
            return;

        ivec3 cell_idx = calculateCellIdx(pa.position, cellSize); // actually pa.position but we only load predicted and its identical here
        vec3 curl = {0, 0, 0};
        vec3 viscosity = {0, 0, 0};

        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                for (int z = -1; z <= 1; z++) {
                    ivec3 neighbor_cell_idx = cell_idx + ivec3(x, y, z);
                    int neighbor_flat_idx = calculateHashIdx(neighbor_cell_idx, cell_dims, cellCount);
                    int neighbor_particle_idx = cell_list[neighbor_flat_idx].first;
                    int end_idx = cell_list[neighbor_flat_idx].second + neighbor_particle_idx;
                    for (; neighbor_particle_idx < end_idx; neighbor_particle_idx++) {
                        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc1*>(&particles[neighbor_particle_idx].velocity), &pb);
                        int rbIDb = pb.rbID;
                        if (!(rbIDb == -2 || rbIDb == -4))
                            continue;

                        // vorticity
                        vec3 d_velocity = pb.velocity - pa.velocity;
                        vec3 d_p = pa.position - pb.position;
                        vec3 spiky = W_spiky(d_p, h, epsilon_spiky);
                        curl += d_velocity.cross(spiky);
                        // viscosity
                        float poly = W_poly6((d_p).norm(), h);
                        viscosity += d_velocity * poly;
                    }
                }
            }
        }
        // compute verticity
        particles[ti.thread_id].sdf = curl;
        // compute velocity change by viscosity
        particles[ti.thread_id].d_momentum = c_viscosity * viscosity;
    }
}

__device__ vec3 calculateWind(vec3 pa, vec3 pb, vec3 wind_direction, float wind_speed) {
    float h = 1;
    vec3 UP = {0, 1, 0};
    vec3 d_p = pa - pb;
    if (d_p.norm() > h) // || d_p.x() * d_p.x() < 1e-5 || d_p.y() * d_p.y() < 1e-5)
        return {0,0,0};
    float wind_force = d_p.dot(wind_direction) * d_p.dot(UP); //d_p.x() * d_p.y();
    float wpoly = W_poly6((d_p).norm(), h) * wind_force;
    return UP * wind_force * wind_speed/10.0f;
}

__global__ void applyVorticityAndViscosity(float dt, Saiga::ArrayView<Particle> particles, std::pair<int, int>* cell_list, int* particle_list, int *constraints, int *constraintCounter, int maxConstraintNum, ivec3 cell_dims, int cellCount, float cellSize, float h, float epsilon_spiky, float epsilon_vorticity, vec3 wind_direction, float wind_speed) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id < particles.size()) {
        ParticleCalc2 pa;
        ParticleCalc3 pb;
        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc2*>(&particles[ti.thread_id].position), &pa);
        int rbIDa = pa.rbID;
        if (rbIDa != -2)
            return;

        ivec3 cell_idx = calculateCellIdx(pa.position, cellSize); // actually pa.position but we only load predicted and its identical here
        vec3 curl_gradient = {0, 0, 0};
        vec3 d_velocity = {0, 0, 0};

        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                for (int z = -1; z <= 1; z++) {
                    ivec3 neighbor_cell_idx = cell_idx + ivec3(x, y, z);
                    int neighbor_flat_idx = calculateHashIdx(neighbor_cell_idx, cell_dims, cellCount);
                    int neighbor_particle_idx = cell_list[neighbor_flat_idx].first;
                    int end_idx = cell_list[neighbor_flat_idx].second + neighbor_particle_idx;
                    for (; neighbor_particle_idx < end_idx; neighbor_particle_idx++) {
                        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc3*>(&particles[neighbor_particle_idx].position), &pb);
                        int rbIDb = pb.rbID;
                        if (!(rbIDb == -2 || rbIDb == -4))
                            continue;
                        if (neighbor_particle_idx == ti.thread_id)
                            continue;

                        // vorticity
                        curl_gradient += pa.sdf.norm() * W_spiky(pa.position - pb.position, h, epsilon_spiky);
                        // wind and waves
                        d_velocity += calculateWind(pa.position, pb.position, wind_direction, wind_speed);
                    }
                }
            }
        }
        vec3 force = epsilon_vorticity * curl_gradient.normalized().cross(pa.sdf);
        // apply vorticity force
        d_velocity += force * pa.massinv; // TODO mass von material abhaengig machen nicht aus particle lesen
        // apply viscosity
        d_velocity += pa.d_momentum;
        // update velocity
        particles[ti.thread_id].velocity += d_velocity;
        // reset curl for sdf
        particles[ti.thread_id].sdf = {0,0,0};
    }
}

__device__ vec3 trochoidalWaveOffset(vec3 gridPoint, vec2 direction, float wave_length, float steepness, float t, float dif) {
    direction = normalize(direction);
    float x = gridPoint[0];
    float y = gridPoint[1];
    float z = gridPoint[2];

    float k = 2 * M_PI / wave_length;
    // compute speed of waves
    float c = 9.8 / (k * 2.5);

    // amplitude
    float a = ((steepness / 10 * y) * 2) / k;
    float a_cos = ((steepness / 10 * y) * dif) / k;
    float f = k * (direction[0] * x + direction[1] * z - c * t);

    //float sin_f = sinf(f);
    //float cos_f = cosf(f);
    float sin_f;
    float cos_f;
    sincosf(f, &sin_f, &cos_f);

    float xOffset = direction[0] * a * sin_f;
    float yOffset = -a_cos * cos_f;
    float zOffset = direction[1] * a * sin_f;

    return vec3(xOffset, yOffset, zOffset);
}

__device__ vec4 colorTrochoidalParticles(vec3 position, vec3 old_position, vec4 color) {
    vec4 d_color = vec4(0.01, 0.01, 0.00, 0.01);
    float rand = abs(((int)position[0] + (int)position[2])) % 17 * 0.07;

    if (old_position[1] - position[1] > -0.15) {
        color -= d_color * 2;
    }
    else {
        color += d_color * abs(rand);
    }

    if (color[0] >= 0.5) {
        color = vec4(0.5 - 0.1*rand, 0.5 - 0.1*rand, 0.8, 1);
    }
    if (color[0] <= 0.1) {
       color = vec4(0.1 + 0.04*rand, 0.1 + 0.04*rand, 0.8, 1);
   }

    return color;
}

__global__ void updateTrochoidalParticles(Saiga::ArrayView<Particle> d_particles, float wave_length, float phase_speed, float steepness, float t, vec3 fluidDim) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id < d_particles.size()) {
        if (d_particles[ti.thread_id].rbID != -4) {
            return;
        }

        vec3 position = d_particles[ti.thread_id].relative;
        float dif = 2;
        float dif_x = 0;
        float dif_z = 0;
        float min_dif = 0.4;
        if (position[0] <= -fluidDim[0]/2) {
            dif_x = abs(-fluidDim[0]/2 - position[0]) + min_dif;
        }
        if (position[0] >= fluidDim[0]/2) {
            dif_x = abs(fluidDim[0]/2 - position[0]) + min_dif;
        }
        if (position[2] <= -fluidDim[2]/2) {
            dif_z = abs(-fluidDim[2]/2 - position[2])+ min_dif;
        }
        if (position[2] >= fluidDim[2]/2) {
            dif_z = abs(fluidDim[2]/2 - position[2]) + min_dif;
        }

        if (dif_z > 0 && dif_x > 0) {
            dif = min(max(dif_x, dif_z), dif);
        }
        else if (dif_z > 0) {
            dif = min(dif, dif_z);
        }
        else {
            dif = min(dif, dif_x);
        }

        // add different trochoidal waves
        vec3 old_position = position;
        // main wave
        position += trochoidalWaveOffset(d_particles[ti.thread_id].relative, vec2(1, 1), wave_length, steepness, t, dif);
        // small waves
        position += trochoidalWaveOffset(d_particles[ti.thread_id].relative, vec2(0.3, 0.8), wave_length * 0.8, steepness * 0.8, t, dif);
        position += trochoidalWaveOffset(d_particles[ti.thread_id].relative, vec2(0.8, 0.3), wave_length * 0.5, steepness * 0.95, t, dif);
        position += trochoidalWaveOffset(d_particles[ti.thread_id].relative, vec2(0.5, 0.5), wave_length * 0.7, steepness * 0.9, t, dif);
        
        // huge waves
        position += trochoidalWaveOffset(d_particles[ti.thread_id].relative, vec2(1, 0.4), wave_length * 2, steepness * 1.5, t, dif);
        position += trochoidalWaveOffset(d_particles[ti.thread_id].relative, vec2(0.1, 0.9), wave_length * 1.9, steepness * 1.2, t, dif);

        d_particles[ti.thread_id].position = position;
        d_particles[ti.thread_id].predicted = position;


        vec4 color = d_particles[ti.thread_id].color;
        d_particles[ti.thread_id].color = colorTrochoidalParticles(position, old_position, color);
    }
}

__global__ void shootCannon(Saiga::ArrayView<Particle> particles, RigidBody *rigidBodies, vec3 direction, float speed, int shipId, int ball1Id, int ball2Id, int cannonId, bool regular_ball) {
    Saiga::CUDA::ThreadInfo<> ti;
    int ballId = (regular_ball) ? ball1Id : ball2Id;

    if (ti.thread_id >= particles.size() || particles[ti.thread_id].rbID != ballId)
        return;

    Particle &p = particles[ti.thread_id];
    vec3 initialCannonDirection = vec3(0, 0.5, 1).normalized();
    p.velocity = (rigidBodies[cannonId].A * initialCannonDirection) * speed;


    vec3 cannonOffset = rigidBodies[shipId].A * vec3(0, 0.9, 0.8);
    // spawn cannonball at ship_position
    rigidBodies[ballId].originOfMass = rigidBodies[shipId].originOfMass + cannonOffset;
    // change position of each particle
    p.position = rigidBodies[p.rbID].A * p.relative + rigidBodies[p.rbID].originOfMass;
}

__global__ void resetEnemyParticles(Saiga::ArrayView<Particle> particles, RigidBody *rigidBodies, ClothConstraint *d_constraintListCloth, vec3 mapDim, vec3 fluidDim, ShipInfo* d_shipInfos, int* d_shipInfosCounter, float random, int enemyGridDim) {   
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    
    Particle &p = particles[ti.thread_id];

    for (int shipIdx = 0; shipIdx < *d_shipInfosCounter; shipIdx++) {
        ShipInfo &shipInfo = d_shipInfos[shipIdx];

        int shipRbId = -1;
        bool cloth = false;
        bool ship = false;
        bool penguin = false;
        bool constraint = false;

        if (p.rbID == -3) {
            if (p.id >= shipInfo.clothStart && p.id < shipInfo.clothEnd) {
                shipRbId = shipInfo.rbID;
                cloth = true;
            }
        } else if (p.rbID == shipInfo.rbID) {
            shipRbId = shipInfo.rbID;
            ship = true;
        } else if (p.rbID == shipInfo.penguinID) {
            shipRbId = shipInfo.rbID;
            penguin = true;
        }
        if (ti.thread_id >= shipInfo.constraintsStart && ti.thread_id < shipInfo.constraintsEnd) {
            shipRbId = shipInfo.rbID;
            constraint = true;
        }
        if (shipRbId == -1) {
            continue;
        }

        vec3 originOfMass = rigidBodies[shipRbId].originOfMass;

        if (originOfMass[0] <= -mapDim[0]/2 || originOfMass[0] >= mapDim[0]/2 || originOfMass[2] <= -mapDim[2]/2 || originOfMass[2] >= mapDim[2]/2) {
            originOfMass = {-fluidDim[0]/2 * random, 2.2, -mapDim[2]/2 + 3};
            if (cloth) {
                int x = 16; // has to be dimX from spawn
                int z = 1;
                int idx = p.id - shipInfo.clothStart;

                int xPos = (idx) % x;
                int zPos = ((idx - xPos) / x) % z;
                int yPos = (((idx - xPos) / x) - zPos) / z;
                vec3 pos = {xPos, yPos, zPos};

                float scaling = 0.25; // has to be clothDistance from spawn
                pos *= scaling;
                //vec3 offset = vec3{-0.8, -0.5, -3}; // from spawnPos
                vec3 offset = vec3{-0.8, -0.5, -2}; // from spawnPos
                vec3 clothCorner = {-1, 2.75, 2.5};
                pos += clothCorner + offset;

                p.position = originOfMass + pos;
                p.predicted = p.position;
                p.d_predicted = {0, 0, 0};
                p.velocity = {0, 0, 0};
                p.lambda = 0;
            } else if (ship) {
                rigidBodies[shipRbId].A = mat3::Identity();
                p.position = rigidBodies[shipRbId].A * p.relative + originOfMass;
                p.predicted = p.position;
                p.d_predicted = {0, 0, 0};
                p.velocity = {0, 0, 0};
            } else if (penguin) {
                vec3 pos = {0.5, 0.7, 1.5};
                p.position = rigidBodies[shipInfo.penguinID].A * p.relative + originOfMass + pos;
                p.predicted = p.position;
                p.d_predicted = {0, 0, 0};
                p.velocity = {0, 0, 0};
            }
            if (constraint) {
                d_constraintListCloth[ti.thread_id].active = 150;
            }
        }
    }
    
}

// merge this function with kernel
__device__ void moveRigidBodyEnemies(Saiga::ArrayView<Particle> particles, RigidBody *rigidBodies, vec3 mapDim, vec3 fluidDim, int rbID, float forward, float rotate, float stabilize = 0.01) {

    vec3 direction = rigidBodies[rbID].A * vec3{0, 1, 0};
    vec3 directionInit = rigidBodies[rbID].initA * vec3{0, 1, 0};
    Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(direction, (directionInit * stabilize + direction * (1 - stabilize)).normalized());
    mat3 normRotation = q.normalized().toRotationMatrix();

    vec3 rot = normRotation.eulerAngles(1, 0, 2);
    rot[0] += rotate * 0.0002;
    normRotation = Eigen::AngleAxisf(rot[0], vec3::UnitY())
        * Eigen::AngleAxisf(rot[1], vec3::UnitX())
        * Eigen::AngleAxisf(rot[2], vec3::UnitZ());

    mat3 rotMat = normRotation * rigidBodies[rbID].A;
    rigidBodies[rbID].A = rotMat;

    vec3 direction3d = rigidBodies[rbID].A * vec3{0, 0, 1};
    vec3 direction2d ={direction3d.x(), 0, direction3d.z()};
    direction2d.normalize();
    // max speed
    float maxSpeed = 2.5;
    vec3 currentDirection = rigidBodies[rbID].originOfMass - rigidBodies[rbID].lastOriginOfMass;
    currentDirection[1] = 0;
    currentDirection.normalize();
    float dot = abs(currentDirection.dot(direction2d));
    if (dot * rigidBodies[rbID].speed < maxSpeed)
        rigidBodies[rbID].originOfMass += dot * direction2d * forward * 0.003;

    // fix ships in trochoidal area
    vec3 originOfMass = rigidBodies[rbID].originOfMass;
    if (originOfMass[0] <= -fluidDim[0]/2 || originOfMass[0] >= fluidDim[0]/2 || originOfMass[2] <= -fluidDim[2]/2 || originOfMass[2] >= fluidDim[2]/2) {
        if (originOfMass[1] < 2.2)
            rigidBodies[rbID].originOfMass[1] += 0.005;
        if (originOfMass[1] < 2.0)
            rigidBodies[rbID].originOfMass[1] += 0.01;
        if (originOfMass[1] < 1.8)
            rigidBodies[rbID].originOfMass[1] = 1.8;
    }
}


// only considers x and z
__device__ int computeTurn(vec3 oldDirection, vec3 newDirection) {
    oldDirection.normalize();
    newDirection.normalize();
    float dotProduct = oldDirection[0] * newDirection[0] + oldDirection[2] * newDirection[2];
    float angle = acosf(dotProduct);

    // do not change direction if angle is too small
    if (angle < 0.10) {
        // 0.1 around 6 degree
        return 0;
    }
        
    float crossProduct = oldDirection[0] * newDirection[2] - newDirection[0] * oldDirection[2];
    int turn = (crossProduct > 0) ? -1 : 1;

    return turn;
}

__global__ void movePlayerPenguin(Saiga::ArrayView<Particle> particles, RigidBody *rigidBodies, int shipId, int penguinId) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id > 0)
        return;

    vec3 originOfMassShip = rigidBodies[shipId].originOfMass;
    vec3 penguinOffset = rigidBodies[shipId].A * vec3(0, 1.15, -0.7);
    rigidBodies[penguinId].originOfMass = originOfMassShip + penguinOffset;

  }

__global__ void moveCannon(Saiga::ArrayView<Particle> particles, RigidBody *rigidBodies, vec3 cameraDirection, int shipId, int cannonId) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id > 0)
        return;

    vec3 originOfMassShip = rigidBodies[shipId].originOfMass;
    vec3 cannonOffset = rigidBodies[shipId].A * vec3(0, 0.9, 0.7);
    rigidBodies[cannonId].originOfMass = originOfMassShip + cannonOffset;

    // project vector on xz axis and compute angle between z axis and vector
    // use this angle to rotate cannon on y axis
    float z_angle = atanf(cameraDirection[0] / cameraDirection[2]);
    
    // third quadrant, add -90 degree
    if (cameraDirection[2] < 0 && cameraDirection[0] < 0) {
        z_angle = -M_PI/2 - (M_PI/2 - atanf(cameraDirection[0] / cameraDirection[2]));
    }

    // fourth quadrant, add 90 degree
    if (cameraDirection[2] < 0 && cameraDirection[0] >= 0) {
        z_angle = M_PI/2 + (M_PI/2 - atanf(cameraDirection[0] / abs(cameraDirection[2])));
    }

    // compute angle between xz plane and y component of vector
    vec3 plane_vec = vec3(cameraDirection[0], 0, cameraDirection[2]);
    float plane_angle =  -atanf(cameraDirection[1]/plane_vec.norm());

    // clamp angle to min and max values
    plane_angle = max(-M_PI/4, plane_angle);
    plane_angle = min(0.26, plane_angle);

    // apply rotation on y axis first, then x axis
    rigidBodies[cannonId].A = Eigen::AngleAxisf(z_angle, Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(plane_angle, Eigen::Vector3f::UnitX()) * Eigen::AngleAxisf(0, Eigen::Vector3f::UnitZ());
}


__global__ void moveEnemies(Saiga::ArrayView<Particle> particles, RigidBody *rigidBodies, int * d_enemyGridWeight, int * d_enemyGridId, vec3 mapDim, vec3 fluidDim, ShipInfo* d_shipInfos, int* d_shipInfosCounter, float random, int enemyGridDim, float enemyGridCell, int ball) {
    Saiga::CUDA::ThreadInfo<> ti;
    bool is_ship_rbID = false;
    int penguinID;
    for (int shipIdx = 0; shipIdx < *d_shipInfosCounter; shipIdx++) {
        ShipInfo &shipInfo = d_shipInfos[shipIdx];
        if (ti.thread_id == shipInfo.rbID) {
            is_ship_rbID = true;
            penguinID = shipInfo.penguinID;
            break;
        }
    }
    if (is_ship_rbID) {
        vec3 originOfMass = rigidBodies[ti.thread_id].originOfMass;

        if (originOfMass[0] <= -mapDim[0]/2 || originOfMass[0] >= mapDim[0]/2 || originOfMass[2] <= -mapDim[2]/2 || originOfMass[2] >= mapDim[2]/2) {
            // reset
            originOfMass ={-fluidDim[0]/2 * random, 2.2, -mapDim[2]/2 + 3};
            rigidBodies[ti.thread_id].originOfMass = originOfMass;

            // penguin
            vec3 pos = {0.5, 0.7, 1.5};
            rigidBodies[penguinID].originOfMass = originOfMass + pos;
            return;
        }

        vec3 oldDirection = rigidBodies[ti.thread_id].A * vec3{0, 0, 1};
        oldDirection[1] = 0;
        oldDirection.normalize();

        int row = int((originOfMass[2] + fluidDim[2]/2) / enemyGridCell);
        int col = int((originOfMass[0] + fluidDim[0]/2) / enemyGridCell);
        static const int X_CONSTS[14] ={-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -2, 2};
        static const int Z_CONSTS[14] ={-1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0,};

        // enemies should aim towards +z direction
        vec3 flee = vec3(0, 0, 1);
        // check all neighboring grid fields
        // TODO increase radius
        for (int i = 0; i < 14; i++) {
            int y = row + X_CONSTS[i];
            int x = col + Z_CONSTS[i];
            if (y >= 0 && y < enemyGridDim && x >= 0 && x < enemyGridDim) {
                if (y == row && x == col && d_enemyGridId[y * enemyGridDim + x] == ti.thread_id) {
                    // TODO fix for multiple ships at same grid
                    continue;
                }

                if (d_enemyGridWeight[y * enemyGridDim + x] > 0) { 
                    int enemyId = d_enemyGridId[y * enemyGridDim + x];
                    vec3 enemy_position = rigidBodies[enemyId].originOfMass;
                    vec3 away = originOfMass - enemy_position;
                    away.normalize();
                    flee += away * d_enemyGridWeight[y * enemyGridDim + x] * 0.5;
                }
            }
        }

        flee[1] = 0;
        flee.normalize();

        int turn = computeTurn(oldDirection, flee);
        float acceleration = 0.8;
        moveRigidBodyEnemies(particles, rigidBodies, mapDim, fluidDim, ti.thread_id, acceleration, turn, 0.005);
        // also stabilize penguin
        moveRigidBodyEnemies(particles, rigidBodies, mapDim, fluidDim, penguinID, 0, 0, 0.05); // TODO extra function for stabilize only?
    }
}

__global__ void fillEnemyGrid(Saiga::ArrayView<Particle> particles, RigidBody *rigidBodies, int * d_enemyGridWeight, int * d_enemyGridId, vec3 mapDim, vec3 fluidDim, ShipInfo* d_shipInfos, int* d_shipInfosCounter, float random, int enemyGridDim, float enemyGridCell, int ball1Id, int ball2Id, int iceStartId, int iceEndId, bool regular_ball) {
    Saiga::CUDA::ThreadInfo<> ti;
    bool is_ship_rbID = false;

    int ballId = (regular_ball) ? ball1Id : ball2Id;
    for (int shipIdx = 0; shipIdx < *d_shipInfosCounter; shipIdx++) {
        ShipInfo &shipInfo = d_shipInfos[shipIdx];
        if (ti.thread_id == shipInfo.rbID) {
            is_ship_rbID = true;
            break;
        }
    }

    if (is_ship_rbID || ti.thread_id == 0 || ti.thread_id == ballId || ti.thread_id >= iceStartId && ti.thread_id < iceEndId) {

        vec3 originOfMass = rigidBodies[ti.thread_id].originOfMass;

        int row = int((originOfMass[2] + fluidDim[2]/2) / enemyGridCell);
        int col = int((originOfMass[0] + fluidDim[0]/2) / enemyGridCell);

        if (row >= 0 && row < enemyGridDim && col >= 0 && col < enemyGridDim) {
            int weight = 1;
            // high weighting for player
            if (ti.thread_id == 0) {
                weight = 2;
            }
            else if (ti.thread_id == ballId) {
                weight = 2;
                // discard ball if it is underwater
                if (originOfMass[1] < 1.5)
                    return;
            }

            // add weight and overwrite id
            // TODO could implement linked list for enemies in same grid
            d_enemyGridId[row * enemyGridDim + col] = ti.thread_id;
            atomicAdd(&d_enemyGridWeight[row * enemyGridDim + col], weight);
        }
    }
}

void ParticleSystem::computeScore() {
    countHits<<<BLOCKS, BLOCK_SIZE>>>(d_particleHits, particleCount, d_rbHits, rigidBodyCount, d_score, d_shipInfos, d_shipInfosCounter);
    int reset = 0;
    int points;
    checkError(cudaMemcpy(&points, d_score, sizeof(int), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    score += points;
    checkError(cudaMemcpy(d_score, &reset, sizeof(int), cudaMemcpyHostToDevice));
}

void ParticleSystem::update(float dt) {
    last_dt = dt;
    if (passed_time >= max_time) {
        passed_time = max_time;
        game_over = true;
        return;
    }
    if (ammo_left <= 0 && cannon_timer >= cannon_timer_reset) {
        game_over = true;
        return;
    }
    passed_time += dt;
    if (bonus_flag && score > bonus_score) {
        bonus_flag = false;
        bonus_score = 0;
        ammo_left = ammo_bonus;
    }
    if (physics_mode == 0) {      
        const unsigned int BLOCKS_CELLS = Saiga::CUDA::getBlockCount(cellCount, BLOCK_SIZE);

        resetConstraintCounter<<<1, 32>>>(d_constraintCounter, d_constraintCounterWalls);
        resetCellListOptimized<<<BLOCKS_CELLS, BLOCK_SIZE>>>(d_cell_list, cellCount, particleCount);
        CUDA_SYNC_CHECK_ERROR();

        if (control_cannonball == 1 && (cannon_timer >= cannon_timer_reset || debug_shooting) && ammo_left > 0) {
            resetHits<<<BLOCKS, BLOCK_SIZE>>>(d_particleHits, particleCount, d_rbHits, rigidBodyCount);
            CUDA_SYNC_CHECK_ERROR();
            shootCannon<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_rigidBodies, camera_direction, cannonball_speed, objects["player"], objects["ball_1"], objects["ball_2"], objects["cannon"], regular_ball);
            CUDA_SYNC_CHECK_ERROR();
            cannon_timer = 0;

            // ammo
            ammo_left--;
            bonus_flag = true;
            bonus_score = score;
        }

        float random = linearRand(-0.9, 0.9);
        resetEnemyGrid<<<BLOCKS, BLOCK_SIZE>>>(d_enemyGridWeight, enemyGridDim);
        fillEnemyGrid<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_rigidBodies, d_enemyGridWeight, d_enemyGridId, mapDim, fluidDim, d_shipInfos, 
            d_shipInfosCounter, random, enemyGridDim, enemyGridCell, objects["ball_1"], objects["ball_2"], objects["ice_start"], objects["ice_end"]+1, regular_ball);
        
        updateParticlesPBD1_radius<<<BLOCKS, BLOCK_SIZE>>>(dt, gravity, d_particles, damp_v, particleRadiusWater, particleRadiusCloth, objects["cannon"], objects["player_penguin"]);


        calculateHash<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_particle_hash, d_cell_list, d_particle_list, cellDim, cellCount, cellSize);
        thrust::sort_by_key(thrust::device_pointer_cast(d_particle_hash), thrust::device_pointer_cast(d_particle_hash) + particleCount, d_particles.device_begin());
        createLinkedCellsOptimized<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_particle_hash, d_cell_list, d_particle_list, cellDim, cellCount, cellSize);
        createConstraintParticlesLinkedCellsRigidBodiesFluid<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_constraintListCloth, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, 
            maxConstraintNum, cellDim, cellCount, cellSize, objects["cannon"], objects["ball_1"], objects["ball_2"], d_particleHits, d_rbHits, regular_ball);
        computeScore();
        createConstraintWalls<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_walls, d_constraintListWalls, d_constraintCounterWalls, maxConstraintNumWalls, objects["ball_1"], objects["enemy_6"]+1);
        
        updateLookupTable<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_particleIdLookup);
        CUDA_SYNC_CHECK_ERROR();

        float w_poly_d_q = W_poly6(delta_q * h, h);
        float calculatedRelaxP = relax_p;

        for (int i = 0; i < solver_iterations; i++) {
            if (i == 0) { // only calculate fluid stuff once (performance)
                computeDensityAndLambda<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, h, epsilon_spiky, omega_lambda_relax, particle_radius_rest_density, fluidDim);
                updateParticlesPBD2IteratorFluid<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, h, epsilon_spiky, particle_radius_rest_density, artificial_pressure_k, artificial_pressure_n, w_poly_d_q);
            }
            
            if (use_calculated_relax_p) {
                calculatedRelaxP = 1 - pow(1 - calculatedRelaxP, 1.0/(i+1));
            }
            updateRigidBodies();

            solverPBDParticlesSDF<<<Saiga::CUDA::getBlockCount(maxConstraintNum, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_constraintList, d_constraintCounter, maxConstraintNum, relax_p, d_rigidBodies, mu_k, mu_s, mu_f);
            solverPBDWalls<<<Saiga::CUDA::getBlockCount(maxConstraintNumWalls, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_walls, d_constraintListWalls, d_constraintCounterWalls, maxConstraintNumWalls, relax_p, mu_k, mu_s, mu_f);
            
            solverPBDCloth<<<Saiga::CUDA::getBlockCount(maxConstraintNumCloth, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_constraintListCloth, d_constraintCounterCloth, maxConstraintNumCloth, d_particleIdLookup, cloth_break_distance);
            if (test_bool)
                solverPBDClothBending<<<Saiga::CUDA::getBlockCount(maxConstraintNumClothBending, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_constraintListClothBending, d_constraintCounterClothBending, maxConstraintNumClothBending, d_particleIdLookup, test_float);
            
            updateParticlesPBD2Iterator<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, calculatedRelaxP);
            CUDA_SYNC_CHECK_ERROR();
        }

        //constraintsShapeMatchingRB
        //constraintsShapeMatchingRB();

        updateRigidBodies();
        controlRigidBody(0, control_forward, control_rotate, dt);
        CUDA_SYNC_CHECK_ERROR();
        resetEnemyParticles<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_rigidBodies, d_constraintListCloth, mapDim, fluidDim, d_shipInfos, d_shipInfosCounter, random, enemyGridDim);
        CUDA_SYNC_CHECK_ERROR();
        moveEnemies<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_rigidBodies, d_enemyGridWeight, d_enemyGridId, mapDim, fluidDim, d_shipInfos, d_shipInfosCounter, random, enemyGridDim, enemyGridCell, objects["ball_1"]);
        CUDA_SYNC_CHECK_ERROR();
        moveCannon<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_rigidBodies, camera_direction, objects["player"], objects["cannon"]);
        CUDA_SYNC_CHECK_ERROR();
        movePlayerPenguin<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_rigidBodies, objects["player"], objects["player_penguin"]);
        CUDA_SYNC_CHECK_ERROR();
        resolveRigidBodyConstraints<<<BLOCKS, BLOCK_SIZE>>>(d_particles, particleCount, d_rigidBodies, objects["player_penguin"]);
        CUDA_SYNC_CHECK_ERROR();
        updateRigidBodySpeed<<<BLOCKS, BLOCK_SIZE>>>(d_rigidBodies, maxRigidBodyCount, dt);
        CUDA_SYNC_CHECK_ERROR();

        updateTrochoidalParticles<<<BLOCKS, BLOCK_SIZE>>>(d_particles, wave_number, phase_speed, steepness, dt * steps, fluidDim);
        updateParticlesPBD2<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, relax_p);

        computeVorticityAndViscosity<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, h, epsilon_spiky, c_viscosity);
        applyVorticityAndViscosity<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, h, epsilon_spiky, epsilon_vorticity, wind_direction, wind_speed);
        CUDA_SYNC_CHECK_ERROR();
        
        cudaDeviceSynchronize();
    }
    steps += 1;
    if (cannon_timer < cannon_timer_reset)
        cannon_timer += 1;
}

__global__ void moveRigidBody(Saiga::ArrayView<Particle> particles, int particleCountRB, RigidBody *rigidBodies, int rbID, float forward, float rotate, float stabilize = 0.01) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id > 0)
        return;

    vec3 direction = rigidBodies[rbID].A * vec3{0, 1, 0};
    vec3 directionInit = rigidBodies[rbID].initA * vec3{0, 1, 0};
    Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(direction, (directionInit * stabilize + direction * (1 - stabilize)).normalized());
    mat3 normRotation = q.normalized().toRotationMatrix();

    vec3 rot = normRotation.eulerAngles(1, 0, 2);
    rot[0] += rotate * 0.0003;
    normRotation = Eigen::AngleAxisf(rot[0], vec3::UnitY())
        * Eigen::AngleAxisf(rot[1], vec3::UnitX())
        * Eigen::AngleAxisf(rot[2], vec3::UnitZ());

    mat3 rotMat = normRotation * rigidBodies[rbID].A;
    rigidBodies[rbID].A = rotMat;

    vec3 direction3d = rigidBodies[rbID].A * vec3{0, 0, 1};
    vec3 direction2d = {direction3d.x(), 0, direction3d.z()};
    direction2d.normalize();
    // max speed
    float maxSpeed = 3.5;
    vec3 currentDirection = rigidBodies[rbID].originOfMass - rigidBodies[rbID].lastOriginOfMass;
    currentDirection[1] = 0;
    currentDirection.normalize();
    float dot = abs(currentDirection.dot(direction2d));
    if (dot * rigidBodies[rbID].speed < maxSpeed)
        rigidBodies[rbID].originOfMass += dot * direction2d * forward * 0.003;
}

void ParticleSystem::controlRigidBody(int rbID, float forward, float rotate, float dt){
    float palyer_acceleration = 1.2;
    moveRigidBody<<<1, 32>>>(d_particles, particleCountRB, d_rigidBodies, rbID, forward * palyer_acceleration, rotate);
    cudaMemcpy(&ship_position, &d_rigidBodies[0].originOfMass, sizeof(vec3) * 1, cudaMemcpyDeviceToHost);
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

__global__ void rayInfo(Saiga::ArrayView<Particle> particles, Saiga::Ray ray, thrust::pair<int, float> *list, int *rayHitCount, int min, RigidBody *d_rigidBodies, ClothConstraint *d_constraintListCloth, ShipInfo *d_shipInfos, int *d_shipInfosCounter, int fishID, int fishStart) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= 1000)
        return;
    if (ti.thread_id == 0) {
        Particle &particle = particles[list[min].first];
        printf("idx: %i; id: %i; rbID: %i, position: %f, radius: %f, mass: %f, rgb: %f, %f, %f;\n", list[min].first, particle.id, particle.rbID, particle.predicted, particle.radius, 1.0f/particle.massinv, particle.color[0], particle.color[1], particle.color[2]);
        
        int shipIdx = -1;
        bool penguin = false;
        for (int idx = 0; idx < *d_shipInfosCounter; idx++) {
            ShipInfo &shipInfo = d_shipInfos[idx];
            if (particle.rbID == shipInfo.rbID || particle.rbID == shipInfo.penguinID) {
                shipIdx = idx;
                break;
            }
        }
        if (particle.rbID >= 0) {
            vec3 pos = d_rigidBodies[particle.rbID].originOfMass;
            printf("rbID: %i; position: %f, %f, %f; speed: %f\n", particle.rbID, pos[0], pos[1], pos[2], d_rigidBodies[particle.rbID].speed);
        }
        if (shipIdx > -1) {
            ShipInfo &shipInfo = d_shipInfos[shipIdx];
            int offset = 0;
            if (particle.rbID == shipInfo.penguinID) {
                penguin = true;
                printf("penguin ");
                offset = particle.id - shipInfo.penguinStart;
            } else {
                offset = particle.id - shipInfo.shipStart;
            }

            printf("belongs to ship: %i, model rbID: %i; offset in model: %i\n", shipIdx, particle.rbID, offset);
        } else if (particle.rbID == fishID) {
            int offset = particle.id - fishStart;
            printf("Fish: model rbID: %i; offset in model: %i\n", particle.rbID, offset);
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
    thrust::device_vector<thrust::pair<int, float>> d_vec(1000);
    resetCounter<<<1, 32>>>(d_rayHitCount);
    rayList<<<BLOCKS, BLOCK_SIZE>>>(d_particles, ray, thrust::raw_pointer_cast(&d_vec[0]), d_rayHitCount);
    CUDA_SYNC_CHECK_ERROR();
    int N = thrust::remove_if(d_vec.begin(), d_vec.end(), remove_predicate()) - d_vec.begin();
    if (N == 0)
        return;
    int min = thrust::min_element(d_vec.begin(), d_vec.begin() + N, compare_predicate()) - d_vec.begin();
    if (action_mode == 0) {
        rayColor<<<BLOCKS, BLOCK_SIZE>>>(d_particles, ray, thrust::raw_pointer_cast(&d_vec[0]), d_rayHitCount, min, color);
    } else if (action_mode == 1) {
        rayImpulse<<<BLOCKS, BLOCK_SIZE>>>(d_particles, ray, thrust::raw_pointer_cast(&d_vec[0]), d_rayHitCount, min);
    } else if (action_mode == 2) {
        rayExplosion<<<BLOCKS, BLOCK_SIZE>>>(d_particles, ray, thrust::raw_pointer_cast(&d_vec[0]), d_rayHitCount, min, true, explosion_force);
    } else if (action_mode == 3) {
        rayExplosion<<<BLOCKS, BLOCK_SIZE>>>(d_particles, ray, thrust::raw_pointer_cast(&d_vec[0]), d_rayHitCount, min, false, explosion_force);
    } else if (action_mode == 4) {
        rayInfo<<<BLOCKS, BLOCK_SIZE>>>(d_particles, ray, thrust::raw_pointer_cast(&d_vec[0]), d_rayHitCount, min, d_rigidBodies, d_constraintListCloth, d_shipInfos, d_shipInfosCounter, fishID, particleFishStart);
    }
    CUDA_SYNC_CHECK_ERROR();
}