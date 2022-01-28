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

// time
float t = 0;

void ParticleSystem::setDevicePtr(void* particleVbo) {
    d_particles = ArrayView<Particle>((Particle*) particleVbo, particleCount);
}

__global__ void updateParticlesPBD1_radius(float dt, vec3 gravity, Saiga::ArrayView<Particle>particles, float damp_v, float particleRadiusWater, float particleRadiusCloth) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particles.size())
        return;
    Particle &p = particles[ti.thread_id];

    if (p.fixed)
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

__global__ void resetOcean(Saiga::ArrayView<Particle> d_particles, int startId, int endId, int xMax, int zMax, vec3 corner, vec4 color, vec3 fluidDim) {
    Saiga::CUDA::ThreadInfo<> ti;
    int id = ti.thread_id;

    if (id >= startId && id < endId) {
        int y = (id - startId) / (xMax * zMax);
        int z = ((id - startId) - (y * xMax * zMax)) / xMax;
        int x = ((id - startId) - (y * xMax * zMax)) % xMax;

        Particle &p = d_particles[id];
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

__global__ void initSingleRigidBodyParticle(Saiga::ArrayView<Particle> particles, int id, vec3 pos, vec3 sdf, vec4 color, int particleCountRB, RigidBody *rigidBodies, bool fixed=false, float mass=1.0, float particleRadius=0.5) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id > 0)
        return;
    
    particles[particleCountRB].position = pos;
    particles[particleCountRB].predicted = pos;
    particles[particleCountRB].rbID = id;

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
int ParticleSystem::loadObj(int rigidBodyCount, int particleCountRB, vec3 pos, vec3 rot, vec4 color) {
    Saiga::UnifiedModel model("objs/teapot.obj");
    Saiga::UnifiedMesh mesh = model.CombinedMesh().first;
    std::vector<Triangle> triangles = mesh.TriangleSoup();
    // 1
    Saiga::AABB bb = model.BoundingBox(); // mesh. or model.BoundingBox()
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
int ParticleSystem::loadBox(int rigidBodyCount, int particleCountRB, ivec3 dim, vec3 pos, vec3 rot, vec4 color, bool fixed=false, float mass=1.0, float scaling=1.0, float particleRadius=0.5) {    
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
                        initSingleRigidBodyParticle<<<1, 32>>>(d_particles, rigidBodyCount, position, sdf, color, particleCountRB++, d_rigidBodies, fixed, mass, particleRadius);
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

__global__ void covariance(Saiga::ArrayView<Particle> particles, int particleCountRB, RigidBody *rigidBodies) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particleCountRB)
        return;
    Particle &p = particles[ti.thread_id];
    if (p.rbID >= 0) {
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

__global__ void resolveRigidBodyConstraints(Saiga::ArrayView<Particle> particles, int particleCountRB, RigidBody *rigidBodies) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= particleCountRB)
        return;
    Particle &p = particles[ti.thread_id];
    if (p.rbID >= 0) {
        // dx = (Q*r + c) - p
        p.predicted += (rigidBodies[p.rbID].A * p.relative + rigidBodies[p.rbID].originOfMass) - p.predicted;
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

__global__ void initRigidBodiesRotation(RigidBody *rigidBodies, int rigidBodyCount) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= rigidBodyCount)
        return;
    RigidBody &rb = rigidBodies[ti.thread_id];
    rb.initA = rb.A;
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

void ParticleSystem::reset(int x, int z, vec3 corner, float distance, float randInitMul, int scenario, vec3 fluidDim, vec3 trochoidal1Dim, vec3 trochoidal2Dim, ivec2 layers) {
    int rbID = -1; // free particles
    vec4 color = {0.0f, 1.0f, 0.0f, 1.f};
    if (scenario >= 7) {
        color = {0.1f, 0.2f, 0.8f, 1.f};
        rbID = -2; // fluid
    }
    if (scenario == 12) { // trochoidal test scenario
        color ={0.1f, 0.2f, 0.8f, 1.f};
        rbID = -4; // trochoidal particles
    }

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
        steepness = 0.6;
        wind_direction = {-1.0, 0.0, -1.0};
        wind_speed = 0.7;
        solver_iterations = 1;
        c_viscosity = 0.02;
        epsilon_vorticity = 0.001;
        float distance = 0.5;

        // adds trochoidal particles
        // generate first layers of fluids and trochoidals
        resetOcean<<<BLOCKS, BLOCK_SIZE>>>(d_particles, 0, layers[0], x, z, corner, color, fluidDim);
        CUDA_SYNC_CHECK_ERROR();

        // generate top layer of fluids and trochoidals
        corner -= vec3(trochoidal2Dim[0], 0, trochoidal2Dim[2]);
        float height = trochoidal1Dim[1] - trochoidal2Dim[1];
        corner += vec3(0, height, 0);
        x += 1/distance * trochoidal2Dim[0] * 2;
        z += 1/distance * trochoidal2Dim[2] * 2;
        resetOcean<<<BLOCKS, BLOCK_SIZE>>>(d_particles, layers[0], layers[1], x, z, corner, color, fluidDim);
        CUDA_SYNC_CHECK_ERROR();
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

    if (scenario == 11 || scenario == 14) {
        vec3 rot = {0,0,0};
        ivec3 dim = {5,5,5};

        color = {1.0, 0., .0, 1};

        vec3 pos = {0, 10, 0};
        int objParticleCount = loadBox(rigidBodyCount++, particleCountRB, dim, pos, rot, color, false, 0.2, 0.5, 0.3);
        particleCountRB += dim.x() * dim.y() * dim.z();
    }

    if (scenario > 2 && scenario != 6 && scenario != 7 && scenario < 7)
        deactivateNonRB<<<BLOCKS, BLOCK_SIZE>>>(d_particles);
    CUDA_SYNC_CHECK_ERROR();
    
    resetRigidBody<<<BLOCKS, BLOCK_SIZE>>>(d_rigidBodies, rigidBodyCount);
    CUDA_SYNC_CHECK_ERROR();

    caclulateRigidBodyOriginOfMass<<<BLOCKS, BLOCK_SIZE>>>(d_particles, particleCountRB, d_rigidBodies);
    CUDA_SYNC_CHECK_ERROR();
    initRigidBodyParticles<<<BLOCKS, BLOCK_SIZE>>>(d_particles, particleCountRB, d_rigidBodies);
    CUDA_SYNC_CHECK_ERROR();

    updateRigidBodies();
    initRigidBodiesRotation<<<BLOCKS_RB, BLOCK_SIZE>>>(d_rigidBodies, rigidBodyCount);

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
            d = collideSphereSphere(pa_copy.radius, pb_copy.radius, pa_copy.predicted, pb_copy.predicted);
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

    ParticleCalc pa_copy;
    Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[idxA]), &pa_copy);
    ParticleCalc pb_copy;
    Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[idxB]), &pb_copy);

    // TODO mass von material abhaengig machen nicht aus particle lesen
    float m1 = pa.massinv;
    float m2 = pb.massinv;

    float d = collideSphereSphere(constraints[ti.thread_id].dist, 0, pa_copy.predicted, pb_copy.predicted);
    vec3 n = (pa_copy.predicted - pb_copy.predicted).normalized();
    float m = (m1 / (m1 + m2));
    vec3 dx1 = m * d * n;
    vec3 dx2 = - (1.0f - m) * d * n;

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

__global__ void createConstraintParticlesLinkedCellsRigidBodiesFluid(Saiga::ArrayView<Particle> particles, std::pair<int, int>* cell_list, int* particle_list, int *constraints, int *constraintCounter, int maxConstraintNum, ivec3 cell_dims, int cellCount, float cellSize) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id < particles.size()) {
        ParticleCalc pa;
        ParticleCalc pb;
        Saiga::CUDA::vectorCopy(reinterpret_cast<ParticleCalc*>(&particles[ti.thread_id]), &pa);
        int rbIDa = particles[ti.thread_id].rbID;

        if (rbIDa == -4)
            return;

        ivec3 cell_idx = calculateCellIdx(pa.predicted, cellSize); // actually pa.position but we only load predicted and its identical here

        static const int X_CONSTS[14] = {-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0};
        static const int Y_CONSTS[14] = {-1,-1,-1, 0, 0, 0, 1, 1, 1,-1,-1,-1, 0, 0};
        static const int Z_CONSTS[14] = {-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0};

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
                if (rbIDb == -4)
                    continue;
                if ( (rbIDa == -1 || rbIDb == -1 || rbIDa != rbIDb) &&
                        (i != 13 || neighbor_particle_idx > ti.thread_id) ) {
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
    float max_density = 7.5 * rho0inv - 1.0; // 1 * W_poly(0, h) + 9 * W_poly(0.5, h) // 1.57 + x * 0.66; x= 3: 3.5, 6: 5.5, 9: 7.5
    float non_spray = (C_density - min_density) / (max_density - min_density);
    non_spray = range(non_spray, 0, 1);
    float spray = 1.0f - (non_spray * non_spray);
    return spray;
}

__global__ void computeDensityAndLambda(Saiga::ArrayView<Particle> particles, std::pair<int, int>* cell_list, int* particle_list, int *constraints, int *constraintCounter, int maxConstraintNum, ivec3 cell_dims, int cellCount, float cellSize, float h, float epsilon_spiky, float omega_lambda_relax, float particleRadius) {
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
                        //int rbIDb = particles[neighbor_particle_idx].rbID;
                        
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

        // gischt (spray)
        float spray = calculateSpray(C_density, rho0inv);
        vec4 water_color = {0, 0, 0.8, 1};
        vec4 spray_color = {1, 1, 1, 1};
        float old_spray = particles[ti.thread_id].color[0];
        float new_spray = spray;
        if (new_spray < old_spray)
            new_spray = old_spray * 0.995;
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

__device__ vec3 trochoidalWaveOffset(vec3 gridPoint, vec2 direction, float wave_length, float steepness, float t) {
    direction = normalize(direction);
    float x = gridPoint[0];
    float y = gridPoint[1];
    float z = gridPoint[2];

    float k = 2 * M_PI / wave_length;
    // compute speed of waves
    float c = 9.8 / (k * 2.5);

    // amplitude
    float a = (steepness / 10 * y) / k;

    float f = k * (direction[0] * x + direction[1] * z - c * t);

    //float sin_f = sinf(f);
    //float cos_f = cosf(f);
    float sin_f;
    float cos_f;
    sincosf(f, &sin_f, &cos_f);

    float xOffset = direction[0] * a * sin_f;
    float yOffset = -a * cos_f;
    float zOffset = direction[1] * a * sin_f;

    return vec3(xOffset, yOffset, zOffset);
}

__global__ void updateTrochoidalParticles(Saiga::ArrayView<Particle> d_particles, float wave_length, float phase_speed, float steepness, float t) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id < d_particles.size()) {
        if (d_particles[ti.thread_id].rbID != -4) {
            return;
        }

        vec3 position = d_particles[ti.thread_id].relative;

        // add different trochoidal waves
        // main wave
        position += trochoidalWaveOffset(d_particles[ti.thread_id].relative, vec2(1, 0), wave_length, steepness, t);
        // small waves
        position += trochoidalWaveOffset(d_particles[ti.thread_id].relative, vec2(0.2, 0.8), wave_length * 0.8, steepness * 0.8, t);
        position += trochoidalWaveOffset(d_particles[ti.thread_id].relative, vec2(0.8, 0.2), wave_length * 0.5, steepness * 0.95, t);
        position += trochoidalWaveOffset(d_particles[ti.thread_id].relative, vec2(0.5, 0.5), wave_length * 0.7, steepness * 0.9, t);
        position += trochoidalWaveOffset(d_particles[ti.thread_id].relative, vec2(0.1, 0.9), wave_length * 1.2, steepness * 1.1, t);
        // huge waves
        position += trochoidalWaveOffset(d_particles[ti.thread_id].relative, vec2(1, 0), wave_length * 2, steepness * 1.5, t);
        position += trochoidalWaveOffset(d_particles[ti.thread_id].relative, vec2(0.1, 0.9), wave_length * 1.9, steepness * 1.2, t);
        position += trochoidalWaveOffset(d_particles[ti.thread_id].relative, vec2(0.9, 0.1), wave_length * 2.1, steepness * 1.1, t);

        d_particles[ti.thread_id].position = position;
        d_particles[ti.thread_id].predicted = position;
    }
}

void ParticleSystem::update(float dt) {
    last_dt = dt;
    if (physics_mode == 0) {      
        const unsigned int BLOCKS_CELLS = Saiga::CUDA::getBlockCount(cellCount, BLOCK_SIZE);

        resetConstraintCounter<<<1, 32>>>(d_constraintCounter, d_constraintCounterWalls);
        resetCellListOptimized<<<BLOCKS_CELLS, BLOCK_SIZE>>>(d_cell_list, cellCount, particleCount);
        updateParticlesPBD1_radius<<<BLOCKS, BLOCK_SIZE>>>(dt, gravity, d_particles, damp_v, particleRadiusWater, particleRadiusCloth);
        
        calculateHash<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_particle_hash, d_cell_list, d_particle_list, cellDim, cellCount, cellSize);
        thrust::sort_by_key(thrust::device_pointer_cast(d_particle_hash), thrust::device_pointer_cast(d_particle_hash) + particleCount, d_particles.device_begin());
        createLinkedCellsOptimized<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_particle_hash, d_cell_list, d_particle_list, cellDim, cellCount, cellSize);
        
        createConstraintParticlesLinkedCellsRigidBodiesFluid<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize);
        createConstraintWalls<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_walls, d_constraintListWalls, d_constraintCounterWalls, maxConstraintNumWalls);
        
        updateLookupTable<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_particleIdLookup);
        CUDA_SYNC_CHECK_ERROR();

        float w_poly_d_q = W_poly6(delta_q * h, h);
        float calculatedRelaxP = relax_p;

        for (int i = 0; i < solver_iterations; i++) {
            computeDensityAndLambda<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, h, epsilon_spiky, omega_lambda_relax, particle_radius_rest_density);
            updateParticlesPBD2IteratorFluid<<<BLOCKS, BLOCK_SIZE>>>(d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, h, epsilon_spiky, particle_radius_rest_density, artificial_pressure_k, artificial_pressure_n, w_poly_d_q);

            if (use_calculated_relax_p) {
                calculatedRelaxP = 1 - pow(1 - calculatedRelaxP, 1.0/(i+1));
            }
            updateRigidBodies();

            solverPBDParticlesSDF<<<Saiga::CUDA::getBlockCount(maxConstraintNum, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_constraintList, d_constraintCounter, maxConstraintNum, relax_p, d_rigidBodies, mu_k, mu_s, mu_f);
            solverPBDWalls<<<Saiga::CUDA::getBlockCount(maxConstraintNumWalls, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_walls, d_constraintListWalls, d_constraintCounterWalls, maxConstraintNumWalls, relax_p, mu_k, mu_s, mu_f);
            
            solverPBDCloth<<<Saiga::CUDA::getBlockCount(maxConstraintNumCloth, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_constraintListCloth, d_constraintCounterCloth, maxConstraintNumCloth, d_particleIdLookup);
            if (test_bool)
                solverPBDClothBending<<<Saiga::CUDA::getBlockCount(maxConstraintNumClothBending, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles, d_constraintListClothBending, d_constraintCounterClothBending, maxConstraintNumClothBending, d_particleIdLookup, test_float);
            
            updateParticlesPBD2Iterator<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, calculatedRelaxP);
            CUDA_SYNC_CHECK_ERROR();
        }

        //constraintsShapeMatchingRB
        //constraintsShapeMatchingRB();

        updateRigidBodies();
        controlRigidBody(0, control_forward, control_rotate, dt);
        resolveRigidBodyConstraints<<<BLOCKS, BLOCK_SIZE>>>(d_particles, particleCount, d_rigidBodies);
        CUDA_SYNC_CHECK_ERROR();

        updateParticlesPBD2<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, relax_p);
        updateTrochoidalParticles<<<BLOCKS, BLOCK_SIZE>>>(d_particles, wave_number, phase_speed, steepness, dt * steps);

        computeVorticityAndViscosity<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, h, epsilon_spiky, c_viscosity);
        applyVorticityAndViscosity<<<BLOCKS, BLOCK_SIZE>>>(dt, d_particles, d_cell_list, d_particle_list, d_constraintList, d_constraintCounter, maxConstraintNum, cellDim, cellCount, cellSize, h, epsilon_spiky, epsilon_vorticity, wind_direction, wind_speed);
        CUDA_SYNC_CHECK_ERROR();
        
        cudaDeviceSynchronize();
    }
    steps += 1;
}

__device__ float stabilize(float rot, float center, float stabilize, float max) {
    float new_rot = 0;
    if (rot > M_PI/2.0) {
        center += M_PI;
    } else if (rot < -M_PI/2.0) {
        center -= M_PI;
    }
    new_rot = rot * (1 - stabilize) + center * stabilize;
    return new_rot - rot;
}

__device__ float normalizeRotation(float rot) {
    if (rot > M_PI)
        rot = rot - 2 * M_PI;
    if (rot < -M_PI)
        rot = rot + 2 * M_PI;
    return rot;
}

__device__ void normalizeRotation(vec3 rot) {
    rot.x() = normalizeRotation(rot.x());
    rot.y() = normalizeRotation(rot.y());
    rot.z() = normalizeRotation(rot.z());
}

__global__ void moveRigidBody(Saiga::ArrayView<Particle> particles, int particleCountRB, RigidBody *rigidBodies, int rbID, float forward, float rotate) {
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id > 0)
        return;
    vec3 rot = rigidBodies[rbID].A.eulerAngles(1, 0, 2);
    vec3 initRot = rigidBodies[rbID].initA.eulerAngles(1, 0, 2);
    vec3 relRot = rot - initRot;

    // another approach
    vec3 direction3d = {1, 0, 0};
    direction3d = rigidBodies[rbID].A * direction3d;

    normalizeRotation(relRot);

    rot.x() += rotate * 0.001;
    rot.y() += stabilize(relRot.y(), 0, 0.01, M_PI/4.0);
    rot.z() += stabilize(relRot.z(), 0, 0.01, M_PI/4.0);
    
    normalizeRotation(rot);

    mat3 rotMat;
    rotMat = Eigen::AngleAxisf(rot.x(), vec3::UnitY())
        * Eigen::AngleAxisf(rot.y(), vec3::UnitX())
        * Eigen::AngleAxisf(rot.z(), vec3::UnitZ());
    rigidBodies[rbID].A = rotMat;

    //vec3 direction = {cosf(rotY), 0, sinf(rotY)};
    vec3 direction = {direction3d.x(), 0, direction3d.z()};
    direction.normalize();
    rigidBodies[rbID].originOfMass += direction * forward * 0.001;
}

void ParticleSystem::controlRigidBody(int rbID, float forward, float rotate, float dt){
    moveRigidBody<<<1, 32>>>(d_particles, particleCountRB, d_rigidBodies, rbID, forward, rotate);
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
    }
    CUDA_SYNC_CHECK_ERROR();
}