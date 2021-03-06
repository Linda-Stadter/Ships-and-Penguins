#pragma once

#include "saiga/core/glfw/all.h"
#include "saiga/cuda/interop.h"
#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/instancedBuffer.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/window/WindowTemplate.h"
#include "saiga/opengl/world/proceduralSkybox.h"

#include "saiga/opengl/world/skybox.h"

#include "particleSystem.h"

#undef SAIGA_USE_FFMPEG
#ifdef SAIGA_USE_FFMPEG
#include "saiga/opengl/ffmpeg/videoEncoder.h"
#endif

using Saiga::AABB;
using Saiga::Camera;
using Saiga::SimpleAssetObject;
using namespace Saiga;


class Agphys : public StandaloneWindow<WindowManagement::GLFW, DeferredRenderer>,
               public glfw_KeyListener,
               public glfw_MouseListener
{
   public:
    Agphys();
    ~Agphys();

    void initParticles();
    void destroyParticles();
    void resetParticles();

    void initWalls();

    void loadScenario();

    // Deferred rendering and update functions (called from Saiga)
    void update(float dt) override;
    void updateSingleStep(float dt);
    void interpolate(float dt, float interpolation) override;
    virtual void render(RenderInfo render_info) override;
    void parallelUpdate(float dt) override {}

    // Key/Mouse Input
    void keyPressed(int key, int scancode, int mods) override;
    void mousePressed(int key, int x, int y) override;

    // controls
    void updateControlsAndCamera(float delta);
    std::vector<int> keyboardmap = {GLFW_KEY_T, GLFW_KEY_G, GLFW_KEY_F, GLFW_KEY_H};
    // camera
    bool camera_follow = false;
    vec3 camera_direction = {1, 0, 0};

   private:
    unsigned int steps = 0;

    // Particles
    int numberParticles = 10000;
    float distance = 0.99;
    int xCount = 20;
    int zCount = 20;
    vec3 corner = {-10, 15, -10};
    vec3 boxDim = {40, 40, 40};
    vec3 fluidDim = {60, 80, 60};
    vec3 trochoidal1Dim = {10, 2.5, 10};
    vec3 trochoidal2Dim = {10, 1, 20};
    ivec2 layers = {0, 0};

    const char* scenarios[15] = {"N Particles", "10K Particles", "1M Particles", "Rigid Bodies (Cuboids)", "Rigid Bodies (Cuboids + Teapod)", 
                                 "Rigid Bodies (SDF demo)", "Particles + Rigid Bodies (Cuboids + Teapod)", "Fluid (dam break + Obstacle)", 
                                 "Fluid (dam break demo)", "Fluid (double dam break demo)", "Cloth test", "Project Test", "Trochoidal Waves",
                                 "SPH Trochoidal Mixture Test", "Game Map"};
    int scenario = 14;

    vec3 last_camera_direction;

    float randInitMul = 0.001;

    float stepsize = 0.01;

    std::shared_ptr<ParticleSystem> particleSystem;


    // Particle Rendering
    bool renderParticles = true;
    bool renderShadows   = true;

    Saiga::CUDA::Interop interop;
    Saiga::VertexBuffer<Particle> particleBuffer;

    std::shared_ptr<Saiga::MVPShader> particleShader, particleShaderFast;
    std::shared_ptr<Saiga::MVPShader> particleDepthShader, particleDepthShaderFast;

    // Saiga Rendering Stuff
    Saiga::Glfw_Camera<Saiga::PerspectiveCamera> camera;
    SimpleAssetObject groundPlane;
    std::shared_ptr<Saiga::Skybox> skybox;
    std::shared_ptr<Saiga::DirectionalLight> sun;

    // ImGUI Stuff
    bool shouldRenderGUI = true;
    bool showSaigaGui    = false;
    ImGui::TimeGraph physicsGraph;
    void renderGUI();

    void toggleGameMode();
    bool gameMode = false;

    // Other stuff
    bool pause = false;
#ifdef SAIGA_USE_FFMPEG
    Saiga::VideoEncoder enc;
#endif

    void map();
    void unmap();
};

namespace Saiga
{
template <>
void VertexBuffer<Particle>::setVertexAttributes();
}  // namespace Saiga
