#pragma once

#include "saiga/core/glfw/all.h"
#include "saiga/cuda/interop.h"
#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/instancedBuffer.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/window/WindowTemplate.h"
#include "saiga/opengl/world/proceduralSkybox.h"

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

    // Deferred rendering and update functions (called from Saiga)
    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;
    virtual void render(RenderInfo render_info) override;
    void parallelUpdate(float dt) override {}

    // Key/Mouse Input
    void keyPressed(int key, int scancode, int mods) override;
    void mousePressed(int key, int x, int y) override;

   private:
    // Particles
    int numberParticles = 1000;

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
    Saiga::ProceduralSkybox skybox;
    std::shared_ptr<Saiga::DirectionalLight> sun;

    // ImGUI Stuff
    bool shouldRenderGUI = true;
    bool showSaigaGui    = false;
    ImGui::TimeGraph physicsGraph;
    void renderGUI();

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
