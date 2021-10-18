#include "agphys.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/math/random.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/cuda/cudaTimer.h"

#include "cuda_profiler_api.h"

Agphys::Agphys()
    : StandaloneWindow("config.ini")
#ifdef SAIGA_USE_FFMPEG
      // , enc(window->getWidth(), window->g)
#endif
{
    auto editor_layout = std::make_unique<EditorLayoutU>(false);
    editor_layout->RegisterImguiWindow("Agphys", EditorLayoutU::WINDOW_POSITION_LEFT);
    editor_layout->RegisterImguiWindow("Timings", EditorLayoutU::WINDOW_POSITION_LEFT);
    editor_layout->RegisterImguiWindow("ParticleSystem", EditorLayoutU::WINDOW_POSITION_RIGHT);
    editor_layout->RegisterImguiWindow("Interaction", EditorLayoutU::WINDOW_POSITION_RIGHT);
    editor_gui.SetLayout(std::move(editor_layout));

    // create a perspective camera
    float aspect = window->getAspectRatio();
    camera.setProj(60.0f, aspect, 0.1f, 400.0f);
    camera.setView(vec3(0, 20, 50), vec3(0, 1, 0), vec3(0, 1, 0));
    camera.enableInput();
    camera.movementSpeed     = 20;
    camera.movementSpeedFast = 50;
    camera.rotationPoint     = vec3(0, 0, 0);
    window->setCamera(&camera);

    groundPlane.asset = std::make_shared<ColoredAsset>(
        Saiga::CheckerBoardPlane(ivec2(150, 150), 1.0f, Colors::gainsboro, Colors::darkgray));

    sun = std::make_shared<DirectionalLight>();
    renderer->lighting.AddLight(sun);
    sun->setDirection(vec3(-1, -3, -2));
    sun->setColorDiffuse(vec3(0.8, 0.8, 1));
    sun->setIntensity(0.6f);
    sun->setAmbientIntensity(0.3f);

    sun->BuildCascades(3);
    sun->castShadows = true;


    std::string shader_name = "shader/particles.glsl";

    Saiga::ShaderCodeInjection depth_inj(GL_FRAGMENT_SHADER, "#define WRITE_DEPTH", 3);
    Saiga::ShaderCodeInjection shadow_inj(GL_GEOMETRY_SHADER, "#define SHADOW", 3);
    Saiga::ShaderCodeInjection shadow_inj2(GL_FRAGMENT_SHADER, "#define SHADOW", 3);

    particleShader      = Saiga::shaderLoader.load<Saiga::MVPShader>(shader_name, {depth_inj});
    particleDepthShader = Saiga::shaderLoader.load<Saiga::MVPShader>(shader_name, {depth_inj, shadow_inj, shadow_inj2});
    particleShaderFast  = Saiga::shaderLoader.load<Saiga::MVPShader>(shader_name, {});
    particleDepthShaderFast = Saiga::shaderLoader.load<Saiga::MVPShader>(shader_name, {shadow_inj, shadow_inj2});

    initParticles();

    std::cout << "Agphys Initialized!" << std::endl;
}

Agphys::~Agphys()
{
    std::cout << "~Agphys" << std::endl;
}


void Agphys::initParticles()
{
    destroyParticles();

    particleSystem = std::make_shared<ParticleSystem>(numberParticles);

    // initialize particles with some random values
    std::vector<Particle> particles(numberParticles);

    // Initialize particles
    for (auto& p : particles)
    {
        p.position = linearRand(vec3(-10, 0, -10), vec3(10, 10, 10));
        //        p.position = vec3(0,0,0);
        p.radius = 0.5f;
        p.color  = make_vec4(linearRand(vec3(0, 0, 0), vec3(1, 1, 1)), 1);
    }

    // upload particle array to opengl
    particleBuffer.set(particles, GL_DYNAMIC_DRAW);
    particleBuffer.setDrawMode(GL_POINTS);

    interop.registerBuffer(particleBuffer.getBufferObject());
    resetParticles();
}

void Agphys::destroyParticles()
{
    interop.unregisterBuffer();
    particleSystem.reset();
}

void Agphys::update(float dt)
{
    if (renderer->use_keyboard_input_in_3dview) camera.update(dt);

    sun->fitShadowToCamera(&camera);
    sun->fitNearPlaneToScene(AABB(vec3(-125, 0, -125), vec3(125, 50, 125)));

#ifdef SAIGA_USE_FFMPEG
    enc.update();
#endif

    if (pause) return;

    map();
    float t;
    {
        Saiga::CUDA::ScopedTimer tim(t);
        particleSystem->update(dt);
    }
    physicsGraph.addTime(t);
    unmap();
}

void Agphys::interpolate(float dt, float interpolation)
{
    if (renderer->use_mouse_input_in_3dview) camera.interpolate(dt, interpolation);
}


void Agphys::render(RenderInfo render_info)
{
    auto [camera, render_pass, render_flags, render_context] = render_info;
    if (render_pass == RenderPass::Deferred)
    {
        // render the particles from the viewpoint of the camera
        if (renderParticles)
        {
            auto shader = (renderShadows) ? particleShader : particleShaderFast;
            if (shader->bind()) {
                shader->uploadModel(mat4::Identity());
                particleBuffer.bind();
                particleBuffer.draw(0, particleSystem->particleCount);
                particleBuffer.unbind();
                shader->unbind();
            }
        }

        groundPlane.render(camera);
    }
    else if (render_pass == RenderPass::Shadow)
    {
        if (renderParticles && renderShadows)
        {
            auto shader = (renderShadows) ? particleDepthShader : particleDepthShaderFast;
            if (shader->bind()) {
                shader->uploadModel(mat4::Identity());
                particleBuffer.bind();
                particleBuffer.draw(0, particleSystem->particleCount);
                particleBuffer.unbind();
                shader->unbind();
            }
        }
        groundPlane.renderDepth(camera);
    }
    else if (render_pass == RenderPass::Forward)
    {
        skybox.render(camera);
    }
    else if (render_pass == RenderPass::GUI)
    {
        if (showSaigaGui) window->renderImGui();

        renderGUI();
    }
}


void Agphys::resetParticles()
{
    map();
    // reset particles
    unmap();
}



template <>
void VertexBuffer<Particle>::setVertexAttributes()
{
    // setting the vertex attributes correctly is required, so that the particle shader knows how to read the input
    // data. adding or removing members from the particle class may or may not requires you to change the
    // vertexAttribPointers.
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    // position radius
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), NULL);
    // radius
    //    glVertexAttribPointer(1,1, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*) (3 * sizeof(GLfloat)) );

    //#ifdef COLOR_IN_PARTICLE
    // color
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(4 * sizeof(GLfloat)));
    //#endif
}
