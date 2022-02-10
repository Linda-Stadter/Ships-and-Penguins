#include "agphys.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/math/random.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/cuda/cudaTimer.h"

#include "saiga/core/util/keyboard.h"

#include "cuda_profiler_api.h"

#include "cuda_runtime.h"

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
    camera.rotationPoint     = vec3(0, 0, 0);
    camera.enableInput();
    camera.movementSpeed     = 20;
    camera.movementSpeedFast = 50;
    window->setCamera(&camera);

    groundPlane.asset = std::make_shared<ColoredAsset>(
        Saiga::CheckerBoardPlane(ivec2(300, 300), 1.0f, {.1, .1, .8, 1}, {.18, .18, .8, 1}));

    sun = std::make_shared<DirectionalLight>();
    renderer->lighting.AddLight(sun);
    sun->setDirection(vec3(-1.5, -2.5, 2));
    sun->setColorDiffuse(vec3(0.8, 0.8, 1));
    sun->setIntensity(0.6f);
    sun->setAmbientIntensity(0.3f);

    sun->BuildCascades(3);
    sun->castShadows = true;

    Image img("textures/skymap.jpg");
    skybox = std::make_shared<Skybox>(std::make_shared<Texture>(img));


    std::string shader_name = "shader/particles.glsl";

    Saiga::ShaderCodeInjection depth_inj(GL_FRAGMENT_SHADER, "#define WRITE_DEPTH", 3);
    Saiga::ShaderCodeInjection shadow_inj(GL_GEOMETRY_SHADER, "#define SHADOW", 3);
    Saiga::ShaderCodeInjection shadow_inj2(GL_FRAGMENT_SHADER, "#define SHADOW", 3);

    particleShader      = Saiga::shaderLoader.load<Saiga::MVPShader>(shader_name, {depth_inj});
    particleDepthShader = Saiga::shaderLoader.load<Saiga::MVPShader>(shader_name, {depth_inj, shadow_inj, shadow_inj2});
    particleShaderFast  = Saiga::shaderLoader.load<Saiga::MVPShader>(shader_name, {});
    particleDepthShaderFast = Saiga::shaderLoader.load<Saiga::MVPShader>(shader_name, {shadow_inj, shadow_inj2});

    loadScenario();
    initParticles();
    initWalls();

    std::cout << "Agphys Initialized!" << std::endl;
}

Agphys::~Agphys()
{
    std::cout << "~Agphys" << std::endl;
}

void Agphys::loadScenario()
{
    if (scenario == 0) {
        return;
    } else if (scenario == 1) {
        numberParticles = 10000;
        distance = 0.99;
        xCount = 20;
        zCount = 20;
        corner = {-10, 15, -10};
        boxDim = {40, 40, 40};
    } else if (scenario == 2) {
        numberParticles = 1000000;
        distance = 0.99;
        xCount = 200;
        zCount = 200;
        corner = {-100, 0.5, -100};
        boxDim = {200, 100, 200};
    } else if (scenario == 7) {
        numberParticles = 2000 * 4 * 4 * 4;
        distance = 0.5;
        xCount = 10 * 4;
        zCount = 20 * 4;
        corner = {-30*2, 0, -10*2};
        boxDim = {60*2, 80*2, 20*2};
    } else if (scenario == 8) {
        numberParticles = 2000 * 4 * 4 * 4;
        distance = 0.5;
        xCount = 10 * 4;
        zCount = 20 * 4;
        corner = {-30*2, 0, -10*2};
        boxDim = {60*2, 80*2, 20*2};
    } else if (scenario == 9) {
        numberParticles = 20*20*20+40*40*40;
        distance = 0.5;
        xCount = 20;
        zCount = 20;
        corner = {-20, 0, -20};
        boxDim = {40, 80, 40};
    } else if (scenario == 10) {
        numberParticles = 50*50+10*10*10;
        distance = 0.5;
        xCount = 50;
        zCount = 50;
        corner = {-15, 15, -15};
        boxDim = {80, 80, 80};
    } else if (scenario == 11) {
        numberParticles = 2000 * 4 * 4 * 4;
        distance = 0.5;
        xCount = 10 * 4;
        zCount = 20 * 4;
        corner = {-30*2, 0, -10*2};
        boxDim = {60*2, 20*2, 20*2};
    } else if (scenario == 12) {
        numberParticles = 150 * 150 * 10;
        distance = 1;
        xCount = 150;
        zCount = 150;
        corner = {-79, 0, -79};
        boxDim = {160, 80, 160};
    }
    else if (scenario == 13) {
        numberParticles = 20 * 20 * 40 * 2;
        distance = 1;
        xCount = 20;
        zCount = 20;
        corner ={-20, 0, -10};
        boxDim ={40, 80, 30};
    } else if (scenario == 14) {
        // change to resize ocean
        float height = 4;
        fluidDim = {170, height, 170}; // width in x and z direction in particles
        trochoidal1Dim = {5, height, 5}; // width of one side in -x, +x, -z and +z direction in particles
        trochoidal2Dim = {60, 1, 60};

        // computation of dimensions  
        int trochoidal1Particles = (fluidDim[0] * trochoidal1Dim[0] * 2 + fluidDim[2] * trochoidal1Dim[2] * 2 + trochoidal1Dim[0] * trochoidal1Dim[2] * 4) * height;
        int trochoidal2Particles = ((trochoidal2Dim[0] + trochoidal1Dim[0]) * (trochoidal2Dim[2] + trochoidal1Dim[2]) - trochoidal1Dim[0] * trochoidal1Dim[2]) * 4; // calculation of corners
        trochoidal2Particles += trochoidal2Dim[0] * fluidDim[0] * 2 + trochoidal2Dim[2] * fluidDim[2] * 2;
        trochoidal2Particles *= trochoidal2Dim[1];

        int fluidParticles = fluidDim[0] * fluidDim[2] * height;
        int rbParticles = 22480;
        numberParticles = trochoidal1Particles + trochoidal2Particles + fluidParticles + rbParticles;
        
        int heightDifference = height - trochoidal2Dim[1];
        int firstLayer = (fluidDim[0] * trochoidal1Dim[0] * 2 + fluidDim[2] * trochoidal1Dim[2] * 2 + trochoidal1Dim[0] * trochoidal1Dim[2] * 4) * heightDifference + fluidDim[0] * fluidDim[2] * heightDifference;

        layers = ivec2(firstLayer + rbParticles, numberParticles);
        distance = 0.5;
        xCount = (int) (fluidDim[0] + trochoidal1Dim[0] * 2);
        zCount = (int) (fluidDim[2] + trochoidal1Dim[2] * 2);
        
        boxDim = {fluidDim[0] * distance, 80, fluidDim[2] * distance}; // in coordinates
        fluidDim = boxDim; // in coordinates
        trochoidal1Dim *= distance;
        trochoidal2Dim *= distance;
        corner = {-boxDim[0]/2, 0, -boxDim[2]/2}; // in coordinates
        corner -= vec3(trochoidal1Dim[0], 0, trochoidal1Dim[2]);
        
        // add border blend transition
        vec3 border = {2, 0, 2};
        boxDim += border;
    } else {
        numberParticles = 100000;
        distance = 0.99;
        xCount = 200;
        zCount = 200;
        corner = {-100, 0.5, -100};
        boxDim = {200, 100, 200};
    }
}


void Agphys::initParticles()
{
    destroyParticles();

    vec3 boxMin = - boxDim / 2;
    boxMin[1] = 0;
    particleSystem = std::make_shared<ParticleSystem>(numberParticles, boxMin, boxDim);

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

__host__ void checkError(cudaError_t err)
{
	if (err != cudaSuccess)
	{
		// Print a human readable error message
		std::cout << cudaGetErrorString(err) << std::endl;
		exit(-1);
	}
}

void Agphys::initWalls()
{
    std::vector<Saiga::Plane> walls(5);

    vec3 boxDim2  = boxDim / 2;

    walls = std::vector<Saiga::Plane>({
        Saiga::Plane({0,0,0},  {0,1,0} ),
        Saiga::Plane({boxDim2[0],0,0}, {-1,0,0}),
        Saiga::Plane({-boxDim2[0],0,0}, {1,0,0}),
        Saiga::Plane({0,0,boxDim2[2]}, {0,0,-1}),
        Saiga::Plane({0,0,-boxDim2[2]}, {0,0,1}),
    });

    auto size = sizeof (Saiga::Plane) * 5;

    Saiga::Plane *planes;

    checkError(cudaMalloc((void **)&planes, size));
    checkError(cudaMemcpy(planes, walls.data(), size, cudaMemcpyHostToDevice));

    // Initialize walls
    particleSystem->d_walls = make_ArrayView(planes, 5);
}

void Agphys::updateSingleStep(float dt)
{
    if (renderer->use_keyboard_input_in_3dview) camera.update(dt);

    sun->fitShadowToCamera(&camera);
    sun->fitNearPlaneToScene(AABB(vec3(-125, 0, -125), vec3(125, 50, 125)));

#ifdef SAIGA_USE_FFMPEG
    enc.update();
#endif

    //if (pause) return;

    map();
    float t;
    {
        Saiga::CUDA::ScopedTimer tim(t);
        particleSystem->update(dt);
    }
    physicsGraph.addTime(t);
    unmap();
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
    // controls
    updateControlsAndCamera(dt);
    float t;
    {
        Saiga::CUDA::ScopedTimer tim(t);
        
        // profiling
        #undef CUDA_PROFILING
        //#define CUDA_PROFILING
        #ifdef CUDA_PROFILING
        if (steps == 300)
            cudaProfilerStart();
        #endif

        particleSystem->update(dt);

        #ifdef CUDA_PROFILING
        if (steps++ == 300) {
            cudaProfilerStop();
            window->close();
        }
        #endif
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
        skybox->render(camera);
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
    particleSystem->reset(xCount, zCount, corner, distance, randInitMul, scenario, fluidDim, trochoidal1Dim, trochoidal2Dim, layers);
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


void Agphys::updateControlsAndCamera(float delta)
{
    int shoot_key = GLFW_KEY_C;
    int ball1_key = GLFW_KEY_1;
    int ball2_key = GLFW_KEY_2;
    std::vector<int> cameraKeyboardmap0 = {GLFW_KEY_W, GLFW_KEY_S, GLFW_KEY_A, GLFW_KEY_D,
                             //            GLFW_KEY_UP,
                             //            GLFW_KEY_DOWN,
                             //            GLFW_KEY_LEFT,
                             //            GLFW_KEY_RIGHT,
                             GLFW_KEY_LEFT_SHIFT, GLFW_KEY_SPACE, GLFW_KEY_LEFT_ALT};
    std::vector<int> cameraKeyboardmap1 = {GLFW_KEY_T, GLFW_KEY_G, GLFW_KEY_F, GLFW_KEY_H,
                             //            GLFW_KEY_UP,
                             //            GLFW_KEY_DOWN,
                             //            GLFW_KEY_LEFT,
                             //            GLFW_KEY_RIGHT,
                             GLFW_KEY_LEFT_SHIFT, GLFW_KEY_SPACE, GLFW_KEY_LEFT_ALT};
    std::vector<int> keyboardmap0 = {GLFW_KEY_W, GLFW_KEY_S, GLFW_KEY_A, GLFW_KEY_D};
    std::vector<int> keyboardmap1 = {GLFW_KEY_T, GLFW_KEY_G, GLFW_KEY_F, GLFW_KEY_H};
    std::vector<int> keyboardmap = keyboardmap1;
    camera.keyboardmap = cameraKeyboardmap0;

    if (gameMode) {
        shoot_key = GLFW_KEY_SPACE;
        keyboardmap = keyboardmap0;
        camera.keyboardmap = cameraKeyboardmap1;
    }

    particleSystem->control_forward = keyboard.getMappedKeyState(0, keyboardmap) - keyboard.getMappedKeyState(1, keyboardmap);
    particleSystem->control_rotate = keyboard.getMappedKeyState(2, keyboardmap) - keyboard.getMappedKeyState(3, keyboardmap);
    particleSystem->control_cannonball = keyboard.getMappedKeyState(0, {shoot_key});

    if (keyboard.getMappedKeyState(0, {ball1_key})) {
        particleSystem->regular_ball = true;
    } 
    if (keyboard.getMappedKeyState(0, {ball2_key})) {
        particleSystem->regular_ball = false;
    }

    if (camera_follow) {
        vec3 position = particleSystem->ship_position;

        // from camera.cpp PixelRay()
        vec3 p = camera.ViewToWorld(camera.NormalizedToView({0, 0, 1}));
        vec3 camera_position = camera.getPosition();
        vec3 camera_direction = (p - camera_position).normalized();
        particleSystem->camera_direction = camera_direction;
            
        //vec3 new_camera_position = position + vec3{10, 20, 0};
        vec3 camera_offset = {0, 5, 0};
        float camera_distance = 20;
        vec3 new_camera_position = position + camera_offset - camera_direction * camera_distance;

        // interpolation smoothing
        new_camera_position = new_camera_position * 0.5 + camera_position * 0.5;

        //vec3 up = camera_direction.cross(vec3{0, 1, 0}).cross(camera_direction).normalized();
        vec3 up = {0, 1, 0};

        //camera.setView(camera_position, camera_position + camera_direction, up); // default (doesnt change anything)
        camera.setView(new_camera_position, new_camera_position + camera_direction, up);
    }
}

void Agphys::toggleGameMode() {
    gameMode = !gameMode;
    camera_follow = !camera_follow;
    shouldRenderGUI = !shouldRenderGUI;
    showSaigaGui = !showSaigaGui;
    renderer->timer->Enable(false);
    renderer->window->setShowImgui(false);
    editor_gui.enabled = false;
    //main_menu.EraseItem("Saiga", "Log");
}