#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/shader/shaderLoader.h"

#include "agphys.h"


void Agphys::renderGUI()
{
    if (!shouldRenderGUI)
    {
        return;
    }



    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(400, 400), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Agphys")) {


        ImGui::InputInt("particleCount", &numberParticles, 0, 10000);
        ImGui::InputFloat("distance", &distance, 0.01, 0.1);
        ImGui::InputInt("xCount", &xCount);
        ImGui::InputInt("zCount", &zCount);
        ImGui::InputFloat3("corner", &corner[0]);
        ImGui::InputFloat3("boxDim", &boxDim[0]);
        ImGui::InputFloat("rand", &randInitMul);

        ImGui::Combo("scenario", &scenario, scenarios, std::size(scenarios));
        if (ImGui::Button("Create Particle System"))
        {
            destroyParticles();
            loadScenario();
            initParticles();
            initWalls();
        }

        ImGui::Checkbox("renderParticles", &renderParticles);
        ImGui::Checkbox("renderShadows", &renderShadows);
        ImGui::Checkbox("showSaigaGui", &showSaigaGui);

        if (ImGui::Button("Pause (H)"))
        {
            pause = !pause;
        }

        ImGui::InputFloat("stepsize", &stepsize, 0.001, 0.01);
        if (ImGui::Button("step (J)"))
        {
            updateSingleStep(stepsize);
        }

        physicsGraph.renderImGui();
    }

    ImGui::End();

#ifdef SAIGA_USE_FFMPEG
    if (show_video_encoding)
    {
        int h = 600;
        ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - 400, h), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_Once);
        if (ImGui::Begin("Video Encoding", &show_video_encoding))
        {
            enc.renderGUI();
        }
        ImGui::End();
    }
#endif

    particleSystem->renderGUI();
}

void Agphys::map()
{
    interop.map();
    void* ptr1 = interop.getDevicePtr();

    particleSystem->setDevicePtr(ptr1);
}

void Agphys::unmap()
{
    interop.unmap();
}

void Agphys::keyPressed(int key, int scancode, int mods)
{
    switch (key)
    {
        case GLFW_KEY_F11:
#ifdef SAIGA_USE_FFMPEG
            if (enc.isEncoding())
            {
                enc.stopRecording();
            }
            else
            {
                enc.startRecording();
            }
#endif

            break;

        case GLFW_KEY_ESCAPE:
            window->close();
            break;
        case GLFW_KEY_R:
            resetParticles();
            break;
        case GLFW_KEY_H:
            pause = !pause;
            break;
        case GLFW_KEY_J:
            updateSingleStep(stepsize);
            break;
        case GLFW_KEY_F12:
            break;
        case GLFW_KEY_F3:
            shouldRenderGUI = !shouldRenderGUI;

        default:
            break;
    }
}


void Agphys::mousePressed(int key, int x, int y)
{
    if (!renderer->use_mouse_input_in_3dview) return;
    ivec2 global_pixel = ivec2(x, y);
    ivec2 local_pixel = renderer->WindowCoordinatesToViewport(global_pixel);

    if (key == GLFW_MOUSE_BUTTON_RIGHT)
    {
        // this gives a ray going through the camera position and the given pixel in world space
        auto ray =
            camera.PixelRay(local_pixel.cast<float>(), renderer->viewport_size.x(), renderer->viewport_size.y(), true);
        particleSystem->ray(ray);
    }
}
