#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/random.h"
#include "saiga/core/util/assert.h"

#include "particleSystem.h"

#include <thrust/extrema.h>
#include <thrust/sort.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

void ParticleSystem::renderGUI()
{

    if(ImGui::Begin("ParticleSystem"))
    {
        ImGui::InputFloat3("gravity", &gravity[0]);
        ImGui::InputFloat("elast", &elast_const);
        ImGui::InputFloat("spring", &spring_const);
        ImGui::InputFloat("frict", &frict_const);
    }
    ImGui::End();
}
