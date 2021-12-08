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

        ImGui::InputInt("solverIterations", &solverIterations);
        ImGui::Checkbox("useCalculatedRelaxP", &useCalculatedRelaxP);
        ImGui::InputFloat("relax", &relaxP);
        ImGui::InputFloat("damp", &dampV);
        ImGui::Checkbox("Jacobi Solver (or Gauss-Seidel)", &jacobi);

        ImGui::Combo("physics", &physicsMode, physics, std::size(physics));
        ImGui::Checkbox("use SDF", &useSDF); // 4.4
        ImGui::Combo("mouse action", &actionMode, actions, std::size(actions));
        ImGui::ColorEdit4("color", &color[0]);
        ImGui::InputInt("explosionForce", &explosionForce);
        ImGui::InputInt("splitCount", &splitCount);

        ImGui::Separator();
        ImGui::Combo("hashing", &hashFunction, hashes, std::size(hashes));
    }
    ImGui::End();
}
