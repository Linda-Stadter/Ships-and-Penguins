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

        ImGui::InputFloat("particleRadiusRestDensity", &particleRadiusRestDensity);
        ImGui::InputFloat("particleRadiusWater", &particleRadiusWater);
        ImGui::InputFloat("particleRadiusCloth", &particleRadiusCloth);

        ImGui::Checkbox("test bool", &testBool);
        ImGui::InputFloat("test float", &testFloat);

        ImGui::InputFloat("h", &h);
        ImGui::InputFloat("epsilon_spiky", &epsilon_spiky);
        ImGui::InputFloat("omega_lambda_relax", &omega_lambda_relax);

        ImGui::InputFloat("artificial_pressure_k", &artificial_pressure_k);
        ImGui::InputInt("artificial_pressure_n", &artificial_pressure_n);
        ImGui::InputFloat("delta_q", &delta_q);

        ImGui::InputFloat("c_viscosity", &c_viscosity);
        ImGui::InputFloat("epsilon_vorticity", &epsilon_vorticity);

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
