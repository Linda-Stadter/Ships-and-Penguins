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

        ImGui::InputInt("solver_iterations", &solver_iterations);
        ImGui::Checkbox("use_calculated_relax_p", &use_calculated_relax_p);
        ImGui::InputFloat("relax", &relax_p);
        ImGui::InputFloat("damp", &damp_v);

        ImGui::InputFloat("particle_radius_rest_density", &particle_radius_rest_density);
        ImGui::InputFloat("particleRadiusWater", &particleRadiusWater);
        ImGui::InputFloat("particleRadiusCloth", &particleRadiusCloth);

        ImGui::Checkbox("test bool", &test_bool);
        ImGui::InputFloat("test float", &test_float);

        ImGui::InputFloat("kinetic friction", &mu_k);
        ImGui::InputFloat("static friction", &mu_s);
        ImGui::InputFloat("friction", &mu_f);

        ImGui::InputFloat("h", &h);
        ImGui::InputFloat("epsilon_spiky", &epsilon_spiky);
        ImGui::InputFloat("omega_lambda_relax", &omega_lambda_relax);

        ImGui::InputFloat("artificial_pressure_k", &artificial_pressure_k);
        ImGui::InputInt("artificial_pressure_n", &artificial_pressure_n);
        ImGui::InputFloat("delta_q", &delta_q);

        ImGui::InputFloat("c_viscosity", &c_viscosity);
        ImGui::InputFloat("epsilon_vorticity", &epsilon_vorticity);

        ImGui::InputFloat3("wind_direction", &wind_direction[0]);
        ImGui::InputFloat("wind_speed", &wind_speed);

        ImGui::Combo("physics", &physics_mode, physics, std::size(physics));
        ImGui::Combo("mouse action", &action_mode, actions, std::size(actions));
        ImGui::ColorEdit4("color", &color[0]);
        ImGui::InputInt("explosion_force", &explosion_force);
        ImGui::InputInt("split_count", &split_count);

        ImGui::Separator();
    }
    ImGui::End();
}
