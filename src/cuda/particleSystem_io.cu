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

        ImGui::InputFloat("cloth_break_distance", &cloth_break_distance);

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

        ImGui::InputFloat("cannonball_speed", &cannonball_speed);

        ImGui::Checkbox("debug_shooting", &debug_shooting);

        ImGui::InputFloat("wave_length", &wave_number);
        ImGui::InputFloat("phase_speed", &phase_speed);
        ImGui::InputFloat("steepness", &steepness);

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

void ParticleSystem::renderIngameGUI()
{
    // game ui
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(500, 1000*2), ImGuiCond_Always);
    if(ImGui::Begin("HUD", NULL, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground))
    {
        ImGui::SetWindowFontScale(2.0);

        // Time
        std::string time = "Time: ";
        int total_seconds = max_time - passed_time;
        int minutes = total_seconds / 60;
        int seconds = total_seconds % 60;
        std::string str_minutes = std::to_string(minutes);
        std::string str_seconds = std::to_string(seconds);
        if (minutes < 10)
            str_minutes = "0" + str_minutes;
        if (seconds < 10)
            str_seconds = "0" + str_seconds;
        time.append(str_minutes + ":" + str_seconds);
        ImColor time_color = ImColor(1.f, 1.f, 1.f, 1.f);
        if (total_seconds <= 10.0)
            time_color = ImColor(1.f, .1f, .1f, 1.f); // red
        ImGui::TextColored(time_color, time.c_str());

        // ammo
        std::string fish_count = "Fish left: ";
        fish_count.append(std::to_string(ammo_left));
        std::string fish_type = "Current Fish: ";
        if (regular_ball) {
            fish_type.append("Pufferfish");
        } else {
            fish_type.append("Swordfish");
        }
        ImColor fish_count_color = ImColor(1.f, 1.f, 1.f, 1.f);
        if (ammo_left <= 1)
            fish_count_color = ImColor(1.f, .1f, .1f, 1.f); // red
        ImGui::TextColored(fish_count_color, fish_count.c_str());
        ImGui::TextColored(ImColor(1.f, 1.f, 1.f, 1.f), fish_type.c_str());

        // reload
        int reload_bar_length = 10;
        int reload_bar_progress = (float)cannon_timer / (float)cannon_timer_reset * reload_bar_length;
        std::string reload = "reload: [";
        for (int i = 0; i < reload_bar_progress; i++) {
            reload.append("#");
        }
        std::string reload2 = "";
        for (int i = reload_bar_progress; i < reload_bar_length; i++) {
            reload2.append("#");
        }
        reload2.append("]");

        ImColor reload_color = ImColor(1.f, 1.f, 1.f, 1.f); // white
        ImColor reload_color2 = ImColor(1.f, .1f, .1f, 1.f); // red
        if (reload_bar_progress == reload_bar_length) {
            reload_color = ImColor(.1f, 1.f, .1f, 1.f); // green
            reload_color2 = reload_color;
        }
        ImGui::TextColored(reload_color, reload.c_str());
        ImGui::SameLine();
        ImGui::TextColored(reload_color2, reload2.c_str());

        // Score
        std::string score_string = "Score: ";
        score_string.append(std::to_string(score));
        ImGui::TextColored(ImColor(1.f, 1.f, 1.f, 1.f), score_string.c_str());
    }
    ImGui::End();

    if (game_over) {
        ImGui::SetNextWindowPos(ImVec2(1920/2, 1080/2), ImGuiCond_Always);
        //ImGui::SetNextWindowSize(ImVec2(1920, 1080), ImGuiCond_Always);
        if(ImGui::Begin("game over", NULL, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground))
        {
            ImGui::SetWindowFontScale(2.0);
            std::string score_string = "Your Score: ";
            score_string.append(std::to_string(score));
            
            ImGui::TextColored(ImColor(1.f, 0.f, 1.f, 1.f), score_string.c_str());
        }
        ImGui::End();
    } else {
        ImGui::SetNextWindowPos(ImVec2(1920/2, 1080/2), ImGuiCond_Always);
        //ImGui::SetNextWindowSize(ImVec2(1920, 1080), ImGuiCond_Always);
        if(ImGui::Begin("crosshair", NULL, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground))
        {
            ImGui::SetWindowFontScale(2.0);
            ImGui::TextColored(ImColor(1.f, 0.f, 1.f, 1.f), "+");
        }
        ImGui::End();
    }
    
}
