//
// LICENSE:
//
// Copyright (c) 2016 -- 2017 Fabio Pellacini
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#define YGL_OPENGL 1
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "../yocto/yocto_gl.h"
#include "../yocto/ext/imgui/imgui.h"
using namespace ygl;

// ---------------------------------------------------------------------------
// SCENE (OBJ or GLTF) AND APPLICATION PARAMETERS
// ---------------------------------------------------------------------------

struct shade_state {
    gl_stdsurface_program prog = {};
};

struct app_state {
    string filename;
    string imfilename;

    // ui
    bool scene_updated = false;
    bool show_widgets = true;
    shade_state* shstate = nullptr;
    vec2i framebuffer_size;
    float fps = 1.0;
    float last_frame;

    //Fragment shader variables
    int AA = 1;
    int max_raymarching_steps = 128;
    float max_distance = 7000;
    float step_size = .8;
    vec2f mouse = zero2f;
    vec3f mov = zero3f;
    float camera_distance = 250;
    float ka = 0.05, kd = 0.2;
    float softshadows = 7.0;
    bool shadows = true;
    bool water_refl = true;
    bool clouds = false;
    bool gamma = false;
    bool lens_flare = false;
    bool hq_water = true;
    bool lava = false;
    int fbm_octaves = 11;
    int sha_octaves = 6;
    float sha_stepsize = 3.0;
    float fog = 0.2;
    float sun_dispersion = 0.5;
    float wavegain = 0.3;

    float water_level = 130;
    float elevate = 2.0;
    float large = 1.5;
    float crater_width = 0.03;

    float seed = 104.0;


    float A = 0.03, B = 1, C = 1, D = 1, E = 1;


    ~app_state() {
        if (shstate) delete shstate;
    }
};


void loadShaderSource(const string& filename, string& out) {
    std::ifstream file;
    file.open(filename.c_str());
    if (!file) throw runtime_error(string("cannot open shader file"));
    std::stringstream stream;
    stream << file.rdbuf();
    file.close();
    out = stream.str();
}

inline gl_stdsurface_program make_my_program() {
    string myvert = R"(
        #version 330 core
        const vec2 quadvertices[4] = vec2[4]( vec2( -1.0, -1.0), vec2( 1.0, -1.0), vec2( -1.0, 1.0), vec2( 1.0, 1.0));
        void main()
        {
        gl_Position = vec4(quadvertices[gl_VertexID], 0.0, 1.0);
        }
    )";

    string myfrag;
    loadShaderSource("fragment_shader.frag", myfrag);

    assert(gl_check_error());
    auto prog = gl_stdsurface_program();
    prog._prog = make_program(myvert, "#version 330 core\n" + myfrag);
    assert(gl_check_error());
    return prog;
}

inline void pass_to_shader(app_state* app, string name, float value) {
    int uniform_name = glGetUniformLocation(app->shstate->prog._prog._pid,name.c_str());
    glUniform1f(uniform_name, value);
}

inline void pass_to_shader(app_state* app, string name, int value) {
    int uniform_name = glGetUniformLocation(app->shstate->prog._prog._pid,name.c_str());
    glUniform1i(uniform_name, value);
}

inline void pass_to_shader(app_state* app, string name, float value1, float value2) {
    int uniform_name = glGetUniformLocation(app->shstate->prog._prog._pid,name.c_str());
    glUniform2f(uniform_name, value1, value2);
}

inline void pass_to_shader(app_state* app, string name, vec3f value) {
    int uniform_name = glGetUniformLocation(app->shstate->prog._prog._pid,name.c_str());
    glUniform3f(uniform_name, value.x, value.y, value.z);
}


// Display a scene
inline void shade_scene(app_state* app) {
    if (!is_program_valid(app->shstate->prog)) app->shstate->prog = make_my_program();

    bind_program(app->shstate->prog._prog);

    //passing parameters to the fragment shader
    pass_to_shader(app, "resolution", app->framebuffer_size.x, app->framebuffer_size.y);
    pass_to_shader(app, "mouse", app->mouse.x, app->mouse.y);
    pass_to_shader(app, "mov", app->mov);
    pass_to_shader(app, "max_raymarching_steps", app->max_raymarching_steps);
    pass_to_shader(app, "max_distance", app->max_distance);
    pass_to_shader(app, "step_size", app->step_size);
    pass_to_shader(app, "camera_distance", app->camera_distance);
    pass_to_shader(app, "ka", app->ka);
    pass_to_shader(app, "kd", app->kd);
    pass_to_shader(app, "softshadows", app->softshadows);
    pass_to_shader(app, "shadows", app->shadows);
    pass_to_shader(app, "water_refl", app->water_refl);
    pass_to_shader(app, "clouds", app->clouds);
    pass_to_shader(app, "gamma", app->gamma);
    pass_to_shader(app, "fbm_octaves", app->fbm_octaves);
    pass_to_shader(app, "sha_octaves", app->sha_octaves);
    pass_to_shader(app, "sha_stepsize", app->sha_stepsize);
    pass_to_shader(app, "fog", app->fog);
    pass_to_shader(app, "sun_dispersion", app->sun_dispersion);
    pass_to_shader(app, "AA", app->AA);
    pass_to_shader(app, "lens_flare", app->lens_flare);
    pass_to_shader(app, "elevate", app->elevate);
    pass_to_shader(app, "large", app->large);
    pass_to_shader(app, "water_level", app->water_level);
    pass_to_shader(app, "hq_water", app->hq_water);
    pass_to_shader(app, "seed", app->seed);
    pass_to_shader(app, "wavegain", app->wavegain);
    pass_to_shader(app, "lava", app->lava);
    pass_to_shader(app, "crater_width", app->crater_width);

    double real_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    int passed = real_time - app->last_frame;
    if (passed >= 5) { //to avoid counting the same frame together (glfw problem?)
        app->fps = app->fps * 0.90f + (1000.0f / passed) * 0.10f;
        app->last_frame = real_time;
    }
    float time = fmod(real_time, 100000) / 1000.0f;
    int uniform_time = glGetUniformLocation(app->shstate->prog._prog._pid,"time");
    glUniform1f(uniform_time, time);

    //we need to pass just 4 vertices to the vertex shader
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    unbind_program(app->shstate->prog._prog);
}

// draw with shading
inline void draw(gl_window* win) {
    auto app = (app_state*)get_user_pointer(win);

    auto window_size = get_window_size(win);
    app->framebuffer_size = get_framebuffer_size(win);
    gl_set_viewport(app->framebuffer_size);
    auto aspect = (float)window_size.x / (float)window_size.y;

    gl_clear_buffers();
    shade_scene(app);

    if (app->show_widgets && begin_widgets(win, "yview")) {
        draw_separator_widget(win);
        draw_label_widget(win, std::to_string(app->fps), "FPS:");
        draw_value_widget(win, "Ray steps", app->max_raymarching_steps, 1, 200, 1);
        draw_value_widget(win, "Max distance", app->max_distance, 0, 10000, 1);
        draw_value_widget(win, "step size", app->step_size, 0, 1, 1);
        draw_value_widget(win, "AA", app->AA, 1, 3, 1);
        draw_separator_widget(win);
        draw_value_widget(win, "ka", app->ka, 0, 1, 1);
        draw_value_widget(win, "kd", app->kd, 0, 1, 1);
        draw_separator_widget(win);
        draw_value_widget(win, "water reflections", app->water_refl);
        draw_value_widget(win, "shadows", app->shadows);
        ImGui::SameLine();
        draw_value_widget(win, "clouds", app->clouds);
        ImGui::SameLine();
        draw_value_widget(win, "gamma", app->gamma);
        if (app->shadows) draw_value_widget(win, "softshadows", app->softshadows, 0, 60, 1);
        if (app->shadows) draw_value_widget(win, "sha_octaves", app->sha_octaves, 0, 14, 1);
        if (app->shadows) draw_value_widget(win, "sha_stepsize", app->sha_stepsize, 0, 50, 1);
        draw_value_widget(win, "fog", app->fog, 0, 1, 1);
        draw_value_widget(win, "sun dispersion", app->sun_dispersion, 0, 1, 1);
        draw_value_widget(win, "lens flare", app->lens_flare);
        ImGui::SameLine();
        draw_value_widget(win, "hq water", app->hq_water);
        ImGui::SameLine();
        draw_value_widget(win, "lava", app->lava);
        draw_separator_widget(win);
        draw_value_widget(win, "fbm_octaves", app->fbm_octaves, 0, 14, 1);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Number of noise octaves to generate terrain");
        draw_value_widget(win, "elevate", app->elevate, 0, 10, 1);
        draw_value_widget(win, "volcano_width", app->large, 0, 3, 1);
        draw_value_widget(win, "crater_width", app->crater_width, 0, 1, 1);
        draw_value_widget(win, "water level", app->water_level, -500, 500, 1);
        draw_value_widget(win, "rough water", app->wavegain, 0, 1, 1);
        if(draw_button_widget(win, "Randomize")) {
            double real_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
            app->seed = fmod(real_time , 5000);
        }
        end_widgets(win);
    }

    swap_buffers(win);
}

bool update(app_state* st) {
    return false;
}

inline void run_ui(app_state* app, int w, int h, const string& title) {
    // window
    auto win = make_window(w, h, title, app);
    set_window_callbacks(win, nullptr, nullptr, draw);

    // window values
    int mouse_button = 0;
    vec2f mouse_pos, mouse_last;

    // load textures and vbos
    app->shstate = new shade_state();
    app->shstate->prog = make_my_program();

    // init widget
    init_widgets(win);

    // loop
    while (!should_close(win)) {
        mouse_last = mouse_pos;
        mouse_pos = get_mouse_posf(win);
        mouse_button = get_mouse_button(win);

        set_window_title(win, ("yview | " + app->filename));

        // handle mouse and keyboard for navigation
        if (mouse_button && !get_widget_active(win)) {
            auto dolly = 0.0f;
            auto pan = zero2f;
            auto rotate = zero2f;
            switch (mouse_button) {
                case 1: app->mouse = mouse_pos; break;
                case 2: app->camera_distance += (mouse_pos.x - mouse_last.x) / 10.0f;
                    break;
                case 3: pan = (mouse_pos - mouse_last) / 100.0f; break;
                default: break;
            }
            app->scene_updated = true;
        }

        // handle keytboard for navigation
        if (!get_widget_active(win)) {
            if (get_key(win, 'a')) app->mov.x -= 0.1;
            if (get_key(win, 'd')) app->mov.x += 0.1;
            if (get_key(win, 's')) app->mov.z -= 0.1;
            if (get_key(win, 'w')) app->mov.z += 0.1;
            if (get_key(win, 'e')) app->mov.y += 0.1;
            if (get_key(win, 'q')) app->mov.y -= 0.1;
            if (get_key(win, GLFW_KEY_ESCAPE)) exit(0);
            if (get_key(win, GLFW_KEY_SPACE)) app->show_widgets = !  (app->show_widgets);
        }

        // draw
        draw(win);

        // update
        update(app);

        // event hadling
        poll_events(win);
    }

    clear_window(win);
}


int main(int argc, char* argv[]) {
    // create empty scene
    auto app = new app_state();

    // run ui
    auto width = 800;
    auto height = 600;
    run_ui(app, width, height, "yview");

    // clear
    delete app;

    // done
    return 0;
}
