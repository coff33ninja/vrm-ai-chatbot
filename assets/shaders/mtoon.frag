#version 330 core

// Inputs from vertex shader
in vec3 world_pos;
in vec3 world_normal;
in vec2 uv;
in vec3 view_pos;

// Textures
uniform sampler2D main_texture;
uniform sampler2D shade_texture;
uniform sampler2D emission_texture;
uniform sampler2D normal_texture;
uniform sampler2D rim_texture;

// MToon material properties
uniform vec4 base_color;
uniform vec4 shade_color;
uniform vec3 emission_color;
uniform float shade_shift;
uniform float shade_toony;
uniform float light_color_attenuation;
uniform float alpha_cutoff;
uniform vec4 rim_color;
uniform float rim_lighting_mix;
uniform float rim_fresnel_power;
uniform float rim_lift;

// Lighting
uniform vec3 light_direction;
uniform vec3 light_color;
uniform vec3 camera_position;

// Output
out vec4 fragColor;

void main() {
    // Sample textures
    vec4 main_tex = texture(main_texture, uv);
    vec4 shade_tex = texture(shade_texture, uv);
    vec3 emission_tex = texture(emission_texture, uv).rgb;
    vec3 normal_tex = texture(normal_texture, uv).rgb * 2.0 - 1.0;
    vec4 rim_tex = texture(rim_texture, uv);
    
    // Calculate normal
    vec3 normal = normalize(world_normal + normal_tex * 0.1);
    vec3 light_dir = normalize(-light_direction);
    vec3 view_dir = normalize(camera_position - world_pos);
    
    // MToon toon shading calculation
    float ndotl = dot(normal, light_dir);
    float shade_intensity = 1.0 - shade_shift;
    float toon_shadow = smoothstep(
        shade_intensity - shade_toony * 0.5,
        shade_intensity + shade_toony * 0.5,
        ndotl * 0.5 + 0.5
    );
    
    // Base lighting
    vec3 lit_color = main_tex.rgb * base_color.rgb;
    vec3 shade_lit_color = shade_tex.rgb * shade_color.rgb;
    vec3 toon_color = mix(shade_lit_color, lit_color, toon_shadow);
    
    // Rim lighting
    float rim_fresnel = 1.0 - max(0.0, dot(normal, view_dir));
    rim_fresnel = pow(rim_fresnel, rim_fresnel_power);
    rim_fresnel = rim_fresnel + rim_lift;
    
    vec3 rim_lighting = rim_color.rgb * rim_tex.rgb * rim_fresnel;
    toon_color = mix(toon_color, toon_color + rim_lighting, rim_lighting_mix);
    
    // Add emission
    toon_color += emission_tex * emission_color;
    
    // Light color attenuation
    toon_color = mix(toon_color, toon_color * light_color, light_color_attenuation);
    
    // Final alpha
    float alpha = main_tex.a * base_color.a;
    
    // Alpha cutoff
    if (alpha < alpha_cutoff) {
        discard;
    }
    
    fragColor = vec4(toot_color, alpha);
}
