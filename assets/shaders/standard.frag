#version 330 core

in vec3 world_normal;
in vec2 uv;

uniform sampler2D main_texture;
uniform vec4 base_color;
uniform vec3 light_direction;

out vec4 fragColor;

void main() {
    vec4 tex_color = texture(main_texture, uv);
    vec3 light_dir = normalize(-light_direction);
    float ndotl = max(0.0, dot(world_normal, light_dir));
    
    vec3 color = tex_color.rgb * base_color.rgb * (0.3 + 0.7 * ndotl);
    fragColor = vec4(color, tex_color.a * base_color.a);
}
