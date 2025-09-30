#version 330 core

in vec3 position;
in vec3 normal;
in vec2 texcoord;

uniform mat4 mvp_matrix;
uniform mat4 model_matrix;

out vec3 world_normal;
out vec2 uv;

void main() {
    world_normal = normalize(mat3(model_matrix) * normal);
    uv = texcoord;
    gl_Position = mvp_matrix * vec4(position, 1.0);
}
