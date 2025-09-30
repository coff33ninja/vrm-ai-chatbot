#version 330 core

// Vertex attributes
in vec3 position;
in vec3 normal;
in vec2 texcoord;
in ivec4 joints;
in vec4 weights;

// Uniforms
uniform mat4 mvp_matrix;
uniform mat4 model_matrix;
uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform mat4 bone_matrices[64];
uniform bool use_skinning;

// Outputs to fragment shader
out vec3 world_pos;
out vec3 world_normal;
out vec2 uv;
out vec3 view_pos;

void main() {
    vec4 pos = vec4(position, 1.0);
    vec3 norm = normal;
    
    // Apply skinning if enabled
    if (use_skinning) {
        mat4 skin_matrix = 
            weights.x * bone_matrices[joints.x] +
            weights.y * bone_matrices[joints.y] +
            weights.z * bone_matrices[joints.z] +
            weights.w * bone_matrices[joints.w];
        
        pos = skin_matrix * pos;
        norm = mat3(skin_matrix) * norm;
    }
    
    // Transform to world space
    world_pos = (model_matrix * pos).xyz;
    world_normal = normalize(mat3(model_matrix) * norm);
    view_pos = (view_matrix * vec4(world_pos, 1.0)).xyz;
    uv = texcoord;
    
    // Final position
    gl_Position = mvp_matrix * pos;
}
