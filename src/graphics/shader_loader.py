"""
Shader loader utility - Loads and compiles OpenGL shaders.
"""

import logging
from pathlib import Path
import moderngl

logger = logging.getLogger(__name__)

def load_shader_program(ctx: moderngl.Context, vertex_path: str, fragment_path: str) -> moderngl.Program:
    """Load and compile a shader program."""

    try:
        # Load vertex shader
        vertex_source = _load_shader_source(vertex_path)

        # Load fragment shader  
        fragment_source = _load_shader_source(fragment_path)

        # Create program
        program = ctx.program(
            vertex_shader=vertex_source,
            fragment_shader=fragment_source
        )

        logger.info(f"Shader program loaded: {vertex_path}, {fragment_path}")
        return program

    except Exception as e:
        logger.error(f"Failed to load shader program: {e}")
        raise

def _load_shader_source(shader_path: str) -> str:
    """Load shader source from file."""
    path = Path(shader_path)

    if not path.exists():
        # Return default shader if file doesn't exist
        logger.warning(f"Shader file not found: {shader_path}, using default")
        return _get_default_shader(path.suffix)

    with open(path, 'r') as f:
        return f.read()

def _get_default_shader(shader_type: str) -> str:
    """Get default shader source."""

    if shader_type == '.vert':
        return """
        #version 330 core
        in vec3 position;
        in vec3 normal;
        in vec2 texcoord;

        uniform mat4 mvp_matrix;

        out vec3 v_normal;
        out vec2 v_texcoord;

        void main() {
            v_normal = normal;
            v_texcoord = texcoord;
            gl_Position = mvp_matrix * vec4(position, 1.0);
        }
        """

    elif shader_type == '.frag':
        return """
        #version 330 core
        in vec3 v_normal;
        in vec2 v_texcoord;

        uniform sampler2D main_texture;
        uniform vec4 base_color;

        out vec4 fragColor;

        void main() {
            vec4 tex_color = texture(main_texture, v_texcoord);
            fragColor = tex_color * base_color;
        }
        """

    else:
        raise ValueError(f"Unknown shader type: {shader_type}")
