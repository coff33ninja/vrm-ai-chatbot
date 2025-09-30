"""
VRM Model Renderer - Handles 3D rendering of VRM characters with OpenGL.
Supports VRM format with MToon shaders, bone animation, and facial expressions.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    import moderngl
    from OpenGL.GL import *  # noqa: F401
    _MODERNGL_AVAILABLE = True
except Exception:
    moderngl = None  # type: ignore
    _MODERNGL_AVAILABLE = False
import pyrr
from pygltflib import GLTF2

from ..models.character import Character
#from ..utils.math_utils import *
from ..utils.shader_loader import load_shader_program

logger = logging.getLogger(__name__)

class VRMRenderer:
    """Advanced VRM character renderer with real-time animation."""
    
    def __init__(self, window, target_fps: int = 60, vsync: bool = True, antialiasing: bool = True):
        self.window = window
        self.target_fps = target_fps
        self.vsync = vsync
        self.antialiasing = antialiasing
        
        # Rendering context
        self.ctx: Optional[moderngl.Context] = None
        self.fbo: Optional[moderngl.Framebuffer] = None
        
        # VRM model data
        self.character: Optional[Character] = None
        self.meshes: List[Dict[str, Any]] = []
        self.materials: Dict[str, Any] = {}
        self.textures: Dict[str, moderngl.Texture] = {}
        self.animations: Dict[str, Any] = {}
        
        # Shaders
        self.mtoon_program: Optional[moderngl.Program] = None
        self.standard_program: Optional[moderngl.Program] = None
        
        # Camera and matrices
        self.view_matrix = pyrr.matrix44.create_look_at(
            eye=[0, 1.6, 3], 
            target=[0, 1.6, 0], 
            up=[0, 1, 0]
        )
        self.projection_matrix = pyrr.matrix44.create_perspective_projection_matrix(
            45.0, 1.0, 0.1, 100.0
        )
        self.model_matrix = pyrr.matrix44.create_identity()
        
        # Animation state
        self.current_frame = 0
        self.animation_time = 0.0
        self.bone_matrices: Optional[np.ndarray] = None
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = 0
        
        logger.info("VRM Renderer initialized")
    
    async def initialize(self):
        """Initialize the OpenGL context and rendering resources."""
        # Try to initialize a real ModernGL context. If unavailable, fall
        # back to a headless no-op renderer to allow the rest of the app to
        # initialize (useful for development or headless environments).
        if not _MODERNGL_AVAILABLE:
            logger.error("ModernGL/OpenGL not available. Rendering will run in headless mode. "
                         "To enable 3D rendering, install ModernGL and ensure your system supports OpenGL.")
            # Replace methods with headless implementations but preserve the window
            headless = HeadlessRenderer(self.target_fps, window=self.window)
            self.__class__ = HeadlessRendererProxy
            # preserve any attributes that should remain (like window)
            new_dict = headless.__dict__
            try:
                new_dict['window'] = self.window
            except Exception:
                pass
            self.__dict__ = new_dict
            await self.initialize()
            return

        try:
            # Try to create a visible ModernGL context. If the environment has no
            # windowing/OpenGL, this may fail. We try a visible context first, then
            # fall back to a standalone (offscreen) context so we can still render
            # into a texture and blit to the Tk canvas.
            try:
                self.ctx = moderngl.create_context()
                self._offscreen = False
            except Exception as e_ctx:
                logger.debug(f"moderngl.create_context() failed: {e_ctx}; trying standalone context")
                # Try standalone offscreen context
                try:
                    self.ctx = moderngl.create_standalone_context()
                    self._offscreen = True
                    logger.info("Created ModernGL standalone (offscreen) context")
                except Exception as e_off:
                    # Re-raise original to be handled below
                    raise RuntimeError(f"Failed to create ModernGL context: {e_off}")

            # Configure OpenGL settings
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

            if self.antialiasing:
                # Not all moderngl builds expose MULTISAMPLE as a module attribute
                try:
                    ms_flag = getattr(moderngl, 'MULTISAMPLE')
                    self.ctx.enable(ms_flag)
                except Exception:
                    # Ignore if unavailable; antialiasing won't be enabled
                    logger.debug('MULTISAMPLE flag not available in moderngl; skipping antialiasing')

            # Create framebuffer for rendering (size may be updated later)
            width = 800
            height = 600
            self.fbo = self.ctx.framebuffer(
                color_attachments=[self.ctx.texture((width, height), 4)],
                depth_attachment=self.ctx.depth_texture((width, height))
            )
            self._fbo_size = (width, height)

            # Load shaders
            await self._load_shaders()

            logger.info("OpenGL context initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize renderer: {e}")
            logger.warning("Falling back to headless renderer. No OpenGL window detected or context creation failed. "
                           "If you want 3D rendering, check your GPU drivers and ModernGL installation.")
            headless = HeadlessRenderer(self.target_fps, window=self.window)
            self.__class__ = HeadlessRendererProxy
            new_dict = headless.__dict__
            try:
                new_dict['window'] = self.window
            except Exception:
                pass
            self.__dict__ = new_dict
            await self.initialize()
    
    async def _load_shaders(self):
        """Load VRM-compatible shaders."""
        try:
            # MToon shader for VRM materials
            self.mtoon_program = load_shader_program(
                self.ctx,
                "assets/shaders/mtoon.vert",
                "assets/shaders/mtoon.frag"
            )
            
            # Standard PBR shader fallback
            self.standard_program = load_shader_program(
                self.ctx,
                "assets/shaders/standard.vert", 
                "assets/shaders/standard.frag"
            )
            
            logger.info("Shaders loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load shaders: {e}")
            # Create basic fallback shaders
            await self._create_fallback_shaders()
    
    async def _create_fallback_shaders(self):
        """Create basic fallback shaders if files don't exist."""
        
        # Basic vertex shader
        vertex_shader = """
        #version 330 core
        
        in vec3 position;
        in vec3 normal;
        in vec2 texcoord;
        in ivec4 joints;
        in vec4 weights;
        
        uniform mat4 mvp_matrix;
        uniform mat4 model_matrix;
        uniform mat4 bone_matrices[64];
        uniform bool use_skinning;
        
        out vec3 world_pos;
        out vec3 world_normal;
        out vec2 uv;
        
        void main() {
            vec4 pos = vec4(position, 1.0);
            vec3 norm = normal;
            
            if (use_skinning) {
                mat4 skin_matrix = 
                    weights.x * bone_matrices[joints.x] +
                    weights.y * bone_matrices[joints.y] +
                    weights.z * bone_matrices[joints.z] +
                    weights.w * bone_matrices[joints.w];
                
                pos = skin_matrix * pos;
                norm = mat3(skin_matrix) * norm;
            }
            
            world_pos = (model_matrix * pos).xyz;
            world_normal = normalize(mat3(model_matrix) * norm);
            uv = texcoord;
            
            gl_Position = mvp_matrix * pos;
        }
        """
        
        # Basic fragment shader with MToon-like features
        fragment_shader = """
        #version 330 core
        
        in vec3 world_pos;
        in vec3 world_normal;
        in vec2 uv;
        
        uniform sampler2D main_texture;
        uniform sampler2D shade_texture;
        uniform sampler2D emission_texture;
        uniform sampler2D normal_texture;
        
        uniform vec4 base_color;
        uniform vec4 shade_color;
        uniform vec3 emission_color;
        uniform float shade_shift;
        uniform float shade_toony;
        uniform float light_color_attenuation;
        uniform float alpha_cutoff;
        
        uniform vec3 light_direction;
        uniform vec3 light_color;
        uniform vec3 camera_position;
        
        out vec4 fragColor;
        
        void main() {
            vec4 main_tex = texture(main_texture, uv);
            vec4 shade_tex = texture(shade_texture, uv);
            vec3 emission_tex = texture(emission_texture, uv).rgb;
            vec3 normal_tex = texture(normal_texture, uv).rgb * 2.0 - 1.0;
            
            vec3 normal = normalize(world_normal + normal_tex * 0.1);
            vec3 light_dir = normalize(-light_direction);
            
            // MToon-style toon shading
            float ndotl = dot(normal, light_dir);
            float shade_intensity = 1.0 - shade_shift;
            float toon_shadow = smoothstep(shade_intensity - shade_toony * 0.5, 
                                         shade_intensity + shade_toony * 0.5, ndotl);
            
            vec3 lit_color = main_tex.rgb * base_color.rgb;
            vec3 shade_lit_color = shade_tex.rgb * shade_color.rgb;
            vec3 final_color = mix(shade_lit_color, lit_color, toon_shadow);
            
            // Add emission
            final_color += emission_tex * emission_color;
            
            // Lighting attenuation
            final_color = mix(final_color, final_color * light_color, light_color_attenuation);
            
            float alpha = main_tex.a * base_color.a;
            
            // Alpha cutoff
            if (alpha < alpha_cutoff) {
                discard;
            }
            
            fragColor = vec4(final_color, alpha);
        }
        """
        
        self.mtoon_program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
        
        logger.info("Fallback shaders created")
    
    async def load_character(self, character: Character):
        """Load a VRM character for rendering."""
        self.character = character
        
        if not character.vrm_data:
            logger.warning("No VRM data to load")
            return
        
        try:
            await self._parse_vrm_data(character.vrm_data)
            await self._create_mesh_objects()
            await self._load_textures()
            await self._setup_materials()
            await self._setup_animations()
            
            logger.info(f"Character '{character.name}' loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load character: {e}")
            raise
    
    async def _parse_vrm_data(self, vrm_data: GLTF2):
        """Parse VRM/glTF data and extract rendering information."""
        self.meshes = []
        
        # Parse meshes
        for mesh_idx, mesh in enumerate(vrm_data.meshes or []):
            mesh_data = {
                'name': mesh.name or f"Mesh_{mesh_idx}",
                'primitives': []
            }
            
            for prim_idx, primitive in enumerate(mesh.primitives):
                # Convert Attributes (pygltflib Attributes object) to a plain dict
                attributes_dict = {}
                try:
                    attrs_obj = primitive.attributes
                    if attrs_obj is not None:
                        # prefer __dict__ if present
                        if hasattr(attrs_obj, '__dict__'):
                            for k, v in vars(attrs_obj).items():
                                # Only include GLTF attribute names (uppercase)
                                if isinstance(k, str) and k.isupper():
                                    attributes_dict[k] = v
                        else:
                            # fallback: iterate attribute names likely present
                            for k in dir(attrs_obj):
                                if k.isupper():
                                    try:
                                        val = getattr(attrs_obj, k)
                                    except Exception:
                                        continue
                                    attributes_dict[k] = val
                except Exception:
                    attributes_dict = {}

                prim_data = {
                    'material': primitive.material,
                    'attributes': attributes_dict,
                    'indices': primitive.indices,
                    'mode': primitive.mode or 4  # TRIANGLES
                }
                mesh_data['primitives'].append(prim_data)
            
            self.meshes.append(mesh_data)
        
        # Parse materials
        self.materials = {}
        for mat_idx, material in enumerate(vrm_data.materials or []):
            material_name = material.name or f"Material_{mat_idx}"
            self.materials[material_name] = self._parse_material(material)
        
        logger.info(f"Parsed {len(self.meshes)} meshes and {len(self.materials)} materials")
    
    def _parse_material(self, material) -> Dict[str, Any]:
        """Parse material properties including MToon extensions."""
        mat_data = {
            'name': material.name,
            'base_color': [1.0, 1.0, 1.0, 1.0],
            'metallic': 0.0,
            'roughness': 1.0,
            'emission': [0.0, 0.0, 0.0],
            'alpha_mode': 'OPAQUE',
            'alpha_cutoff': 0.5,
            'double_sided': False,
            'textures': {}
        }
        
        # Standard PBR properties
        if hasattr(material, 'pbrMetallicRoughness') and material.pbrMetallicRoughness:
            pbr = material.pbrMetallicRoughness
            if pbr.baseColorFactor:
                mat_data['base_color'] = pbr.baseColorFactor
            if pbr.metallicFactor is not None:
                mat_data['metallic'] = pbr.metallicFactor
            if pbr.roughnessFactor is not None:
                mat_data['roughness'] = pbr.roughnessFactor
            
            # Textures
            if pbr.baseColorTexture:
                mat_data['textures']['base_color'] = pbr.baseColorTexture.index
        
        # MToon extension properties
        if hasattr(material, 'extensions') and material.extensions:
            mtoon = material.extensions.get('VRM_materials_mtoon')
            if mtoon:
                # MToon-specific properties
                mat_data.update({
                    'shade_color': mtoon.get('shadeColorFactor', [0.97, 0.81, 0.86, 1.0]),
                    'shade_shift': mtoon.get('shadingShiftFactor', 0.0),
                    'shade_toony': mtoon.get('shadingToonyFactor', 0.9),
                    'light_color_attenuation': mtoon.get('lightColorAttenuationFactor', 0.0),
                    'rim_color': mtoon.get('rimColorFactor', [0.0, 0.0, 0.0, 1.0]),
                    'rim_lighting_mix': mtoon.get('rimLightingMixFactor', 0.0),
                    'rim_fresnel_power': mtoon.get('rimFresnelPowerFactor', 1.0),
                    'rim_lift': mtoon.get('rimLiftFactor', 0.0),
                })
                
                # MToon textures
                texture_mappings = {
                    'shadeColorTexture': 'shade_texture',
                    'emissionTexture': 'emission_texture', 
                    'rimTexture': 'rim_texture',
                    'normalTexture': 'normal_texture'
                }
                
                for mtoon_key, tex_key in texture_mappings.items():
                    if mtoon_key in mtoon and mtoon[mtoon_key]:
                        mat_data['textures'][tex_key] = mtoon[mtoon_key]['index']
        
        return mat_data
    
    async def _create_mesh_objects(self):
        """Create ModernGL mesh objects from parsed data."""
        if not self.character or not self.character.vrm_data:
            return
        
        vrm_data = self.character.vrm_data
        
        for mesh_data in self.meshes:
            for primitive in mesh_data['primitives']:
                # Extract vertex data
                vertex_data = self._extract_vertex_data(vrm_data, primitive['attributes'])
                
                # Create vertex buffer
                vbo = self.ctx.buffer(vertex_data.tobytes())
                
                # Create index buffer if available
                ibo = None
                if primitive['indices'] is not None:
                    indices = self._extract_indices(vrm_data, primitive['indices'])
                    ibo = self.ctx.buffer(indices.tobytes())
                
                # Create vertex array object
                vao = self.ctx.vertex_array(
                    self.mtoon_program,
                    [(vbo, '3f 3f 2f 4i 4f', 'position', 'normal', 'texcoord', 'joints', 'weights')],
                    ibo
                )
                
                primitive['vao'] = vao
                primitive['vertex_count'] = len(vertex_data)
    
    def _extract_vertex_data(self, vrm_data: GLTF2, attributes: Dict) -> np.ndarray:
        """Extract and interleave vertex data from glTF accessors."""
        vertex_count = 0
        attribute_data = {}
        
        # Define required attributes with defaults
        required_attrs = {
            'POSITION': np.zeros((0, 3), dtype=np.float32),
            'NORMAL': np.zeros((0, 3), dtype=np.float32), 
            'TEXCOORD_0': np.zeros((0, 2), dtype=np.float32),
            'JOINTS_0': np.zeros((0, 4), dtype=np.int32),
            'WEIGHTS_0': np.zeros((0, 4), dtype=np.float32)
        }
        
        # Extract each attribute
        for attr_name, accessor_idx in attributes.items():
            if accessor_idx is not None:
                data = self._get_accessor_data(vrm_data, accessor_idx)
                attribute_data[attr_name] = data
                if vertex_count == 0:
                    vertex_count = len(data)
        
        # Ensure all required attributes exist
        for attr_name, default_data in required_attrs.items():
            if attr_name not in attribute_data:
                if vertex_count > 0:
                    # Create default data for missing attributes
                    shape = (vertex_count,) + default_data.shape[1:]
                    attribute_data[attr_name] = np.zeros(shape, dtype=default_data.dtype)
                    
                    # Special defaults
                    if attr_name == 'NORMAL':
                        attribute_data[attr_name][:, 1] = 1.0  # Up vector
                    elif attr_name == 'WEIGHTS_0':
                        attribute_data[attr_name][:, 0] = 1.0  # First bone weight
        
        # Interleave vertex data
        vertex_data = np.zeros(vertex_count, dtype=[
            ('position', np.float32, 3),
            ('normal', np.float32, 3),
            ('texcoord', np.float32, 2),
            ('joints', np.int32, 4),
            ('weights', np.float32, 4)
        ])
        
        vertex_data['position'] = attribute_data.get('POSITION', np.zeros((vertex_count, 3)))
        vertex_data['normal'] = attribute_data.get('NORMAL', np.zeros((vertex_count, 3)))
        vertex_data['texcoord'] = attribute_data.get('TEXCOORD_0', np.zeros((vertex_count, 2)))
        vertex_data['joints'] = attribute_data.get('JOINTS_0', np.zeros((vertex_count, 4), dtype=np.int32))
        vertex_data['weights'] = attribute_data.get('WEIGHTS_0', np.zeros((vertex_count, 4)))
        
        return vertex_data.view(np.float32).reshape(-1, 16)  # 3+3+2+4+4 = 16 floats per vertex
    
    def _get_accessor_data(self, vrm_data: GLTF2, accessor_idx: int) -> np.ndarray:
        """Get data from a glTF accessor."""
        if accessor_idx >= len(vrm_data.accessors):
            raise ValueError(f"Invalid accessor index: {accessor_idx}")
        
        accessor = vrm_data.accessors[accessor_idx]

        if accessor.bufferView is None:
            # Some accessors may be sparse or have no buffer view
            raise ValueError(f"Accessor {accessor_idx} has no bufferView")

        buffer_view = vrm_data.bufferViews[accessor.bufferView]
        buffer = vrm_data.buffers[buffer_view.buffer]

        # Try multiple methods to load buffer data
        buffer_data = None
        
        # Method 1: Check if buffer already has binary_blob (GLB format)
        # This should be set for GLB/VRM files loaded from cache or directly
        # Note: binary_blob might be a method, not actual data
        existing_blob = None
        try:
            if hasattr(vrm_data, 'binary_blob'):
                blob_attr = getattr(vrm_data, 'binary_blob')
                if isinstance(blob_attr, (bytes, bytearray, memoryview)):
                    existing_blob = blob_attr
        except Exception:
            pass
        
        if existing_blob:
            # For GLB files, buffer 0 is typically the binary blob
            if buffer_view.buffer == 0:
                buffer_data = existing_blob
                logger.debug(f"Using binary_blob for buffer {buffer_view.buffer}")
            else:
                logger.debug(f"binary_blob exists but buffer index is {buffer_view.buffer}, not 0")
        
        # Method 2: For GLB files without uri, try to use binary_blob even if buffer index isn't 0
        if buffer_data is None and existing_blob:
            try:
                # In GLB files, all buffers typically point to the same binary blob
                buffer_data = existing_blob
                logger.debug(f"Using binary_blob fallback for buffer {buffer_view.buffer}")
            except Exception as e:
                logger.debug(f"binary_blob fallback failed: {e}")
        
        # Method 3: Try GLTF2 helper method
        if buffer_data is None:
            try:
                buffer_uri = getattr(buffer, 'uri', None)
                if buffer_uri:  # Only try if uri exists
                    buffer_data = vrm_data.get_data_from_buffer_uri(buffer_uri)
                    logger.debug(f"Loaded buffer via get_data_from_buffer_uri for buffer {buffer_view.buffer}")
            except Exception as e:
                logger.debug(f"get_data_from_buffer_uri failed: {e}")
        
        # Method 4: Handle data URI (base64 encoded)
        if buffer_data is None and hasattr(buffer, 'uri') and buffer.uri:
            buffer_uri = buffer.uri
            if buffer_uri.startswith('data:'):
                try:
                    import base64
                    # Extract base64 data after the comma
                    header, data_b64 = buffer_uri.split(',', 1)
                    buffer_data = base64.b64decode(data_b64)
                    logger.debug(f"Decoded data URI for buffer {buffer_view.buffer}")
                except Exception as e:
                    logger.warning(f"Failed to decode data URI: {e}")
        
        # Method 5: Try reading external file
        if buffer_data is None and hasattr(buffer, 'uri') and buffer.uri and not buffer.uri.startswith('data:'):
            try:
                # Resolve relative to the model path if present
                buffer_uri = buffer.uri
                possible_bases = []
                
                if hasattr(vrm_data, '_path') and vrm_data._path:
                    possible_bases.append(Path(vrm_data._path))
                
                if self.character and self.character.model_path:
                    possible_bases.append(Path(self.character.model_path).parent)
                
                for base in possible_bases:
                    buffer_path = Path(base) / buffer_uri
                    if buffer_path.exists():
                        buffer_data = buffer_path.read_bytes()
                        logger.debug(f"Loaded external buffer file: {buffer_path}")
                        break
            except Exception as e:
                logger.warning(f"Failed to load external buffer file: {e}")
        
        # Final check: if still None, raise error with detailed diagnostics
        if buffer_data is None:
            has_blob = existing_blob is not None
            blob_size = len(existing_blob) if existing_blob else 0
            raise RuntimeError(
                f"Failed to load buffer data for accessor {accessor_idx}.\n"
                f"  Buffer index: {buffer_view.buffer}\n"
                f"  Buffer URI: {getattr(buffer, 'uri', 'None')}\n"
                f"  Has binary_blob: {has_blob}\n"
                f"  Binary blob size: {blob_size} bytes\n"
                f"  Buffer view offset: {buffer_view.byteOffset or 0}\n"
                f"  Buffer view length: {buffer_view.byteLength or 0}\n"
                f"  Model path: {getattr(vrm_data, '_path', 'Unknown')}\n"
                f"Possible fixes:\n"
                f"  1. Clear cache and reload: delete {Path.home() / '.cache' / 'vrm_ai_chatbot'}\n"
                f"  2. Check if VRM file is corrupted\n"
                f"  3. Try re-downloading the VRM model"
            )

        # Normalize buffer_data into a bytes-like object. Some pygltflib
        # helpers or external loaders may return memoryviews, numpy arrays,
        # or even bound methods in certain circumstances. Coerce common
        # types to bytes so slicing works reliably below and provide a
        # helpful error if we can't.
        try:
            # If buffer_data is a callable (unexpected), try calling it to
            # obtain the actual bytes. This covers cases where a method
            # reference was accidentally stored instead of its result.
            if callable(buffer_data):
                logger.debug(f"buffer_data is callable; attempting to call it to obtain bytes (type={type(buffer_data)})")
                buffer_data = buffer_data()

            # memoryview -> bytes
            if isinstance(buffer_data, memoryview):
                buffer_data = buffer_data.tobytes()

            # numpy arrays -> bytes
            try:
                import numpy as _np
                if isinstance(buffer_data, _np.ndarray):
                    buffer_data = buffer_data.tobytes()
            except Exception:
                # numpy may not be available here (should be, but guard)
                pass

            # Objects exposing tobytes() -> use it
            if not isinstance(buffer_data, (bytes, bytearray)) and hasattr(buffer_data, 'tobytes'):
                try:
                    buffer_data = buffer_data.tobytes()
                except Exception:
                    # ignore and fallthrough to final type check
                    pass

        except Exception as e:
            logger.debug(f"Failed to normalize buffer_data: {e}")

        # Final type guard: we need a bytes-like object for slicing
        if not isinstance(buffer_data, (bytes, bytearray, memoryview)):
            raise RuntimeError(
                f"Unsupported buffer data type for accessor {accessor_idx}: {type(buffer_data)}. "
                "Expected bytes/bytearray/memoryview or numpy array."
            )

        # Calculate offset and stride. Guard defaults to zero when attributes are None
        bv_offset = buffer_view.byteOffset or 0
        acc_offset = accessor.byteOffset or 0
        offset = bv_offset + acc_offset

        # Determine stride: prefer bufferView.byteStride if present and non-zero,
        # otherwise compute tightly-packed stride from accessor layout
        stride = buffer_view.byteStride or self._get_accessor_stride(accessor)

        # Determine numpy dtype for a single component
        dtype = self._get_accessor_dtype(accessor)
        component_count = 1
        if accessor.type != 'SCALAR':
            component_count = {
                'VEC2': 2, 'VEC3': 3, 'VEC4': 4,
                'MAT2': 4, 'MAT3': 9, 'MAT4': 16
            }[accessor.type]

        # Size in bytes for one accessor element (component_count * dtype.itemsize)
        element_size = component_count * dtype().nbytes if hasattr(dtype, '__call__') else component_count * dtype().itemsize

        # If stride equals element_size then tightly packed; otherwise handle stride
        data = None
        if stride == element_size:
            end_offset = offset + accessor.count * element_size
            raw_slice = buffer_data[offset:end_offset]
            data = np.frombuffer(raw_slice, dtype=dtype)
            if component_count > 1:
                data = data.reshape(-1, component_count)
        else:
            # Strided access: extract each element respecting the stride
            data = np.zeros((accessor.count, component_count), dtype=dtype)
            for i in range(accessor.count):
                item_offset = offset + i * stride
                item_bytes = buffer_data[item_offset:item_offset + element_size]
                if len(item_bytes) < element_size:
                    raise RuntimeError(f"Insufficient data for accessor {accessor_idx} at index {i}")
                item = np.frombuffer(item_bytes, dtype=dtype)
                if component_count > 1:
                    data[i, :] = item.reshape(component_count)
                else:
                    data[i] = item[0]

        return data
    
    def _get_accessor_stride(self, accessor) -> int:
        """Calculate the byte stride for an accessor."""
        component_size = {
            5120: 1,  # BYTE
            5121: 1,  # UNSIGNED_BYTE  
            5122: 2,  # SHORT
            5123: 2,  # UNSIGNED_SHORT
            5125: 4,  # UNSIGNED_INT
            5126: 4   # FLOAT
        }[accessor.componentType]
        
        component_count = {
            'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4,
            'MAT2': 4, 'MAT3': 9, 'MAT4': 16
        }[accessor.type]
        
        return component_size * component_count
    
    def _get_accessor_dtype(self, accessor) -> np.dtype:
        """Get numpy dtype for an accessor."""
        component_dtypes = {
            5120: np.int8,    # BYTE
            5121: np.uint8,   # UNSIGNED_BYTE
            5122: np.int16,   # SHORT  
            5123: np.uint16,  # UNSIGNED_SHORT
            5125: np.uint32,  # UNSIGNED_INT
            5126: np.float32  # FLOAT
        }
        return component_dtypes[accessor.componentType]
    
    def _extract_indices(self, vrm_data: GLTF2, accessor_idx: int) -> np.ndarray:
        """Extract index data from accessor."""
        indices = self._get_accessor_data(vrm_data, accessor_idx)
        return indices.astype(np.uint32).flatten()
    
    async def _load_textures(self):
        """Load all textures referenced by materials."""
        if not self.character or not self.character.vrm_data:
            return
        
        vrm_data = self.character.vrm_data
        self.textures = {}
        
        if not vrm_data.images:
            logger.info("No textures to load")
            return
        
        for img_idx, image in enumerate(vrm_data.images):
            try:
                # Resolve image data robustly
                image_data = None
                
                # 1) data URI
                if getattr(image, 'uri', None):
                    uri = image.uri
                    if uri.startswith('data:'):
                        import base64
                        try:
                            header, data = uri.split(',', 1)
                            image_data = base64.b64decode(data)
                        except Exception as e:
                            logger.warning(f"Failed to decode data URI for image {img_idx}: {e}")
                    else:
                        # 2) external file relative to model path
                        possible_bases = []
                        if hasattr(vrm_data, '_path') and vrm_data._path:
                            possible_bases.append(Path(vrm_data._path))
                        if self.character and self.character.model_path:
                            possible_bases.append(Path(self.character.model_path).parent)

                        for base in possible_bases:
                            try_path = (Path(base) / uri)
                            if try_path.exists():
                                try:
                                    image_data = try_path.read_bytes()
                                    break
                                except Exception:
                                    logger.debug(f"Failed to read image file {try_path}")

                # 3) bufferView fallback - THIS IS THE FIX
                if image_data is None:
                    if getattr(image, 'bufferView', None) is not None:
                        buffer_view = vrm_data.bufferViews[image.bufferView]
                        buffer = vrm_data.buffers[buffer_view.buffer]
                        
                        # Check for binary_blob first (GLB format)
                        buffer_data = None
                        if hasattr(vrm_data, 'binary_blob'):
                            blob = vrm_data.binary_blob
                            # FIX: Check if it's callable before calling it
                            if callable(blob):
                                buffer_data = blob()
                            elif isinstance(blob, (bytes, bytearray, memoryview)):
                                buffer_data = blob
                        
                        # Fallback to get_data_from_buffer_uri
                        if buffer_data is None:
                            try:
                                buf_uri = getattr(buffer, 'uri', None)
                                if buf_uri:
                                    result = vrm_data.get_data_from_buffer_uri(buf_uri)
                                    # FIX: Handle if result is callable
                                    if callable(result):
                                        buffer_data = result()
                                    else:
                                        buffer_data = result
                            except Exception as e:
                                logger.debug(f"get_data_from_buffer_uri failed: {e}")
                        
                        # Last resort: try reading from file
                        if buffer_data is None:
                            try:
                                buf_uri = buffer.uri
                                if buf_uri and not buf_uri.startswith('data:'):
                                    base = getattr(vrm_data, '_path', Path('.'))
                                    buffer_path = Path(base) / buf_uri
                                    buffer_data = buffer_path.read_bytes()
                            except Exception as e:
                                raise RuntimeError(f"Failed to load image buffer for image {img_idx}: {e}")
                        
                        if buffer_data is None:
                            raise RuntimeError(f"Failed to load image buffer for image {img_idx}")
                        
                        # Extract the slice for this image
                        offset = buffer_view.byteOffset or 0
                        length = buffer_view.byteLength or 0
                        image_data = buffer_data[offset:offset + length]

                if not image_data:
                    raise RuntimeError(f"No image data found for image index {img_idx}")
                
                # Load image with PIL
                from PIL import Image
                import io
                
                pil_image = Image.open(io.BytesIO(image_data))
                pil_image = pil_image.convert('RGBA')
                
                # Create OpenGL texture
                texture = self.ctx.texture(pil_image.size, 4)
                texture.write(pil_image.tobytes())
                texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                texture.repeat_x = True
                texture.repeat_y = True
                
                self.textures[img_idx] = texture
                
            except Exception as e:
                logger.error(f"Failed to load texture {img_idx}: {e}")
                # Create default white texture
                default_texture = self.ctx.texture((1, 1), 4)
                default_texture.write(b'\xff\xff\xff\xff')
                self.textures[img_idx] = default_texture
        
        logger.info(f"Loaded {len(self.textures)} textures")
    
    async def _setup_materials(self):
        """Setup material uniforms and texture bindings."""
        # This would set up material properties for rendering
        pass
    
    async def _setup_animations(self):
        """Setup bone animations and blend shapes."""
        # This would parse and setup VRM animations
        pass
    
    async def render_frame(self):
        """Render a single frame."""
        if not self.character or not self.meshes:
            return
        
        try:
            # Clear framebuffer
            self.fbo.clear(0.0, 0.0, 0.0, 0.0)
            # Bind FBO for rendering
            self.fbo.use()
            
            # Update animation
            await self._update_animation()
            
            # Calculate matrices
            mvp_matrix = self.projection_matrix @ self.view_matrix @ self.model_matrix
            
            # Render each mesh
            for mesh_data in self.meshes:
                for primitive in mesh_data['primitives']:
                    if 'vao' not in primitive:
                        continue
                    
                    vao = primitive['vao']
                    
                    # Set uniforms
                    if 'mvp_matrix' in self.mtoon_program:
                        self.mtoon_program['mvp_matrix'].write(mvp_matrix.astype(np.float32).tobytes())
                    if 'model_matrix' in self.mtoon_program:
                        self.mtoon_program['model_matrix'].write(self.model_matrix.astype(np.float32).tobytes())
                    
                    # Bind textures
                    texture_units = 0
                    material_idx = primitive.get('material', 0)
                    if material_idx < len(list(self.materials.values())):
                        material = list(self.materials.values())[material_idx]
                        for tex_name, tex_idx in material.get('textures', {}).items():
                            if tex_idx in self.textures:
                                self.textures[tex_idx].use(texture_units)
                                if tex_name + '_sampler' in self.mtoon_program:
                                    self.mtoon_program[tex_name + '_sampler'].value = texture_units
                                texture_units += 1
                    
                    # Render
                    vao.render()
            
            # Copy to main framebuffer
            if getattr(self, '_offscreen', False):
                # Read pixels from the offscreen framebuffer and blit to Tk canvas
                try:
                    # Read the rendered frame
                    pixels = self.fbo.read(components=4, alignment=1)
                    
                    # Convert to PIL Image
                    from PIL import Image as PImage
                    from PIL import ImageTk
                    
                    img = PImage.frombytes('RGBA', self._fbo_size, pixels)
                    # Flip vertically (OpenGL is bottom-up, Tkinter is top-down)
                    img = img.transpose(PImage.FLIP_TOP_BOTTOM)
                    
                    # Resize to window size if needed
                    if self.window:
                        try:
                            w, h = self.window.get_window_size()
                            if (w, h) != self._fbo_size:
                                img = img.resize((w, h), PImage.Resampling.LANCZOS)
                        except Exception as e:
                            logger.debug(f"Could not get window size: {e}")
                    
                    # Convert to Tkinter PhotoImage
                    # Must provide master window to avoid "no default root" error
                    photo = ImageTk.PhotoImage(img, master=self.window.root)
                    
                    # Keep reference to prevent garbage collection
                    self._tk_photo = photo
                    
                    # Clear previous image and draw new one
                    if self.window and self.window.canvas:
                        self.window.canvas.delete('gl_render')
                        self.window.canvas.create_image(
                            0, 0, 
                            anchor='nw', 
                            image=photo, 
                            tags='gl_render'
                        )
                        # Force canvas update
                        self.window.canvas.update_idletasks()
                    
                except Exception as e:
                    logger.error(f"Failed to display offscreen render: {e}", exc_info=True)
            else:
                # If not offscreen, assume there's a screen to bind
                try:
                    self.ctx.screen.use()
                except Exception:
                    pass
                # make sure color attachment is bound if needed
                try:
                    self.fbo.color_attachments[0].use(0)
                except Exception:
                    pass
            
            # Update performance counters
            self.frame_count += 1
            
        except Exception as e:
            logger.error(f"Render error: {e}")
    
    async def _update_animation(self):
        """Update character animation state."""
        if not self.character:
            return
        
        # Update animation time
        import time
        current_time = time.time()
        if not hasattr(self, '_last_time'):
            self._last_time = current_time
        
        dt = current_time - self._last_time
        self._last_time = current_time
        
        self.animation_time += dt
        
        # Update character state
        if hasattr(self.character, 'update_animation'):
            await self.character.update_animation(dt)
    
    async def update_character(self):
        """Update character rendering after state changes."""
        # This would update character-specific rendering state
        pass
    
    async def shutdown(self):
        """Cleanup rendering resources."""
        try:
            # Release textures
            for texture in self.textures.values():
                texture.release()
            
            # Release programs
            if self.mtoon_program:
                self.mtoon_program.release()
            if self.standard_program:
                self.standard_program.release()
            
            # Release framebuffer
            if self.fbo:
                self.fbo.release()
            
            # Release context
            if self.ctx:
                self.ctx.release()
            
            logger.info("Renderer shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during renderer shutdown: {e}")


class HeadlessRenderer:
    """A simple headless renderer that implements the same public API
    expected by the application but does no GPU work. Useful for running
    in environments without OpenGL support.
    """

    def __init__(self, target_fps: int = 60, window=None):
        self.target_fps = target_fps
        self.window = window
        self.character = None
        self.meshes = []
        self.materials = {}
        self.textures = {}
        self.mtoon_program = None
        self.standard_program = None
        self.frame_count = 0
        self.fps = 0
        # Keep a reference to any Tk PhotoImage used for preview to avoid GC
        self._tk_preview = None

    async def initialize(self):
        logger.info("HeadlessRenderer initialized (no GPU/OpenGL)")

    async def _load_shaders(self):
        logger.info("HeadlessRenderer: _load_shaders (noop)")

    async def load_character(self, character):
        self.character = character
        logger.info(f"HeadlessRenderer: loaded character {getattr(character, 'name', None)}")

        # Draw a simple preview into the window's Tk canvas so the user sees something
        try:
            if self.window and getattr(self.window, 'canvas', None):
                from PIL import Image, ImageDraw, ImageFont, ImageTk

                # Get canvas size
                try:
                    w, h = self.window.get_window_size()
                except Exception:
                    w, h = (self.window.width or 800, self.window.height or 600)

                # Create an RGBA image for preview
                img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)

                # Draw a translucent panel and character name
                panel_color = (20, 20, 20, 200)
                draw.rectangle([(10, 10), (w - 10, h - 10)], fill=panel_color)

                title = getattr(character, 'name', 'Character') or 'Character'
                subtitle = '(Headless preview - 3D disabled)'

                try:
                    # Try to use a default truetype font if available
                    font = ImageFont.truetype('arial.ttf', 24)
                    font_sm = ImageFont.truetype('arial.ttf', 14)
                except Exception:
                    font = ImageFont.load_default()
                    font_sm = ImageFont.load_default()

                # Title centered
                text_x = 20
                text_y = 20
                draw.text((text_x, text_y), title, font=font, fill=(255, 255, 255, 255))
                draw.text((text_x, text_y + 40), subtitle, font=font_sm, fill=(200, 200, 200, 255))

                # Convert to PhotoImage and place on canvas
                photo = ImageTk.PhotoImage(img)
                self._tk_preview = photo
                try:
                    # Remove previous preview if any
                    self.window.canvas.delete('headless_preview')
                except Exception:
                    pass
                try:
                    self.window.canvas.create_image(0, 0, anchor='nw', image=photo, tags='headless_preview')
                except Exception as e:
                    logger.debug(f"Failed to draw headless preview on canvas: {e}")

        except Exception as e:
            logger.debug(f"Headless preview creation failed: {e}")

    async def render_frame(self):
        # Simulate work and increment counters
        self.frame_count += 1

    async def update_character(self):
        logger.debug("HeadlessRenderer: update_character noop")

    async def shutdown(self):
        logger.info("HeadlessRenderer shutdown complete")


class HeadlessRendererProxy(HeadlessRenderer):
    """Proxy class used to swap instances into VRMRenderer variable locations
    while preserving method names and attributes expected by the rest of the
    application. It simply inherits HeadlessRenderer.
    """
    pass
