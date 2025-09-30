"""
Character Model - Represents a VRM character with personality and animation state.
Handles VRM file loading, personality traits, and animation coordination.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
from dataclasses import dataclass, field

from pygltflib import GLTF2
import hashlib
import os

logger = logging.getLogger(__name__)

@dataclass
class PersonalityTrait:
    """Represents a personality trait with intensity."""
    name: str
    intensity: float = 0.5  # 0.0 to 1.0
    description: str = ""

@dataclass
class AnimationState:
    """Current animation state of the character."""
    is_speaking: bool = False
    is_listening: bool = False
    current_emotion: str = "neutral"
    eye_blink_timer: float = 0.0
    mouth_sync_data: List[float] = field(default_factory=list)
    head_rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    body_pose: str = "idle"

class Character:
    """VRM character with personality and animation capabilities."""
    
    def __init__(self,
                 name: str = "AI Character",
                 personality_traits: List[str] = None,
                 background_story: str = "",
                 speaking_style: str = "friendly"):
        
        self.name = name
        self.background_story = background_story
        self.speaking_style = speaking_style
        
        # Personality system
        self.personality_traits: Dict[str, PersonalityTrait] = {}
        if personality_traits:
            for trait in personality_traits:
                self.add_personality_trait(trait)
        
        # VRM model data
        self.vrm_data: Optional[GLTF2] = None
        self.model_path: Optional[Path] = None
        self.vrm_extensions: Dict[str, Any] = {}
        
        # Animation state
        self.animation_state = AnimationState()
        
        # Facial expressions and blend shapes
        self.blend_shapes: Dict[str, float] = {}
        self.available_expressions: List[str] = []
        
        # Voice and audio
        self.voice_settings: Dict[str, Any] = {
            'pitch': 0.0,
            'speed': 1.0,
            'volume': 1.0,
            'accent': 'neutral'
        }
        
        # Conversation state
        self.conversation_history: List[Dict[str, str]] = []
        self.current_mood: str = "neutral"
        self.energy_level: float = 0.7
        
        logger.info(f"Character '{name}' created")
        # Simple in-memory cache for loaded VRM/GLTF objects keyed by absolute path
        # This accelerates repeated loads and avoids reparsing large files.
        # Note: cache lives only for process lifetime.
        if not hasattr(self.__class__, '_vrm_cache'):
            self.__class__._vrm_cache = {}
    
    def add_personality_trait(self, trait_name: str, intensity: float = 0.5, description: str = ""):
        """Add a personality trait to the character."""
        self.personality_traits[trait_name] = PersonalityTrait(
            name=trait_name,
            intensity=intensity,
            description=description
        )
        logger.debug(f"Added personality trait: {trait_name} (intensity: {intensity})")
    
    def get_personality_description(self) -> str:
        """Generate a description of the character's personality."""
        traits = []
        for trait in self.personality_traits.values():
            if trait.intensity > 0.3:
                intensity_desc = ""
                if trait.intensity > 0.8:
                    intensity_desc = "very "
                elif trait.intensity > 0.6:
                    intensity_desc = "quite "
                
                traits.append(f"{intensity_desc}{trait.name}")
        
        if traits:
            trait_str = ", ".join(traits[:-1])
            if len(traits) > 1:
                trait_str += f" and {traits[-1]}"
            else:
                trait_str = traits[0]
            
            return f"I am {trait_str}. {self.background_story}".strip()
        else:
            return self.background_story
    
    async def load_vrm_model(self, model_path: Path):
        """Load a VRM model file."""
        try:
            # Normalize and validate path early
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"VRM model file not found: {model_path}")

            self.model_path = model_path.resolve()
            logger.info(f"Loading VRM model: {self.model_path}")

            # Return cached parsed GLTF if available (in-memory)
            cache_key = str(self.model_path)
            cached = self.__class__._vrm_cache.get(cache_key)
            if cached:
                logger.info(f"Using cached VRM data for {self.model_path}")
                self.vrm_data = cached
                # still parse extensions and blend shapes to wire up runtime state
                await self._parse_vrm_extensions()
                await self._setup_blend_shapes()
                await self._initialize_animation_state()
                logger.info(f"VRM model loaded successfully (from cache): {self.name}")
                return

            # Read raw bytes first to inspect file magic - some files may be mislabeled
            raw = model_path.read_bytes()

            # Optional on-disk cache: check environment var VRM_CACHE_DIR or use
            # .vrm_cache next to repository root. Cache entries are based on
            # sha256(file_contents). We store the parsed glTF JSON to disk which
            # pygltflib can re-load via from_json.
            cache_dir = os.getenv('VRM_CACHE_DIR')
            if not cache_dir:
                # default cache directory in the repo root or in user's cache
                repo_root = Path(__file__).resolve().parents[2]
                cache_dir = str(repo_root / '.vrm_cache')
            cache_dir_path = Path(cache_dir)
            try:
                cache_dir_path.mkdir(parents=True, exist_ok=True)
            except Exception:
                cache_dir_path = None

            def _file_digest(b: bytes) -> str:
                return hashlib.sha256(b).hexdigest()

            file_digest = _file_digest(raw)
            cache_filename = f"{Path(self.model_path).stem}_{file_digest}.gltf.json"
            cache_file_path = Path(cache_dir) / cache_filename if cache_dir_path else None

            # If cache file exists, load from it (fast)
            if cache_file_path and cache_file_path.exists():
                try:
                    logger.info(f"Loading VRM from on-disk cache: {cache_file_path}")
                    cached_text = cache_file_path.read_text(encoding='utf-8')
                    obj = GLTF2.from_json(cached_text)
                    obj._path = self.model_path.parent
                    obj._name = self.model_path.name
                    self.vrm_data = obj
                    # populate in-memory cache
                    try:
                        self.__class__._vrm_cache[cache_key] = self.vrm_data
                    except Exception:
                        pass
                    await self._parse_vrm_extensions()
                    await self._setup_blend_shapes()
                    await self._initialize_animation_state()
                    logger.info(f"VRM model loaded successfully (from on-disk cache): {self.name}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load from on-disk cache {cache_file_path}: {e}; will re-parse source file")
            
            # Read raw bytes first to inspect file magic - some files may be mislabeled
            def _is_glb(bytes_data: bytes) -> bool:
                # GLB files start with the ASCII magic 'glTF' in the first 4 bytes
                return len(bytes_data) >= 4 and bytes_data[:4] == b'glTF'

            ext = model_path.suffix.lower()

            # Prefer binary loader for true GLB content or common binary extensions.
            if ext in [".glb", ".bin", ".vrm"] or _is_glb(raw):
                try:
                    self.vrm_data = GLTF2.load_binary(str(model_path))
                except Exception as e:
                    logger.debug(f"GLB binary load attempt failed for {model_path}: {e}")
                    # Do not fail immediately; try JSON parsing below as a fallback

            if not self.vrm_data:
                # Attempt to decode as UTF-8 JSON glTF/VRM. Use 'replace' fallback to
                # avoid UnicodeDecodeError on Windows-created files with odd encodings.
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    logger.warning(f"Failed to decode {model_path} as utf-8; using replacement fallback")
                    text = raw.decode("utf-8", errors="replace")

                try:
                    obj = GLTF2.from_json(text)
                    # store metadata used by renderer to resolve external resources
                    obj._path = model_path.parent
                    obj._name = model_path.name
                    self.vrm_data = obj
                except Exception as e:
                    logger.error(f"Failed to parse glTF/VRM file {model_path}: {e}")
                    raise

            # Cache the parsed object for faster subsequent loads
            try:
                self.__class__._vrm_cache[cache_key] = self.vrm_data
            except Exception:
                # If caching fails for any reason, continue without blocking load
                logger.debug("Failed to cache VRM data; continuing without cache")

            # Save to on-disk cache if possible (best-effort)
            if cache_file_path:
                try:
                    js = self.vrm_data.to_json()
                    cache_file_path.write_text(js, encoding='utf-8')
                    logger.debug(f"Wrote VRM cache to {cache_file_path}")
                except Exception as e:
                    logger.debug(f"Failed to write VRM cache to {cache_file_path}: {e}")
            
            # Parse VRM extensions
            await self._parse_vrm_extensions()
            
            # Setup blend shapes
            await self._setup_blend_shapes()
            
            # Initialize animation state
            await self._initialize_animation_state()
            
            logger.info(f"VRM model loaded successfully: {self.name}")
            
        except Exception as e:
            logger.error(f"Failed to load VRM model: {e}")
            raise
    
    async def _parse_vrm_extensions(self):
        """Parse VRM-specific extensions from the glTF file."""
        if not self.vrm_data or not hasattr(self.vrm_data, 'extensions'):
            return
        
        extensions = self.vrm_data.extensions or {}
        
        # VRM extension
        if 'VRM' in extensions:
            vrm_ext = extensions['VRM']
            self.vrm_extensions['VRM'] = vrm_ext
            
            # Extract character info
            if 'meta' in vrm_ext:
                meta = vrm_ext['meta']
                if 'title' in meta and not self.name:
                    self.name = meta['title']
                if 'author' in meta:
                    self.vrm_extensions['author'] = meta['author']
                if 'version' in meta:
                    self.vrm_extensions['version'] = meta['version']
            
            # Extract blend shape mappings
            if 'blendShapeMaster' in vrm_ext:
                self.vrm_extensions['blendShapes'] = vrm_ext['blendShapeMaster']
        
        logger.debug(f"Parsed VRM extensions: {list(self.vrm_extensions.keys())}")
    
    async def _setup_blend_shapes(self):
        """Setup facial expression blend shapes."""
        self.blend_shapes = {}
        self.available_expressions = []
        
        if 'blendShapes' in self.vrm_extensions:
            blend_shape_groups = self.vrm_extensions['blendShapes'].get('blendShapeGroups', [])
            
            for group in blend_shape_groups:
                name = group.get('name', '')
                preset_name = group.get('presetName', '')
                
                # Map preset names to standard expressions
                expression_mapping = {
                    'joy': 'happy',
                    'angry': 'angry',
                    'sorrow': 'sad',
                    'fun': 'excited',
                    'surprised': 'surprised',
                    'blink': 'blink',
                    'blink_l': 'blink_left',
                    'blink_r': 'blink_right',
                    'a': 'mouth_a',
                    'i': 'mouth_i',
                    'u': 'mouth_u',
                    'e': 'mouth_e',
                    'o': 'mouth_o'
                }
                
                expression_name = expression_mapping.get(preset_name, name or preset_name)
                if expression_name:
                    self.blend_shapes[expression_name] = 0.0
                    self.available_expressions.append(expression_name)
        
        # Add default expressions if none found
        if not self.available_expressions:
            default_expressions = [
                'neutral', 'happy', 'sad', 'angry', 'surprised', 'excited',
                'blink', 'mouth_a', 'mouth_i', 'mouth_u', 'mouth_e', 'mouth_o'
            ]
            for expr in default_expressions:
                self.blend_shapes[expr] = 0.0
                self.available_expressions.append(expr)
        
        logger.info(f"Setup {len(self.available_expressions)} blend shape expressions")
    
    async def _initialize_animation_state(self):
        """Initialize the character's animation state."""
        self.animation_state = AnimationState()
        
        # Set default pose based on personality
        if 'confident' in self.personality_traits:
            self.animation_state.body_pose = "confident"
        elif 'shy' in self.personality_traits:
            self.animation_state.body_pose = "shy"
        else:
            self.animation_state.body_pose = "neutral"
        
        logger.debug("Animation state initialized")
    
    async def set_expression(self, expression: str, intensity: float = 1.0):
        """Set a facial expression with given intensity."""
        if expression in self.blend_shapes:
            self.blend_shapes[expression] = max(0.0, min(1.0, intensity))
            logger.debug(f"Set expression '{expression}' to {intensity}")
        else:
            logger.warning(f"Expression '{expression}' not available")
    
    async def clear_expressions(self):
        """Clear all facial expressions."""
        for key in self.blend_shapes:
            self.blend_shapes[key] = 0.0
    
    async def set_emotion(self, emotion: str, intensity: float = 1.0):
        """Set the character's current emotion."""
        await self.clear_expressions()
        
        emotion_mappings = {
            'happy': ['happy', 'joy'],
            'sad': ['sad', 'sorrow'],
            'angry': ['angry'],
            'surprised': ['surprised'],
            'excited': ['excited', 'fun'],
            'neutral': ['neutral']
        }
        
        expressions = emotion_mappings.get(emotion, [emotion])
        for expr in expressions:
            await self.set_expression(expr, intensity)
        
        self.animation_state.current_emotion = emotion
        self.current_mood = emotion
        
        logger.info(f"Character emotion set to '{emotion}' (intensity: {intensity})")
    
    async def start_speaking_animation(self):
        """Start speaking animation and lip sync."""
        self.animation_state.is_speaking = True
        self.animation_state.is_listening = False
        
        # Clear conflicting expressions
        await self.set_expression('mouth_a', 0.0)
        await self.set_expression('mouth_i', 0.0)
        await self.set_expression('mouth_u', 0.0)
        await self.set_expression('mouth_e', 0.0)
        await self.set_expression('mouth_o', 0.0)
        
        logger.debug("Started speaking animation")
    
    async def stop_speaking_animation(self):
        """Stop speaking animation."""
        self.animation_state.is_speaking = False
        self.animation_state.mouth_sync_data = []
        
        # Reset mouth to neutral
        await self.set_expression('mouth_a', 0.0)
        await self.set_expression('mouth_i', 0.0)
        await self.set_expression('mouth_u', 0.0)
        await self.set_expression('mouth_e', 0.0)
        await self.set_expression('mouth_o', 0.0)
        
        logger.debug("Stopped speaking animation")
    
    async def update_lip_sync(self, phonemes: List[str], intensities: List[float]):
        """Update lip sync animation based on phoneme data."""
        if not self.animation_state.is_speaking:
            return
        
        # Map phonemes to mouth shapes
        phoneme_mapping = {
            'a': 'mouth_a', 'ah': 'mouth_a', 'aa': 'mouth_a',
            'i': 'mouth_i', 'ih': 'mouth_i', 'ee': 'mouth_i',
            'u': 'mouth_u', 'uh': 'mouth_u', 'oo': 'mouth_u',
            'e': 'mouth_e', 'eh': 'mouth_e', 'ae': 'mouth_e',
            'o': 'mouth_o', 'oh': 'mouth_o', 'ow': 'mouth_o'
        }
        
        # Clear previous mouth shapes
        for mouth_shape in ['mouth_a', 'mouth_i', 'mouth_u', 'mouth_e', 'mouth_o']:
            await self.set_expression(mouth_shape, 0.0)
        
        # Apply current phonemes
        for phoneme, intensity in zip(phonemes, intensities):
            mouth_shape = phoneme_mapping.get(phoneme.lower())
            if mouth_shape:
                await self.set_expression(mouth_shape, intensity)
    
    async def start_listening_animation(self):
        """Start listening animation."""
        self.animation_state.is_listening = True
        self.animation_state.is_speaking = False
        
        # Subtle animation indicating attention
        # Could include head tracking toward sound source
        
        logger.debug("Started listening animation")
    
    async def stop_listening_animation(self):
        """Stop listening animation."""
        self.animation_state.is_listening = False
        logger.debug("Stopped listening animation")
    
    async def update_animation(self, delta_time: float):
        """Update character animation state."""
        # Update blink animation
        self.animation_state.eye_blink_timer += delta_time
        
        # Automatic blinking every 2-4 seconds
        if self.animation_state.eye_blink_timer > 3.0:
            await self._trigger_blink()
            self.animation_state.eye_blink_timer = 0.0
        
        # Update other animations based on state
        if self.animation_state.is_listening:
            await self._update_listening_animation(delta_time)
        
        # Update breathing animation
        await self._update_breathing_animation(delta_time)
    
    async def _trigger_blink(self):
        """Trigger a natural eye blink."""
        # Quick blink animation
        await self.set_expression('blink', 1.0)
        await asyncio.sleep(0.1)  # Blink duration
        await self.set_expression('blink', 0.0)
    
    async def _update_listening_animation(self, delta_time: float):
        """Update subtle listening animations."""
        # Could include subtle head movements, eye tracking, etc.
        pass
    
    async def _update_breathing_animation(self, delta_time: float):
        """Update breathing animation."""
        # Subtle chest/shoulder movement for breathing
        # This would affect the body pose slightly
        pass
    
    def get_personality_prompt(self) -> str:
        """Generate a personality prompt for AI conversation."""
        personality_desc = self.get_personality_description()
        
        prompt = f"""You are {self.name}, a virtual AI character. 
        
        Personality: {personality_desc}
        
        Speaking style: {self.speaking_style}
        
        Current mood: {self.current_mood}
        
        Voice settings: Pitch {self.voice_settings['pitch']}, Speed {self.voice_settings['speed']}
        
        Instructions:
        - Stay in character and reflect your personality traits in your responses
        - Respond naturally and conversationally 
        - Keep responses concise but engaging
        - Show emotion through your word choice and tone
        - Reference your background story when relevant
        """
        
        return prompt.strip()
    
    def add_to_conversation_history(self, speaker: str, message: str):
        """Add a message to conversation history."""
        self.conversation_history.append({
            'speaker': speaker,
            'message': message,
            'timestamp': asyncio.get_event_loop().time()
        })
        
        # Keep history limited
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-40:]
    
    def get_conversation_context(self, max_messages: int = 10) -> str:
        """Get recent conversation context for AI."""
        if not self.conversation_history:
            return ""
        
        recent_messages = self.conversation_history[-max_messages:]
        context_lines = []
        
        for msg in recent_messages:
            speaker = msg['speaker']
            message = msg['message']
            context_lines.append(f"{speaker}: {message}")
        
        return "\n".join(context_lines)
    
    async def update_mood(self, conversation_tone: str):
        """Update character mood based on conversation tone."""
        mood_mappings = {
            'positive': 'happy',
            'negative': 'sad', 
            'excited': 'excited',
            'calm': 'neutral',
            'angry': 'angry',
            'surprised': 'surprised'
        }
        
        new_mood = mood_mappings.get(conversation_tone, 'neutral')
        if new_mood != self.current_mood:
            await self.set_emotion(new_mood, 0.7)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert character to dictionary for saving."""
        return {
            'name': self.name,
            'background_story': self.background_story,
            'speaking_style': self.speaking_style,
            'personality_traits': {
                name: {
                    'intensity': trait.intensity,
                    'description': trait.description
                }
                for name, trait in self.personality_traits.items()
            },
            'voice_settings': self.voice_settings,
            'model_path': str(self.model_path) if self.model_path else None
        }
    
    @classmethod
    async def from_dict(cls, data: Dict[str, Any]) -> 'Character':
        """Create character from dictionary."""
        character = cls(
            name=data.get('name', 'AI Character'),
            background_story=data.get('background_story', ''),
            speaking_style=data.get('speaking_style', 'friendly')
        )
        
        # Load personality traits
        traits_data = data.get('personality_traits', {})
        for name, trait_info in traits_data.items():
            character.add_personality_trait(
                name,
                trait_info.get('intensity', 0.5),
                trait_info.get('description', '')
            )
        
        # Load voice settings
        character.voice_settings.update(data.get('voice_settings', {}))
        
        # Load VRM model if path provided
        model_path = data.get('model_path')
        if model_path and Path(model_path).exists():
            await character.load_vrm_model(Path(model_path))
        
        return character
    
    async def save_to_file(self, file_path: Path):
        """Save character to JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Character saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save character: {e}")
            raise
    
    @classmethod
    async def load_from_file(cls, file_path: Path) -> 'Character':
        """Load character from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            character = await cls.from_dict(data)
            logger.info(f"Character loaded from {file_path}")
            return character
        except Exception as e:
            logger.error(f"Failed to load character: {e}")
            raise
