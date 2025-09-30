# Create the main application structure and core scripts

import os
from pathlib import Path

# First, let's create the requirements.txt file
requirements = """# Core Python Packages
numpy>=1.24.0
pillow>=10.0.0
asyncio-extensions>=0.1.0

# 3D Graphics and Rendering
PyOpenGL>=3.1.7
moderngl>=5.8.0
pyrr>=0.10.3  # 3D math utilities
glfw>=2.6.0   # Window management

# VRM and 3D Model Support
pygltflib>=1.16.0
trimesh>=4.0.0
scipy>=1.11.0

# GUI and Window Management
tkinter  # Built into Python
pywin32>=306; sys_platform=="win32"
pillow>=10.0.0

# AI and Language Models
openai>=1.0.0
anthropic>=0.5.0
ollama>=0.1.0
langchain>=0.1.0
langchain-openai>=0.0.5

# Voice and Audio
pyttsx3>=2.90
SpeechRecognition>=3.10.0
pyaudio>=0.2.11
azure-cognitiveservices-speech>=1.34.0
openai-whisper>=20231117
sounddevice>=0.4.6
librosa>=0.10.0

# Real-time Communication  
livekit>=0.10.0
livekit-api>=0.5.0
websockets>=12.0
aiohttp>=3.9.0

# System Integration
psutil>=5.9.0
pyautogui>=0.9.54
keyboard>=0.13.5
mouse>=0.7.1

# Configuration and Utilities
pydantic>=2.5.0
python-dotenv>=1.0.0
PyYAML>=6.0.1
rich>=13.7.0
typer>=0.9.0
watchdog>=3.0.0

# Development and Testing
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.7.0
"""

with open('requirements.txt', 'w', encoding='utf-8') as f:
    f.write(requirements)

print("✅ Created requirements.txt with all necessary dependencies")

# Create the main application entry point
main_app = '''#!/usr/bin/env python3
"""
VRM AI Chatbot - Main Application Entry Point
A desktop AI companion with VRM avatar support, voice interaction, and system integration.

Features:
- 3D VRM avatar rendering with real-time animation
- Transparent desktop overlay window
- Local and cloud AI integration
- Voice synthesis and speech recognition
- LiveKit video calling support
- Windows system integration
- Customizable personality traits

Author: VRM AI Chatbot Project
Version: 1.0.0
Python: 3.11+
"""

import sys
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import core modules
from src.core.application import VRMAIApplication
from src.core.config import Config, load_config
from src.utils.logger import setup_logging
from src.gui.splash import SplashScreen

def check_python_version():
    """Ensure compatible Python version."""
    if sys.version_info < (3, 11):
        print("❌ Python 3.11 or higher is required!")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'numpy', 'OpenGL', 'moderngl', 'tkinter', 'pyttsx3', 
        'speech_recognition', 'openai', 'livekit', 'pygltflib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'OpenGL':
                import OpenGL.GL
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\\nPlease install missing packages with:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    print("✅ All required dependencies are installed")

def main():
    """Main application entry point."""
    print("🤖 VRM AI Chatbot - Starting Application...")
    print("=" * 50)
    
    # Check system requirements
    check_python_version()
    check_dependencies()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting VRM AI Chatbot Application")
    
    try:
        # Load configuration
        config = load_config()
        logger.info(f"Configuration loaded: {config.app_name}")
        
        # Show splash screen
        splash = SplashScreen()
        splash.show()
        
        # Initialize and run the main application
        app = VRMAIApplication(config)
        
        # Start the application event loop
        asyncio.run(app.run())
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"❌ Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(main_app)

print("✅ Created main.py - Application entry point")

# Create project structure
directories = [
    'src',
    'src/core',
    'src/ai',
    'src/voice',
    'src/graphics',
    'src/gui',
    'src/models',
    'src/utils',
    'src/integrations',
    'assets',
    'assets/models',
    'assets/voices',
    'assets/textures',
    'assets/shaders',
    'configs',
    'data',
    'logs',
    'tests'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    # Create __init__.py files for Python packages
    if directory.startswith('src'):
        with open(f'{directory}/__init__.py', 'w', encoding='utf-8') as f:
            f.write('"""VRM AI Chatbot package."""\\n')

print("✅ Created project directory structure")

# Create configuration system
config_py = '''"""
Configuration management for VRM AI Chatbot.
Handles loading and validation of application settings.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AIConfig(BaseModel):
    """AI service configuration."""
    local_model: str = "llama3.1:8b"
    local_api_url: str = "http://localhost:11434"
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    use_local_ai: bool = True
    temperature: float = 0.7
    max_tokens: int = 1000

class VoiceConfig(BaseModel):
    """Voice synthesis and recognition configuration."""
    tts_engine: str = "pyttsx3"  # pyttsx3, azure, openai, google
    tts_voice: str = "default"
    tts_rate: int = 200
    tts_volume: float = 0.9
    
    stt_engine: str = "whisper"  # whisper, azure, google
    stt_language: str = "en-US"
    stt_timeout: float = 5.0
    
    # Azure Speech Services
    azure_speech_key: Optional[str] = None
    azure_speech_region: Optional[str] = None

class GraphicsConfig(BaseModel):
    """3D graphics and rendering configuration."""
    window_width: int = 800
    window_height: int = 600
    target_fps: int = 60
    vsync: bool = True
    antialiasing: bool = True
    transparency: float = 0.8
    
    # VRM model settings
    default_model: str = "assets/models/default_character.vrm"
    animation_speed: float = 1.0
    lip_sync_enabled: bool = True
    eye_tracking_enabled: bool = True

class PersonalityConfig(BaseModel):
    """AI personality configuration."""
    name: str = "Aria"
    personality_traits: List[str] = [
        "friendly", "helpful", "curious", "empathetic"
    ]
    background_story: str = "I'm your AI companion, here to help and chat!"
    speaking_style: str = "casual and warm"
    interests: List[str] = ["technology", "art", "music", "science"]
    
    # Custom personality file
    personality_file: Optional[str] = None

class SystemConfig(BaseModel):
    """System integration configuration."""
    enable_system_integration: bool = True
    startup_with_windows: bool = False
    minimize_to_tray: bool = True
    hotkey_toggle: str = "ctrl+shift+a"
    
    # Window behavior
    always_on_top: bool = True
    click_through: bool = False
    follow_mouse: bool = False

class LiveKitConfig(BaseModel):
    """LiveKit video calling configuration."""
    enabled: bool = False
    server_url: str = "wss://your-livekit-server.com"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    room_name: str = "vrm-ai-room"

class Config(BaseModel):
    """Main application configuration."""
    app_name: str = "VRM AI Chatbot"
    version: str = "1.0.0"
    debug: bool = False
    data_dir: Path = Path("data")
    log_level: str = "INFO"
    
    # Component configurations
    ai: AIConfig = Field(default_factory=AIConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    graphics: GraphicsConfig = Field(default_factory=GraphicsConfig)
    personality: PersonalityConfig = Field(default_factory=PersonalityConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    livekit: LiveKitConfig = Field(default_factory=LiveKitConfig)
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"

def load_config(config_file: Optional[str] = None) -> Config:
    """Load configuration from file or environment variables."""
    
    # Default config file path
    if config_file is None:
        config_file = "configs/config.yaml"
    
    config_path = Path(config_file)
    
    # Load from YAML if exists
    config_data = {}
    if config_path.exists() and config_path.suffix in ['.yaml', '.yml']:
        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
    
    # Override with environment variables
    env_overrides = {}
    
    # AI Configuration
    if os.getenv('OPENAI_API_KEY'):
        env_overrides.setdefault('ai', {})['openai_api_key'] = os.getenv('OPENAI_API_KEY')
    if os.getenv('ANTHROPIC_API_KEY'):
        env_overrides.setdefault('ai', {})['anthropic_api_key'] = os.getenv('ANTHROPIC_API_KEY')
    
    # Voice Configuration  
    if os.getenv('AZURE_SPEECH_KEY'):
        env_overrides.setdefault('voice', {})['azure_speech_key'] = os.getenv('AZURE_SPEECH_KEY')
    if os.getenv('AZURE_SPEECH_REGION'):
        env_overrides.setdefault('voice', {})['azure_speech_region'] = os.getenv('AZURE_SPEECH_REGION')
    
    # LiveKit Configuration
    if os.getenv('LIVEKIT_API_KEY'):
        env_overrides.setdefault('livekit', {})['api_key'] = os.getenv('LIVEKIT_API_KEY')
    if os.getenv('LIVEKIT_API_SECRET'):
        env_overrides.setdefault('livekit', {})['api_secret'] = os.getenv('LIVEKIT_API_SECRET')
    if os.getenv('LIVEKIT_SERVER_URL'):
        env_overrides.setdefault('livekit', {})['server_url'] = os.getenv('LIVEKIT_SERVER_URL')
    
    # Merge configurations
    def deep_merge(base: dict, override: dict) -> dict:
        """Deep merge two dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    final_config = deep_merge(config_data, env_overrides)
    
    # Create and return Config object
    return Config(**final_config)

def save_config(config: Config, config_file: str = "configs/config.yaml"):
    """Save configuration to YAML file."""
    import yaml
    
    config_path = Path(config_file)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict and save
    config_dict = config.dict()
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
'''

with open('src/core/config.py', 'w', encoding='utf-8') as f:
    f.write(config_py)

print("✅ Created configuration system")

# Create the main application class
app_py = '''"""
Main application class for VRM AI Chatbot.
Coordinates all components and manages the main application loop.
"""

import asyncio
import logging
import signal
import sys
from typing import Optional
from pathlib import Path

from .config import Config
from ..graphics.renderer import VRMRenderer
from ..graphics.window import TransparentWindow
from ..ai.conversation import ConversationManager
from ..voice.speech import SpeechManager
from ..voice.synthesis import VoiceSynthesis
from ..integrations.system import SystemIntegration
from ..integrations.livekit_client import LiveKitClient
from ..models.character import Character
from ..utils.event_bus import EventBus

logger = logging.getLogger(__name__)

class VRMAIApplication:
    """Main application class that orchestrates all components."""
    
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        self.event_bus = EventBus()
        
        # Core components
        self.window: Optional[TransparentWindow] = None
        self.renderer: Optional[VRMRenderer] = None
        self.character: Optional[Character] = None
        self.conversation_manager: Optional[ConversationManager] = None
        self.speech_manager: Optional[SpeechManager] = None
        self.voice_synthesis: Optional[VoiceSynthesis] = None
        self.system_integration: Optional[SystemIntegration] = None
        self.livekit_client: Optional[LiveKitClient] = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("VRM AI Application initialized")
    
    async def initialize(self):
        """Initialize all application components."""
        logger.info("Initializing application components...")
        
        try:
            # Initialize event bus
            await self.event_bus.initialize()
            
            # Initialize transparent window
            self.window = TransparentWindow(
                width=self.config.graphics.window_width,
                height=self.config.graphics.window_height,
                transparency=self.config.graphics.transparency,
                always_on_top=self.config.system.always_on_top,
                click_through=self.config.system.click_through
            )
            await self.window.initialize()
            
            # Initialize 3D renderer
            self.renderer = VRMRenderer(
                window=self.window,
                target_fps=self.config.graphics.target_fps,
                vsync=self.config.graphics.vsync,
                antialiasing=self.config.graphics.antialiasing
            )
            await self.renderer.initialize()
            
            # Load character
            self.character = Character(
                name=self.config.personality.name,
                personality_traits=self.config.personality.personality_traits,
                background_story=self.config.personality.background_story
            )
            
            # Load VRM model
            model_path = Path(self.config.graphics.default_model)
            if model_path.exists():
                await self.character.load_vrm_model(model_path)
                await self.renderer.load_character(self.character)
            else:
                logger.warning(f"Default VRM model not found: {model_path}")
            
            # Initialize AI conversation manager
            self.conversation_manager = ConversationManager(
                config=self.config.ai,
                character=self.character,
                event_bus=self.event_bus
            )
            await self.conversation_manager.initialize()
            
            # Initialize voice synthesis
            self.voice_synthesis = VoiceSynthesis(
                config=self.config.voice,
                event_bus=self.event_bus
            )
            await self.voice_synthesis.initialize()
            
            # Initialize speech recognition
            self.speech_manager = SpeechManager(
                config=self.config.voice,
                event_bus=self.event_bus
            )
            await self.speech_manager.initialize()
            
            # Initialize system integration
            if self.config.system.enable_system_integration:
                self.system_integration = SystemIntegration(
                    config=self.config.system,
                    event_bus=self.event_bus
                )
                await self.system_integration.initialize()
            
            # Initialize LiveKit client
            if self.config.livekit.enabled:
                self.livekit_client = LiveKitClient(
                    config=self.config.livekit,
                    event_bus=self.event_bus
                )
                await self.livekit_client.initialize()
            
            # Setup event handlers
            self._setup_event_handlers()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}", exc_info=True)
            raise
    
    def _setup_event_handlers(self):
        """Setup event handlers for inter-component communication."""
        
        # Speech recognition -> Conversation
        self.event_bus.subscribe("speech_recognized", self._handle_speech_input)
        
        # Conversation -> Voice synthesis
        self.event_bus.subscribe("ai_response", self._handle_ai_response)
        
        # Voice synthesis -> Animation
        self.event_bus.subscribe("speech_started", self._handle_speech_animation)
        self.event_bus.subscribe("speech_ended", self._handle_speech_end)
        
        # System events
        self.event_bus.subscribe("window_closed", self._handle_window_close)
        self.event_bus.subscribe("hotkey_pressed", self._handle_hotkey)
        
        logger.info("Event handlers configured")
    
    async def _handle_speech_input(self, text: str):
        """Handle recognized speech input."""
        logger.info(f"Speech input: {text}")
        if self.conversation_manager:
            await self.conversation_manager.process_input(text)
    
    async def _handle_ai_response(self, response: str):
        """Handle AI response for voice synthesis."""
        logger.info(f"AI response: {response[:100]}...")
        if self.voice_synthesis:
            await self.voice_synthesis.speak(response)
    
    async def _handle_speech_animation(self, text: str):
        """Handle speech animation start."""
        if self.character and self.renderer:
            await self.character.start_speaking_animation()
            await self.renderer.update_character()
    
    async def _handle_speech_end(self):
        """Handle speech animation end."""
        if self.character and self.renderer:
            await self.character.stop_speaking_animation()
            await self.renderer.update_character()
    
    async def _handle_window_close(self):
        """Handle window close event."""
        logger.info("Window close requested")
        await self.shutdown()
    
    async def _handle_hotkey(self, hotkey: str):
        """Handle global hotkey press."""
        logger.info(f"Hotkey pressed: {hotkey}")
        # Toggle visibility or other actions
        if self.window:
            await self.window.toggle_visibility()
    
    async def run(self):
        """Main application run loop."""
        try:
            # Initialize all components
            await self.initialize()
            
            self.running = True
            logger.info("Starting main application loop")
            
            # Main event loop
            while self.running:
                try:
                    # Update renderer
                    if self.renderer:
                        await self.renderer.render_frame()
                    
                    # Process window events
                    if self.window:
                        await self.window.process_events()
                    
                    # Small delay to prevent excessive CPU usage
                    await asyncio.sleep(1/60)  # ~60 FPS
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    # Continue running unless it's a critical error
                    if not self.running:
                        break
            
        except Exception as e:
            logger.error(f"Application error: {e}", exc_info=True)
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the application gracefully."""
        if not self.running:
            return
            
        logger.info("Shutting down application...")
        self.running = False
        
        # Shutdown components in reverse order
        components = [
            self.livekit_client,
            self.system_integration,
            self.speech_manager,
            self.voice_synthesis,
            self.conversation_manager,
            self.renderer,
            self.window,
            self.event_bus
        ]
        
        for component in components:
            if component:
                try:
                    await component.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down component: {e}")
        
        logger.info("Application shutdown complete")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {signum}")
        self.running = False
'''

with open('src/core/application.py', 'w', encoding='utf-8') as f:
    f.write(app_py)

print("✅ Created main application class")

# Create additional essential files
files_created = [
    'requirements.txt',
    'main.py', 
    'src/core/config.py',
    'src/core/application.py'
]

print(f"\\n📁 PROJECT STRUCTURE CREATED:")
print(f"   Root: {os.getcwd()}")
print(f"   Files: {len(files_created)} core files created")
print(f"   Directories: {len(directories)} directories created")
print(f"\\n🚀 NEXT STEPS:")
print(f"   1. Install dependencies: pip install -r requirements.txt")
print(f"   2. Configure settings in configs/config.yaml")
print(f"   3. Add VRM models to assets/models/")
print(f"   4. Set API keys in .env file")
print(f"   5. Run: python main.py")