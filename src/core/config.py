"""
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
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"
    azure_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    use_local_ai: bool = True
    temperature: float = 0.7
    max_tokens: int = 1000

class VoiceConfig(BaseModel):
    """Voice synthesis and recognition configuration."""
    tts_engine: str = "pyttsx3"  # pyttsx3, azure, openai, gemini
    tts_voice: str = "default"
    tts_rate: int = 200
    tts_volume: float = 0.9
    
    stt_engine: str = "whisper"  # whisper, azure, gemini
    stt_language: str = "en-US"
    stt_timeout: float = 5.0
    # Whisper model selection (e.g., tiny, base, small, medium, large)
    whisper_model: str = "base"
    
    # Azure Speech Services
    azure_speech_key: Optional[str] = None
    azure_speech_region: Optional[str] = None
    
    # Google Cloud Services
    google_credentials_path: Optional[str] = None
    language_code: str = "en-US"

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
    if os.getenv('GEMINI_API_KEY'):
        env_overrides.setdefault('ai', {})['gemini_api_key'] = os.getenv('GEMINI_API_KEY')
    
    # Voice Configuration  
    if os.getenv('AZURE_SPEECH_KEY'):
        env_overrides.setdefault('voice', {})['azure_speech_key'] = os.getenv('AZURE_SPEECH_KEY')
    if os.getenv('AZURE_SPEECH_REGION'):
        env_overrides.setdefault('voice', {})['azure_speech_region'] = os.getenv('AZURE_SPEECH_REGION')
    
    # Google Cloud Configuration
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        env_overrides.setdefault('voice', {})['google_credentials_path'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
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
