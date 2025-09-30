"""
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
