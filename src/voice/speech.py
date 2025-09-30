"""
Speech recognition manager for VRM AI Chatbot.
Handles speech-to-text functionality and manages audio input.
"""

import asyncio
import logging
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

from .gemini_stt import GeminiSTT, GeminiSTTConfig, create_gemini_stt
from ..utils.event_bus import EventBus

logger = logging.getLogger(__name__)


@dataclass
class SpeechConfig:
    """Configuration for speech recognition."""
    enabled: bool = True
    provider: str = "gemini"  # "gemini" or other providers
    language_code: str = "en-US"
    sample_rate_hertz: int = 16000
    credentials_path: Optional[str] = None
    auto_start: bool = False
    device_index: Optional[int] = None
    enable_continuous: bool = True


class SpeechManager:
    """
    Manages speech recognition and audio input.
    Coordinates between STT providers and the event bus.
    """
    
    def __init__(self, config: SpeechConfig, event_bus: EventBus):
        """
        Initialize the speech manager.
        
        Args:
            config: Speech configuration
            event_bus: Event bus for inter-component communication
        """
        self.config = config
        self.event_bus = event_bus
        self.stt_client: Optional[GeminiSTT] = None
        self.is_listening = False
        self.is_initialized = False
        
        logger.info("Speech Manager created")
    
    async def initialize(self):
        """Initialize the speech recognition system."""
        try:
            if not self.config.enabled:
                logger.info("Speech recognition disabled in configuration")
                return
            
            # Initialize STT client based on provider
            if self.config.provider == "gemini":
                self.stt_client = await self._initialize_gemini_stt()
            else:
                logger.warning(f"Unknown STT provider: {self.config.provider}")
                return
            
            if not self.stt_client:
                logger.warning("Failed to initialize STT client")
                return
            
            self.is_initialized = True
            logger.info("Speech Manager initialized successfully")
            
            # Auto-start listening if configured
            if self.config.auto_start:
                await self.start_listening()
                
        except Exception as e:
            logger.error(f"Failed to initialize Speech Manager: {e}", exc_info=True)
            raise
    
    async def _initialize_gemini_stt(self) -> Optional[GeminiSTT]:
        """Initialize Google Cloud STT client."""
        try:
            stt_config = GeminiSTTConfig(
                credentials_path=self.config.credentials_path,
                language_code=self.config.language_code,
                sample_rate_hertz=self.config.sample_rate_hertz,
                enable_automatic_punctuation=True,
                model="latest_long"
            )
            
            stt_client = GeminiSTT(stt_config)
            logger.info("Gemini STT client initialized")
            return stt_client
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini STT: {e}")
            return None
    
    async def start_listening(self):
        """Start continuous speech recognition."""
        try:
            if not self.is_initialized:
                logger.warning("Speech Manager not initialized")
                return
            
            if self.is_listening:
                logger.warning("Already listening")
                return
            
            if not self.stt_client:
                logger.error("No STT client available")
                return
            
            # Start listening with callback
            self.stt_client.start_listening(
                callback=self._on_speech_recognized,
                device_index=self.config.device_index
            )
            
            self.is_listening = True
            logger.info("Started speech recognition")
            
            # Publish event
            await self.event_bus.publish("speech_listening_started", {})
            
        except Exception as e:
            logger.error(f"Failed to start listening: {e}")
    
    async def stop_listening(self):
        """Stop continuous speech recognition."""
        try:
            if not self.is_listening:
                return
            
            if self.stt_client:
                self.stt_client.stop_listening()
            
            self.is_listening = False
            logger.info("Stopped speech recognition")
            
            # Publish event
            await self.event_bus.publish("speech_listening_stopped", {})
            
        except Exception as e:
            logger.error(f"Failed to stop listening: {e}")
    
    def _on_speech_recognized(self, text: str):
        """
        Callback for when speech is recognized.
        
        Args:
            text: Recognized speech text
        """
        logger.info(f"Speech recognized: {text}")
        
        # Publish to event bus (run in event loop)
        asyncio.create_task(self._publish_speech_event(text))
    
    async def _publish_speech_event(self, text: str):
        """Publish speech recognized event to the event bus."""
        try:
            await self.event_bus.publish("speech_recognized", text)
        except Exception as e:
            logger.error(f"Failed to publish speech event: {e}")
    
    async def transcribe_file(self, audio_file: str) -> Optional[str]:
        """
        Transcribe audio from a file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcribed text or None
        """
        try:
            if not self.stt_client:
                logger.error("No STT client available")
                return None
            
            return await self.stt_client.transcribe_file(audio_file)
            
        except Exception as e:
            logger.error(f"Failed to transcribe file: {e}")
            return None
    
    def get_audio_devices(self) -> list[Dict[str, Any]]:
        """
        Get available audio input devices.
        
        Returns:
            List of audio device information
        """
        try:
            if not self.stt_client:
                return []
            
            return self.stt_client.get_audio_devices()
            
        except Exception as e:
            logger.error(f"Failed to get audio devices: {e}")
            return []
    
    async def set_audio_device(self, device_index: int):
        """
        Set the audio input device.
        
        Args:
            device_index: Index of the audio device to use
        """
        try:
            self.config.device_index = device_index
            
            # Restart listening if currently active
            if self.is_listening:
                await self.stop_listening()
                await self.start_listening()
            
            logger.info(f"Audio device set to index: {device_index}")
            
        except Exception as e:
            logger.error(f"Failed to set audio device: {e}")
    
    async def toggle_listening(self):
        """Toggle speech recognition on/off."""
        if self.is_listening:
            await self.stop_listening()
        else:
            await self.start_listening()
    
    async def shutdown(self):
        """Shutdown the speech manager."""
        try:
            logger.info("Shutting down Speech Manager...")
            
            # Stop listening
            await self.stop_listening()
            
            # Cleanup STT client
            if self.stt_client:
                self.stt_client.cleanup()
            
            self.is_initialized = False
            logger.info("Speech Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Speech Manager shutdown: {e}")
