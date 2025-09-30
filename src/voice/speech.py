"""
Speech recognition manager for VRM AI Chatbot.
Handles speech-to-text functionality and manages audio input.
"""

import asyncio
import logging
from typing import Optional, Dict, Any

from .gemini_stt import GeminiSTT, GeminiSTTConfig
from ..core.config import VoiceConfig
from ..utils.event_bus import EventBus

# Optional Whisper integration (local OpenAI whisper)
try:
    import whisper
    _WHISPER_AVAILABLE = True
except Exception:
    whisper = None
    _WHISPER_AVAILABLE = False

logger = logging.getLogger(__name__)


class SpeechManager:
    """
    Manages speech recognition and audio input.
    Coordinates between STT providers and the event bus.
    """
    
    def __init__(self, config: VoiceConfig, event_bus: EventBus):
        """
        Initialize the speech manager.
        
        Args:
            config: Voice configuration
            event_bus: Event bus for inter-component communication
        """
        self.config = config
        self.event_bus = event_bus
        self.stt_client: Optional[GeminiSTT] = None
        self.is_listening = False
        self.is_initialized = False
        # Whisper model instance (loaded lazily if requested)
        self._whisper_model = None

        logger.info("Speech Manager created")
    
    async def initialize(self):
        """Initialize the speech recognition system."""
        try:
            # Check if STT is configured
            if self.config.stt_engine == "none":
                logger.info("Speech recognition disabled in configuration")
                return
            
            # Initialize STT client based on provider
            if self.config.stt_engine == "gemini":
                self.stt_client = await self._initialize_gemini_stt()
            elif self.config.stt_engine == "whisper":
                # Initialize local Whisper model for non-realtime file transcription
                if not _WHISPER_AVAILABLE:
                    logger.error("Whisper library not available. Install 'openai-whisper' (pip install openai-whisper) to enable Whisper STT.")
                    return
                try:
                    loop = asyncio.get_event_loop()
                    model_name = getattr(self.config, 'whisper_model', 'base')
                    logger.info(f"Config whisper_model='{model_name}' requested")

                    def _load():
                        return whisper.load_model(model_name)

                    # Load model in thread pool to avoid blocking
                    self._whisper_model = await loop.run_in_executor(None, _load)
                    logger.info(f"Whisper model '{model_name}' loaded")
                    self.is_initialized = True
                    return
                except Exception as e:
                    logger.error(f"Failed to load Whisper model: {e}")
                    return
            else:
                logger.warning(f"Unknown STT engine: {self.config.stt_engine}")
                return
            
            if not self.stt_client:
                logger.warning("Failed to initialize STT client")
                return
            
            self.is_initialized = True
            logger.info("Speech Manager initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize Speech Manager: {e}", exc_info=True)
            # Don't raise - allow the app to continue without speech recognition
    
    async def _initialize_gemini_stt(self) -> Optional[GeminiSTT]:
        """Initialize Google Cloud STT client."""
        try:
            stt_config = GeminiSTTConfig(
                credentials_path=self.config.google_credentials_path,
                language_code=self.config.stt_language,
                sample_rate_hertz=16000,
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
                device_index=None
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
            # If Whisper is configured and model is available, use it for file transcription
            if self.config.stt_engine == 'whisper':
                if not _WHISPER_AVAILABLE or not self._whisper_model:
                    logger.error('Whisper model not available for transcription')
                    return None

                try:
                    loop = asyncio.get_event_loop()

                    def _transcribe():
                        result = self._whisper_model.transcribe(audio_file)
                        return result.get('text', '').strip()

                    text = await loop.run_in_executor(None, _transcribe)
                    return text
                except Exception as e:
                    logger.error(f"Whisper transcription failed: {e}")
                    return None

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
