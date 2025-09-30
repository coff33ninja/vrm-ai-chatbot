"""
Google Cloud Text-to-Speech integration for VRM AI Chatbot.
Provides high-quality voice synthesis using Google's TTS service.
"""

import os
import asyncio
import tempfile
from typing import Optional, Dict, Any, List
import logging
from dataclasses import dataclass
from pathlib import Path

try:
    from google.cloud import texttospeech
    import pygame
    GOOGLE_TTS_AVAILABLE = True
except ImportError:
    GOOGLE_TTS_AVAILABLE = False

from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class GeminiTTSConfig:
    """Configuration for Google Cloud TTS."""
    credentials_path: Optional[str] = None
    language_code: str = "en-US"
    voice_name: str = "en-US-Neural2-F"
    speaking_rate: float = 1.0
    pitch: float = 0.0
    volume_gain_db: float = 0.0
    audio_encoding: str = "MP3"

class GeminiTTS:
    """Google Cloud Text-to-Speech client."""
    
    def __init__(self, config: GeminiTTSConfig):
        """Initialize Google TTS client."""
        if not GOOGLE_TTS_AVAILABLE:
            raise ImportError("google-cloud-texttospeech package not installed")
        
        self.config = config
        self.client = None
        self.temp_dir = Path(tempfile.gettempdir()) / "vrm_tts"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Set up credentials if provided
        if config.credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.credentials_path
        
        # Initialize client
        self._initialize_client()
        
        # Initialize pygame mixer for audio playback
        try:
            pygame.mixer.init()
        except Exception as e:
            logger.warning(f"Failed to initialize pygame mixer: {e}")
        
        logger.info("Google TTS client initialized")
    
    def _initialize_client(self):
        """Initialize the Google TTS client."""
        try:
            self.client = texttospeech.TextToSpeechClient()
            logger.info("Google TTS client connected successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google TTS client: {e}")
            raise
    
    def get_available_voices(self, language_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get available voices for the specified language."""
        try:
            if not self.client:
                return []
            
            # List available voices
            voices = self.client.list_voices(language_code=language_code or self.config.language_code)
            
            voice_list = []
            for voice in voices.voices:
                voice_info = {
                    "name": voice.name,
                    "language_codes": list(voice.language_codes),
                    "ssml_gender": voice.ssml_gender.name,
                    "natural_sample_rate_hertz": voice.natural_sample_rate_hertz
                }
                voice_list.append(voice_info)
            
            return voice_list
            
        except Exception as e:
            logger.error(f"Failed to get available voices: {e}")
            return []
    
    async def synthesize_speech(self, text: str, output_file: Optional[str] = None) -> Optional[str]:
        """Synthesize speech from text."""
        try:
            if not self.client:
                logger.error("Google TTS client not initialized")
                return None
            
            # Set up synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Configure voice
            voice = texttospeech.VoiceSelectionParams(
                language_code=self.config.language_code,
                name=self.config.voice_name
            )
            
            # Configure audio
            audio_encoding = getattr(texttospeech.AudioEncoding, self.config.audio_encoding)
            audio_config = texttospeech.AudioConfig(
                audio_encoding=audio_encoding,
                speaking_rate=self.config.speaking_rate,
                pitch=self.config.pitch,
                volume_gain_db=self.config.volume_gain_db
            )
            
            # Perform synthesis
            response = await asyncio.to_thread(
                self.client.synthesize_speech,
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            # Save audio to file
            if not output_file:
                output_file = str(self.temp_dir / f"tts_{hash(text)}.mp3")
            
            with open(output_file, "wb") as out:
                out.write(response.audio_content)
            
            logger.info(f"Speech synthesized and saved to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to synthesize speech: {e}")
            return None
    
    async def speak(self, text: str) -> bool:
        """Synthesize and play speech."""
        try:
            # Synthesize speech
            audio_file = await self.synthesize_speech(text)
            if not audio_file:
                return False
            
            # Play audio
            return await self.play_audio(audio_file)
            
        except Exception as e:
            logger.error(f"Failed to speak text: {e}")
            return False
    
    async def play_audio(self, audio_file: str) -> bool:
        """Play audio file."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
            # Load and play audio
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
            
            logger.info(f"Audio playback completed: {audio_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            return False
    
    def stop_playback(self):
        """Stop current audio playback."""
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
            logger.info("Audio playback stopped")
        except Exception as e:
            logger.error(f"Failed to stop playback: {e}")
    
    def update_config(self, **kwargs):
        """Update TTS configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        logger.info("Google TTS configuration updated")
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            for file in self.temp_dir.glob("tts_*.mp3"):
                file.unlink()
            logger.info("TTS temporary files cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")

def create_gemini_tts(credentials_path: Optional[str] = None, **kwargs) -> Optional[GeminiTTS]:
    """Create a Google TTS client with the given configuration."""
    if not GOOGLE_TTS_AVAILABLE:
        logger.warning("Google TTS not available - google-cloud-texttospeech package not installed")
        return None
    
    try:
        config = GeminiTTSConfig(
            credentials_path=credentials_path,
            **kwargs
        )
        return GeminiTTS(config)
    except Exception as e:
        logger.error(f"Failed to create Google TTS client: {e}")
        return None