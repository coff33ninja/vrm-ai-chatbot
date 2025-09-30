"""
Google Cloud Speech-to-Text integration for VRM AI Chatbot.
Provides high-quality speech recognition using Google's STT service.
"""

import os
import asyncio
import tempfile
from typing import Optional, Dict, Any, List, Callable
import logging
from dataclasses import dataclass
from pathlib import Path
import threading

try:
    from google.cloud import speech
    import pyaudio
    import wave
    GOOGLE_STT_AVAILABLE = True
except ImportError:
    GOOGLE_STT_AVAILABLE = False

from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class GeminiSTTConfig:
    """Configuration for Google Cloud STT."""
    credentials_path: Optional[str] = None
    language_code: str = "en-US"
    sample_rate_hertz: int = 16000
    audio_channel_count: int = 1
    encoding: str = "LINEAR16"
    enable_automatic_punctuation: bool = True
    enable_word_time_offsets: bool = False
    model: str = "latest_long"

class GeminiSTT:
    """Google Cloud Speech-to-Text client."""
    
    def __init__(self, config: GeminiSTTConfig):
        """Initialize Google STT client."""
        if not GOOGLE_STT_AVAILABLE:
            raise ImportError("google-cloud-speech package not installed")
        
        self.config = config
        self.client = None
        self.audio = None
        self.stream = None
        self.is_listening = False
        self.temp_dir = Path(tempfile.gettempdir()) / "vrm_stt"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Set up credentials if provided
        if config.credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.credentials_path
        
        # Initialize client
        self._initialize_client()
        
        # Initialize PyAudio
        self._initialize_audio()
        
        logger.info("Google STT client initialized")
    
    def _initialize_client(self):
        """Initialize the Google STT client."""
        try:
            self.client = speech.SpeechClient()
            logger.info("Google STT client connected successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google STT client: {e}")
            raise
    
    def _initialize_audio(self):
        """Initialize PyAudio for recording."""
        try:
            self.audio = pyaudio.PyAudio()
            logger.info("PyAudio initialized for recording")
        except Exception as e:
            logger.error(f"Failed to initialize PyAudio: {e}")
            raise
    
    def get_audio_devices(self) -> List[Dict[str, Any]]:
        """Get available audio input devices."""
        try:
            if not self.audio:
                return []
            
            devices = []
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info["maxInputChannels"] > 0:
                    devices.append({
                        "index": i,
                        "name": device_info["name"],
                        "channels": device_info["maxInputChannels"],
                        "sample_rate": device_info["defaultSampleRate"]
                    })
            
            return devices
            
        except Exception as e:
            logger.error(f"Failed to get audio devices: {e}")
            return []
    
    async def transcribe_file(self, audio_file: str) -> Optional[str]:
        """Transcribe audio from file."""
        try:
            if not self.client:
                logger.error("Google STT client not initialized")
                return None
            
            # Read audio file
            with open(audio_file, "rb") as audio_file_obj:
                content = audio_file_obj.read()
            
            # Configure recognition
            audio = speech.RecognitionAudio(content=content)
            
            encoding = getattr(speech.RecognitionConfig.AudioEncoding, self.config.encoding)
            config = speech.RecognitionConfig(
                encoding=encoding,
                sample_rate_hertz=self.config.sample_rate_hertz,
                language_code=self.config.language_code,
                audio_channel_count=self.config.audio_channel_count,
                enable_automatic_punctuation=self.config.enable_automatic_punctuation,
                enable_word_time_offsets=self.config.enable_word_time_offsets,
                model=self.config.model
            )
            
            # Perform recognition
            response = await asyncio.to_thread(
                self.client.recognize,
                config=config,
                audio=audio
            )
            
            # Extract transcript
            if response.results:
                transcript = response.results[0].alternatives[0].transcript
                logger.info(f"Transcribed: {transcript}")
                return transcript
            else:
                logger.warning("No speech detected in audio file")
                return None
                
        except Exception as e:
            logger.error(f"Failed to transcribe file: {e}")
            return None
    
    def start_listening(self, callback: Callable[[str], None], device_index: Optional[int] = None):
        """Start continuous speech recognition."""
        try:
            if self.is_listening:
                logger.warning("Already listening")
                return
            
            self.is_listening = True
            
            # Start listening in a separate thread
            listen_thread = threading.Thread(
                target=self._listen_continuously,
                args=(callback, device_index),
                daemon=True
            )
            listen_thread.start()
            
            logger.info("Started continuous speech recognition")
            
        except Exception as e:
            logger.error(f"Failed to start listening: {e}")
    
    def _listen_continuously(self, callback: Callable[[str], None], device_index: Optional[int] = None):
        """Continuously listen for speech."""
        try:
            # Configure audio stream
            chunk = 1024
            format = pyaudio.paInt16
            channels = self.config.audio_channel_count
            rate = self.config.sample_rate_hertz
            
            # Open stream
            stream = self.audio.open(
                format=format,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=chunk
            )
            
            logger.info("Audio stream opened for continuous recognition")
            
            frames = []
            silence_threshold = 500  # Adjust based on your needs
            silence_duration = 0
            max_silence = 2  # seconds of silence before processing
            
            while self.is_listening:
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)
                
                # Simple silence detection (you might want to improve this)
                audio_data = int.from_bytes(data, byteorder='little', signed=True)
                if abs(audio_data) < silence_threshold:
                    silence_duration += chunk / rate
                else:
                    silence_duration = 0
                
                # Process audio when silence is detected
                if silence_duration >= max_silence and frames:
                    audio_file = self._save_audio_frames(frames, rate, channels)
                    if audio_file:
                        # Transcribe in background
                        asyncio.create_task(self._process_audio_async(audio_file, callback))
                    frames = []
                    silence_duration = 0
            
            stream.stop_stream()
            stream.close()
            logger.info("Audio stream closed")
            
        except Exception as e:
            logger.error(f"Error in continuous listening: {e}")
            self.is_listening = False
    
    def _save_audio_frames(self, frames: List[bytes], rate: int, channels: int) -> Optional[str]:
        """Save audio frames to a temporary file."""
        try:
            audio_file = str(self.temp_dir / f"recording_{hash(str(frames))}.wav")
            
            with wave.open(audio_file, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(rate)
                wf.writeframes(b''.join(frames))
            
            return audio_file
            
        except Exception as e:
            logger.error(f"Failed to save audio frames: {e}")
            return None
    
    async def _process_audio_async(self, audio_file: str, callback: Callable[[str], None]):
        """Process audio file asynchronously."""
        try:
            transcript = await self.transcribe_file(audio_file)
            if transcript and transcript.strip():
                callback(transcript)
            
            # Clean up temp file
            Path(audio_file).unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Failed to process audio: {e}")
    
    def stop_listening(self):
        """Stop continuous speech recognition."""
        self.is_listening = False
        logger.info("Stopped continuous speech recognition")
    
    def update_config(self, **kwargs):
        """Update STT configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        logger.info("Google STT configuration updated")
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.stop_listening()
            
            if self.audio:
                self.audio.terminate()
            
            # Clean up temp files
            for file in self.temp_dir.glob("recording_*.wav"):
                file.unlink()
            
            logger.info("Google STT resources cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup STT resources: {e}")

def create_gemini_stt(credentials_path: Optional[str] = None, **kwargs) -> Optional[GeminiSTT]:
    """Create a Google STT client with the given configuration."""
    if not GOOGLE_STT_AVAILABLE:
        logger.warning("Google STT not available - google-cloud-speech package not installed")
        return None
    
    try:
        config = GeminiSTTConfig(
            credentials_path=credentials_path,
            **kwargs
        )
        return GeminiSTT(config)
    except Exception as e:
        logger.error(f"Failed to create Google STT client: {e}")
        return None