"""
Voice Synthesis System - Handles text-to-speech with multiple backends.
Supports local engines (pyttsx3) and cloud services (Azure, OpenAI, Google).
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List, Callable
import threading
import queue
from pathlib import Path
import tempfile
import os

import pyttsx3

# Optional cloud TTS imports
try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    speechsdk = None

try:
    import openai
except ImportError:
    openai = None

try:
    from .gemini_tts import create_gemini_tts, GeminiTTS
except ImportError:
    create_gemini_tts = None
    GeminiTTS = None

from ..core.config import VoiceConfig
from ..utils.event_bus import EventBus

logger = logging.getLogger(__name__)

class VoiceSynthesis:
    """Advanced voice synthesis with multiple TTS backends."""
    
    def __init__(self, config: VoiceConfig, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        
        # TTS engines
        self.pyttsx3_engine: Optional[pyttsx3.Engine] = None
        self.azure_synthesizer: Optional[Any] = None
        self.openai_client: Optional[Any] = None
        self.gemini_tts: Optional[GeminiTTS] = None
        
        # Audio playback
        self.audio_queue = queue.Queue()
        self.playback_thread: Optional[threading.Thread] = None
        self.is_speaking = False
        self.should_stop = False
        
        # Voice settings
        self.current_voice_id = None
        self.available_voices: List[Dict[str, Any]] = []
        
        # Callbacks
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable] = None
        self.on_phoneme: Optional[Callable] = None
        
        logger.info(f"Voice synthesis initialized with engine: {config.tts_engine}")
    
    async def initialize(self):
        """Initialize the selected TTS engine."""
        try:
            if self.config.tts_engine == "pyttsx3":
                await self._init_pyttsx3()
            elif self.config.tts_engine == "azure":
                await self._init_azure_tts()
            elif self.config.tts_engine == "openai":
                await self._init_openai_tts()
            elif self.config.tts_engine == "gemini":
                await self._init_gemini_tts()
            else:
                logger.warning(f"Unknown TTS engine: {self.config.tts_engine}")
                await self._init_pyttsx3()  # Fallback
            
            # Start audio playback thread
            self.playback_thread = threading.Thread(target=self._audio_playback_loop, daemon=True)
            self.playback_thread.start()
            
            logger.info("Voice synthesis initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice synthesis: {e}")
            raise
    
    async def _init_pyttsx3(self):
        """Initialize pyttsx3 text-to-speech engine."""
        try:
            self.pyttsx3_engine = pyttsx3.init()
            
            # Configure voice settings
            voices = self.pyttsx3_engine.getProperty('voices')
            self.available_voices = []
            
            for i, voice in enumerate(voices):
                voice_info = {
                    'id': voice.id,
                    'name': voice.name,
                    'gender': 'male' if 'male' in voice.name.lower() else 'female',
                    'language': getattr(voice, 'languages', ['en-US'])[0] if hasattr(voice, 'languages') else 'en-US'
                }
                self.available_voices.append(voice_info)
            
            # Set default voice
            if self.config.tts_voice != "default" and len(voices) > 1:
                for voice in voices:
                    if self.config.tts_voice.lower() in voice.name.lower():
                        self.pyttsx3_engine.setProperty('voice', voice.id)
                        self.current_voice_id = voice.id
                        break
            
            # Set rate and volume
            self.pyttsx3_engine.setProperty('rate', self.config.tts_rate)
            self.pyttsx3_engine.setProperty('volume', self.config.tts_volume)
            
            logger.info(f"pyttsx3 initialized with {len(self.available_voices)} voices")
            
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
            raise
    
    async def _init_azure_tts(self):
        """Initialize Azure Cognitive Services Speech."""
        if not speechsdk:
            raise ImportError("Azure Speech SDK not installed")
        
        if not self.config.azure_speech_key or not self.config.azure_speech_region:
            raise ValueError("Azure Speech key and region required")
        
        try:
            # Create speech config
            speech_config = speechsdk.SpeechConfig(
                subscription=self.config.azure_speech_key,
                region=self.config.azure_speech_region
            )
            
            # Set voice
            if self.config.tts_voice != "default":
                speech_config.speech_synthesis_voice_name = self.config.tts_voice
            else:
                speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
            
            # Create synthesizer
            self.azure_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
            
            # Get available voices
            voices_result = await self._get_azure_voices()
            if voices_result:
                self.available_voices = voices_result
            
            logger.info("Azure TTS initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure TTS: {e}")
            raise
    
    async def _init_openai_tts(self):
        """Initialize OpenAI Text-to-Speech."""
        if not openai:
            raise ImportError("OpenAI library not installed")
        
        try:
            # Initialize OpenAI client
            self.openai_client = openai.OpenAI()
            
            # Available OpenAI voices
            self.available_voices = [
                {'id': 'alloy', 'name': 'Alloy', 'gender': 'neutral'},
                {'id': 'echo', 'name': 'Echo', 'gender': 'male'},
                {'id': 'fable', 'name': 'Fable', 'gender': 'neutral'},
                {'id': 'onyx', 'name': 'Onyx', 'gender': 'male'},
                {'id': 'nova', 'name': 'Nova', 'gender': 'female'},
                {'id': 'shimmer', 'name': 'Shimmer', 'gender': 'female'},
            ]
            
            # Set default voice
            if self.config.tts_voice in [v['id'] for v in self.available_voices]:
                self.current_voice_id = self.config.tts_voice
            else:
                self.current_voice_id = 'nova'  # Default female voice
            
            logger.info("OpenAI TTS initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI TTS: {e}")
            raise
    
    async def _init_gemini_tts(self):
        """Initialize Google Cloud Text-to-Speech."""
        if not create_gemini_tts:
            raise ImportError("Google Cloud TTS not available")
        
        try:
            # Get credentials path from config
            credentials_path = getattr(self.config, 'google_credentials_path', None)
            
            # Initialize Gemini TTS
            self.gemini_tts = create_gemini_tts(
                credentials_path=credentials_path,
                language_code=getattr(self.config, 'language_code', 'en-US'),
                voice_name=self.config.tts_voice if self.config.tts_voice != "default" else "en-US-Neural2-F",
                speaking_rate=self.config.tts_rate / 200.0,  # Convert to 0.25-4.0 range
                volume_gain_db=0.0
            )
            
            if self.gemini_tts:
                # Get available voices
                self.available_voices = self.gemini_tts.get_available_voices()
                self.current_voice_id = self.config.tts_voice if self.config.tts_voice != "default" else "en-US-Neural2-F"
                
                logger.info("Google Cloud TTS initialized successfully")
            else:
                raise Exception("Failed to create Gemini TTS client")
                
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud TTS: {e}")
            raise
    
    async def _get_azure_voices(self) -> List[Dict[str, Any]]:
        """Get available Azure voices."""
        try:
            # This would query Azure for available voices
            # For now, return common voices
            return [
                {'id': 'en-US-AriaNeural', 'name': 'Aria (US English)', 'gender': 'female', 'language': 'en-US'},
                {'id': 'en-US-JennyNeural', 'name': 'Jenny (US English)', 'gender': 'female', 'language': 'en-US'},
                {'id': 'en-US-GuyNeural', 'name': 'Guy (US English)', 'gender': 'male', 'language': 'en-US'},
                {'id': 'en-GB-SoniaNeural', 'name': 'Sonia (UK English)', 'gender': 'female', 'language': 'en-GB'},
                {'id': 'ja-JP-NanamiNeural', 'name': 'Nanami (Japanese)', 'gender': 'female', 'language': 'ja-JP'},
            ]
        except Exception as e:
            logger.error(f"Failed to get Azure voices: {e}")
            return []
    
    async def speak(self, text: str, interrupt: bool = True) -> bool:
        """Synthesize and play speech from text."""
        if not text.strip():
            return False
        
        try:
            # Stop current speech if interrupting
            if interrupt and self.is_speaking:
                await self.stop()
            
            # Emit speech start event
            if self.event_bus:
                await self.event_bus.emit("speech_started", text)
            
            self.is_speaking = True
            
            # Generate audio based on engine
            if self.config.tts_engine == "pyttsx3":
                success = await self._speak_pyttsx3(text)
            elif self.config.tts_engine == "azure":
                success = await self._speak_azure(text)
            elif self.config.tts_engine == "openai":
                success = await self._speak_openai(text)
            elif self.config.tts_engine == "gemini":
                success = await self._speak_gemini(text)
            else:
                success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
            self.is_speaking = False
            return False
    
    async def _speak_pyttsx3(self, text: str) -> bool:
        """Synthesize speech using pyttsx3."""
        try:
            def on_start(name):
                if self.on_speech_start:
                    self.on_speech_start(name)
            
            def on_end(name, completed):
                if self.on_speech_end:
                    self.on_speech_end(name, completed)
                self.is_speaking = False
            
            # Set callbacks
            self.pyttsx3_engine.connect('started-utterance', on_start)
            self.pyttsx3_engine.connect('finished-utterance', on_end)
            
            # Synthesize speech
            self.pyttsx3_engine.say(text)
            
            # Run in thread to avoid blocking
            def run_tts():
                try:
                    self.pyttsx3_engine.runAndWait()
                except Exception as e:
                    logger.error(f"pyttsx3 playback error: {e}")
                    self.is_speaking = False
            
            tts_thread = threading.Thread(target=run_tts, daemon=True)
            tts_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"pyttsx3 synthesis error: {e}")
            self.is_speaking = False
            return False
    
    async def _speak_azure(self, text: str) -> bool:
        """Synthesize speech using Azure."""
        if not self.azure_synthesizer:
            return False
        
        try:
            # Create SSML for better control
            ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                <voice name="{self.azure_synthesizer.speech_config.speech_synthesis_voice_name}">
                    <prosody rate="{self._get_rate_modifier()}" pitch="{self._get_pitch_modifier()}">
                        {text}
                    </prosody>
                </voice>
            </speak>
            """
            
            # Synthesize
            result = self.azure_synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.debug("Azure TTS synthesis completed")
                
                # Emit speech end event
                if self.event_bus:
                    await self.event_bus.emit("speech_ended")
                
                self.is_speaking = False
                return True
            else:
                logger.error(f"Azure TTS synthesis failed: {result.reason}")
                self.is_speaking = False
                return False
                
        except Exception as e:
            logger.error(f"Azure TTS error: {e}")
            self.is_speaking = False
            return False
    
    async def _speak_openai(self, text: str) -> bool:
        """Synthesize speech using OpenAI."""
        if not self.openai_client:
            return False
        
        try:
            # Generate speech
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice=self.current_voice_id,
                input=text,
                speed=self._get_openai_speed()
            )
            
            # Save to temporary file and play
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                response.stream_to_file(tmp_file.name)
                
                # Queue audio for playback
                self.audio_queue.put(tmp_file.name)
            
            return True
            
        except Exception as e:
            logger.error(f"OpenAI TTS error: {e}")
            self.is_speaking = False
            return False
    
    async def _speak_gemini(self, text: str) -> bool:
        """Synthesize speech using Google Cloud TTS."""
        if not self.gemini_tts:
            return False
        
        try:
            # Synthesize and play speech
            success = await self.gemini_tts.speak(text)
            
            if success:
                logger.debug("Google Cloud TTS synthesis completed")
                
                # Emit speech end event
                if self.event_bus:
                    await self.event_bus.emit("speech_ended")
                
                self.is_speaking = False
                return True
            else:
                logger.error("Google Cloud TTS synthesis failed")
                self.is_speaking = False
                return False
                
        except Exception as e:
            logger.error(f"Google Cloud TTS error: {e}")
            self.is_speaking = False
            return False
    
    def _get_rate_modifier(self) -> str:
        """Get rate modifier for SSML."""
        rate_pct = int((self.config.tts_rate / 200) * 100)  # Normalize to percentage
        return f"{rate_pct}%"
    
    def _get_pitch_modifier(self) -> str:
        """Get pitch modifier for SSML."""
        return "medium"  # Could be configured
    
    def _get_openai_speed(self) -> float:
        """Get speed for OpenAI TTS."""
        return max(0.25, min(4.0, self.config.tts_rate / 200))
    
    def _audio_playback_loop(self):
        """Audio playback thread loop."""
        try:
            import pygame
            pygame.mixer.init()
            
            while not self.should_stop:
                try:
                    # Get audio file from queue
                    audio_file = self.audio_queue.get(timeout=1.0)
                    
                    if audio_file and os.path.exists(audio_file):
                        # Play audio
                        pygame.mixer.music.load(audio_file)
                        pygame.mixer.music.play()
                        
                        # Wait for playback to complete
                        while pygame.mixer.music.get_busy() and not self.should_stop:
                            threading.Event().wait(0.1)
                        
                        # Cleanup
                        try:
                            os.unlink(audio_file)
                        except:
                            pass
                        
                        self.is_speaking = False
                        
                        # Emit speech end event
                        if self.event_bus:
                            asyncio.run_coroutine_threadsafe(
                                self.event_bus.emit("speech_ended"),
                                asyncio.get_event_loop()
                            )
                
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Audio playback error: {e}")
                    self.is_speaking = False
        
        except ImportError:
            logger.error("pygame not available for audio playback")
        except Exception as e:
            logger.error(f"Audio playback loop error: {e}")
    
    async def stop(self):
        """Stop current speech synthesis."""
        try:
            self.is_speaking = False
            
            if self.config.tts_engine == "pyttsx3" and self.pyttsx3_engine:
                self.pyttsx3_engine.stop()
            
            # Clear audio queue
            while not self.audio_queue.empty():
                try:
                    audio_file = self.audio_queue.get_nowait()
                    if os.path.exists(audio_file):
                        os.unlink(audio_file)
                except:
                    pass
            
            logger.debug("Speech synthesis stopped")
            
        except Exception as e:
            logger.error(f"Error stopping speech: {e}")
    
    async def set_voice(self, voice_id: str):
        """Change the current voice."""
        try:
            if self.config.tts_engine == "pyttsx3" and self.pyttsx3_engine:
                voices = self.pyttsx3_engine.getProperty('voices')
                for voice in voices:
                    if voice.id == voice_id or voice_id.lower() in voice.name.lower():
                        self.pyttsx3_engine.setProperty('voice', voice.id)
                        self.current_voice_id = voice.id
                        break
            
            elif self.config.tts_engine == "azure" and self.azure_synthesizer:
                self.azure_synthesizer.speech_config.speech_synthesis_voice_name = voice_id
                self.current_voice_id = voice_id
            
            elif self.config.tts_engine == "openai":
                if voice_id in [v['id'] for v in self.available_voices]:
                    self.current_voice_id = voice_id
            
            elif self.config.tts_engine == "gemini" and self.gemini_tts:
                self.gemini_tts.update_config(voice_name=voice_id)
                self.current_voice_id = voice_id
            
            logger.info(f"Voice changed to: {voice_id}")
            
        except Exception as e:
            logger.error(f"Failed to change voice: {e}")
    
    async def set_rate(self, rate: int):
        """Set speech rate."""
        self.config.tts_rate = max(50, min(400, rate))
        
        if self.config.tts_engine == "pyttsx3" and self.pyttsx3_engine:
            self.pyttsx3_engine.setProperty('rate', self.config.tts_rate)
    
    async def set_volume(self, volume: float):
        """Set speech volume."""
        self.config.tts_volume = max(0.0, min(1.0, volume))
        
        if self.config.tts_engine == "pyttsx3" and self.pyttsx3_engine:
            self.pyttsx3_engine.setProperty('volume', self.config.tts_volume)
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices."""
        return self.available_voices.copy()
    
    def get_current_voice(self) -> Optional[str]:
        """Get current voice ID."""
        return self.current_voice_id
    
    def is_currently_speaking(self) -> bool:
        """Check if currently speaking."""
        return self.is_speaking
    
    async def test_speech(self, test_text: str = "Hello, this is a test of the voice synthesis system."):
        """Test the speech synthesis with sample text."""
        logger.info("Testing speech synthesis...")
        success = await self.speak(test_text)
        return success
    
    async def shutdown(self):
        """Shutdown the voice synthesis system."""
        try:
            await self.stop()
            
            # Stop playback thread
            self.should_stop = True
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(timeout=2.0)
            
            # Cleanup engines
            if self.pyttsx3_engine:
                try:
                    self.pyttsx3_engine.stop()
                except:
                    pass
            
            logger.info("Voice synthesis shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during voice synthesis shutdown: {e}")
