"""
Google Gemini AI client for VRM AI Chatbot.
Provides integration with Google's Gemini AI models for conversation.
"""

import os
import asyncio
from typing import Optional, Dict, Any, List
import logging
from dataclasses import dataclass

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class GeminiConfig:
    """Configuration for Gemini AI client."""
    api_key: str
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 40
    safety_settings: Optional[Dict] = None

class GeminiClient:
    """Google Gemini AI client for conversation generation."""
    
    def __init__(self, config: GeminiConfig):
        """Initialize Gemini client."""
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed")
        
        self.config = config
        self.model = None
        self.chat_session = None
        
        # Configure Gemini
        genai.configure(api_key=config.api_key)
        
        # Set up safety settings
        self.safety_settings = config.safety_settings or {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"Gemini client initialized with model: {config.model_name}")
    
    def _initialize_model(self):
        """Initialize the Gemini model."""
        try:
            generation_config = {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "max_output_tokens": self.config.max_tokens,
            }
            
            self.model = genai.GenerativeModel(
                model_name=self.config.model_name,
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )
            
            logger.info("Gemini model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
    
    def start_chat(self, system_prompt: Optional[str] = None, history: Optional[List[Dict]] = None):
        """Start a new chat session."""
        try:
            # Convert history to Gemini format if provided
            gemini_history = []
            if history:
                for msg in history:
                    role = "user" if msg.get("role") == "user" else "model"
                    gemini_history.append({
                        "role": role,
                        "parts": [msg.get("content", "")]
                    })
            
            # Start chat session
            self.chat_session = self.model.start_chat(history=gemini_history)
            
            # Send system prompt if provided
            if system_prompt:
                self.chat_session.send_message(f"System: {system_prompt}")
            
            logger.info("Gemini chat session started")
            
        except Exception as e:
            logger.error(f"Failed to start Gemini chat: {e}")
            raise
    
    async def generate_response(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a response using Gemini."""
        try:
            if not self.chat_session:
                self.start_chat()
            
            # Add context to message if provided
            if context:
                character_info = context.get("character", {})
                if character_info:
                    name = character_info.get("name", "Assistant")
                    personality = character_info.get("personality_traits", [])
                    context_msg = f"[Character: {name}, Traits: {', '.join(personality)}] {message}"
                else:
                    context_msg = message
            else:
                context_msg = message
            
            # Generate response
            response = await asyncio.to_thread(
                self.chat_session.send_message, 
                context_msg
            )
            
            response_text = response.text.strip()
            logger.info(f"Generated Gemini response: {len(response_text)} characters")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Failed to generate Gemini response: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        """Get list of available Gemini models."""
        try:
            models = []
            for model in genai.list_models():
                if 'generateContent' in model.supported_generation_methods:
                    models.append(model.name)
            return models
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Reinitialize model with new config
        self._initialize_model()
        logger.info("Gemini configuration updated")

def create_gemini_client(api_key: str, model_name: str = "gemini-1.5-flash", **kwargs) -> Optional[GeminiClient]:
    """Create a Gemini client with the given configuration."""
    if not GEMINI_AVAILABLE:
        logger.warning("Gemini not available - google-generativeai package not installed")
        return None
    
    if not api_key:
        logger.warning("Gemini API key not provided")
        return None
    
    try:
        config = GeminiConfig(
            api_key=api_key,
            model_name=model_name,
            **kwargs
        )
        return GeminiClient(config)
    except Exception as e:
        logger.error(f"Failed to create Gemini client: {e}")
        return None