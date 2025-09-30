"""
AI Conversation Manager - Handles conversations with local and cloud AI models.
Supports personality-driven responses and context management.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List, Callable
import json
import time
from datetime import datetime
import re

# AI client imports
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import requests  # For Ollama local AI
except ImportError:
    requests = None

try:
    from .gemini_client import create_gemini_client, GeminiClient
except ImportError:
    create_gemini_client = None
    GeminiClient = None

from ..core.config import AIConfig
from ..models.character import Character
from ..utils.event_bus import EventBus

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages AI conversations with personality and context awareness."""
    
    def __init__(self, config: AIConfig, character: Character, event_bus: EventBus):
        self.config = config
        self.character = character
        self.event_bus = event_bus
        
        # AI clients
        self.openai_client: Optional[Any] = None
        self.anthropic_client: Optional[Any] = None
        self.gemini_client: Optional[GeminiClient] = None
        self.ollama_base_url = config.local_api_url
        
        # Conversation state
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt = ""
        self.current_context = ""
        
        # Response processing
        self.response_filters: List[Callable[[str], str]] = []
        self.emotion_keywords = {
            'happy': ['happy', 'joy', 'excited', 'cheerful', 'delighted'],
            'sad': ['sad', 'depressed', 'melancholy', 'down', 'upset'],
            'angry': ['angry', 'furious', 'irritated', 'annoyed', 'mad'],
            'surprised': ['surprised', 'shocked', 'amazed', 'astonished'],
            'excited': ['excited', 'enthusiastic', 'thrilled', 'energetic'],
            'calm': ['calm', 'peaceful', 'relaxed', 'serene', 'tranquil']
        }
        
        # Performance tracking
        self.response_times: List[float] = []
        self.error_count = 0
        
        logger.info(f"Conversation manager initialized for character: {character.name}")
    
    async def initialize(self):
        """Initialize AI clients and system prompt."""
        try:
            # Initialize cloud AI clients
            if self.config.openai_api_key and openai:
                self.openai_client = openai.OpenAI(api_key=self.config.openai_api_key)
                logger.info("OpenAI client initialized")
            
            if self.config.anthropic_api_key and anthropic:
                self.anthropic_client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
                logger.info("Anthropic client initialized")
            
            if hasattr(self.config, 'gemini_api_key') and self.config.gemini_api_key and create_gemini_client:
                self.gemini_client = create_gemini_client(
                    api_key=self.config.gemini_api_key,
                    model_name=getattr(self.config, 'gemini_model', 'gemini-1.5-flash'),
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                if self.gemini_client:
                    logger.info("Gemini client initialized")
            
            # Test local AI connection
            if self.config.use_local_ai:
                await self._test_local_ai_connection()
            
            # Setup system prompt
            await self._setup_system_prompt()
            
            # Add response filters
            self._setup_response_filters()
            
            logger.info("Conversation manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize conversation manager: {e}")
            raise
    
    async def _test_local_ai_connection(self):
        """Test connection to local AI (Ollama)."""
        if not requests:
            logger.warning("requests library not available for local AI")
            return
        
        try:
            response = requests.get(f"{self.ollama_base_url}/api/version", timeout=5)
            if response.status_code == 200:
                logger.info("Local AI (Ollama) connection successful")
            else:
                logger.warning(f"Local AI connection failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to local AI: {e}")
    
    async def _setup_system_prompt(self):
        """Setup the system prompt based on character personality."""
        personality_prompt = self.character.get_personality_prompt()
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        self.system_prompt = f"""
        {personality_prompt}
        
        Additional Instructions:
        - Current date and time: {current_time}
        - You are an interactive AI character appearing as a 3D avatar on the user's screen
        - Respond naturally and maintain your character's personality
        - Keep responses conversational and engaging
        - Use emotion and personality in your responses that matches your character
        - If asked about your nature, explain that you're an AI companion designed to be helpful and engaging
        - You can see and interact with the user through voice and text
        - Express emotions that will be reflected in your avatar's facial expressions
        - Be helpful with tasks but stay in character
        
        Remember: You are {self.character.name}, and you should always respond as this character would.
        """.strip()
        
        logger.debug("System prompt configured")
    
    def _setup_response_filters(self):
        """Setup response processing filters."""
        
        def remove_asterisk_actions(text: str) -> str:
            """Remove asterisk-enclosed actions like *smiles* or *nods*."""
            return re.sub(r'\*[^*]+\*', '', text).strip()
        
        def clean_formatting(text: str) -> str:
            """Clean up formatting artifacts."""
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            # Remove leading/trailing whitespace
            text = text.strip()
            return text
        
        def limit_length(text: str) -> str:
            """Limit response length for TTS."""
            if len(text) > 500:  # Configurable limit
                # Find a good break point
                sentences = text.split('. ')
                result = ""
                for sentence in sentences:
                    if len(result + sentence) < 450:
                        result += sentence + ". "
                    else:
                        break
                return result.strip()
            return text
        
        self.response_filters = [
            remove_asterisk_actions,
            clean_formatting,
            limit_length
        ]
    
    async def process_input(self, user_input: str) -> Optional[str]:
        """Process user input and generate AI response."""
        if not user_input.strip():
            return None
        
        try:
            start_time = time.time()
            
            # Add user input to conversation history
            self.character.add_to_conversation_history("User", user_input)
            
            # Update character mood based on input sentiment
            await self._analyze_input_sentiment(user_input)
            
            # Generate response
            response = await self._generate_response(user_input)
            
            if response:
                # Process response through filters
                processed_response = self._apply_response_filters(response)
                
                # Add to conversation history
                self.character.add_to_conversation_history(self.character.name, processed_response)
                
                # Analyze response for emotions
                await self._analyze_response_emotion(processed_response)
                
                # Emit AI response event
                if self.event_bus:
                    await self.event_bus.emit("ai_response", processed_response)
                
                # Track performance
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                if len(self.response_times) > 100:
                    self.response_times = self.response_times[-50:]
                
                logger.info(f"AI response generated in {response_time:.2f}s")
                return processed_response
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            self.error_count += 1
            return await self._get_fallback_response()
    
    async def _generate_response(self, user_input: str) -> Optional[str]:
        """Generate AI response using the configured backend."""
        
        # Prepare conversation context
        messages = self._prepare_messages(user_input)
        
        try:
            if self.config.use_local_ai:
                return await self._generate_local_response(messages)
            else:
                return await self._generate_cloud_response(messages)
        
        except Exception as e:
            logger.error(f"AI generation error: {e}")
            # Try fallback method
            if self.config.use_local_ai:
                logger.info("Trying cloud AI as fallback")
                return await self._generate_cloud_response(messages)
            else:
                logger.info("Trying local AI as fallback") 
                return await self._generate_local_response(messages)
    
    def _prepare_messages(self, user_input: str) -> List[Dict[str, str]]:
        """Prepare message history for AI generation."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add recent conversation context
        context = self.character.get_conversation_context(max_messages=10)
        if context:
            messages.append({"role": "assistant", "content": f"Previous conversation:\n{context}"})
        
        # Add current user message
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    async def _generate_local_response(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Generate response using local AI (Ollama)."""
        if not requests:
            raise ImportError("requests library required for local AI")
        
        try:
            # Convert messages to prompt format for Ollama
            prompt_parts = []
            for msg in messages:
                if msg["role"] == "system":
                    prompt_parts.append(f"System: {msg['content']}")
                elif msg["role"] == "user":
                    prompt_parts.append(f"Human: {msg['content']}")
                elif msg["role"] == "assistant":
                    prompt_parts.append(f"Assistant: {msg['content']}")
            
            prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
            
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.config.local_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"Local AI request failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Local AI generation error: {e}")
            return None
    
    async def _generate_cloud_response(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Generate response using cloud AI."""
        
        # Try Gemini first (cost-effective and fast)
        if self.gemini_client:
            try:
                # Extract user message for Gemini
                user_message = ""
                character_context = {}
                
                for msg in messages:
                    if msg["role"] == "user":
                        user_message = msg["content"]
                    elif msg["role"] == "system":
                        # Add character context
                        character_context = {
                            "character": {
                                "name": self.character.name,
                                "personality_traits": [trait.name for trait in self.character.personality_traits]
                            }
                        }
                
                if user_message:
                    # Start chat with system prompt if not already started
                    if not hasattr(self.gemini_client, 'chat_session') or not self.gemini_client.chat_session:
                        system_prompt = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
                        self.gemini_client.start_chat(system_prompt=system_prompt)
                    
                    response = await self.gemini_client.generate_response(user_message, context=character_context)
                    if response:
                        return response
                        
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
        
        # Try OpenAI as fallback
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",  # Cost-effective model
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                if response.choices:
                    return response.choices[0].message.content.strip()
                    
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
        
        # Try Anthropic as final fallback
        if self.anthropic_client:
            try:
                # Convert messages for Anthropic format
                system_content = ""
                anthropic_messages = []
                
                for msg in messages:
                    if msg["role"] == "system":
                        system_content = msg["content"]
                    else:
                        anthropic_messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                
                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",  # Cost-effective model
                    system=system_content,
                    messages=anthropic_messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                if response.content:
                    return response.content[0].text.strip()
                    
            except Exception as e:
                logger.error(f"Anthropic API error: {e}")
        
        return None
    
    def _apply_response_filters(self, response: str) -> str:
        """Apply response processing filters."""
        for filter_func in self.response_filters:
            try:
                response = filter_func(response)
            except Exception as e:
                logger.error(f"Response filter error: {e}")
        
        return response
    
    async def _analyze_input_sentiment(self, user_input: str):
        """Analyze user input sentiment and update character mood."""
        try:
            # Simple keyword-based sentiment analysis
            user_input_lower = user_input.lower()
            
            # Check for emotional keywords
            detected_emotions = []
            for emotion, keywords in self.emotion_keywords.items():
                for keyword in keywords:
                    if keyword in user_input_lower:
                        detected_emotions.append(emotion)
                        break
            
            # Update character mood if emotions detected
            if detected_emotions:
                primary_emotion = detected_emotions[0]  # Use first detected
                await self.character.update_mood(primary_emotion)
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
    
    async def _analyze_response_emotion(self, response: str):
        """Analyze AI response for emotional content."""
        try:
            response_lower = response.lower()
            
            # Detect emotions in response
            for emotion, keywords in self.emotion_keywords.items():
                for keyword in keywords:
                    if keyword in response_lower:
                        # Set character emotion
                        await self.character.set_emotion(emotion, 0.6)
                        return
            
            # Default to current mood
            await self.character.set_emotion(self.character.current_mood, 0.4)
            
        except Exception as e:
            logger.error(f"Response emotion analysis error: {e}")
    
    async def _get_fallback_response(self) -> str:
        """Get a fallback response when AI generation fails."""
        fallback_responses = [
            "I'm having a bit of trouble thinking right now. Could you repeat that?",
            "My thoughts seem a bit scattered at the moment. What did you say?",
            "I'm experiencing some technical difficulties. Can you try again?",
            "Sorry, I didn't quite catch that. Could you rephrase?",
            "I need a moment to process. What were you saying?"
        ]
        
        import random
        return random.choice(fallback_responses)
    
    async def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        self.character.conversation_history = []
        logger.info("Conversation history cleared")
    
    async def set_personality_trait(self, trait: str, intensity: float):
        """Update a personality trait and refresh system prompt."""
        self.character.add_personality_trait(trait, intensity)
        await self._setup_system_prompt()
        logger.info(f"Updated personality trait: {trait} = {intensity}")
    
    async def get_conversation_summary(self) -> str:
        """Generate a summary of the current conversation."""
        if not self.character.conversation_history:
            return "No conversation yet."
        
        # Simple summary of recent messages
        recent_messages = self.character.conversation_history[-10:]
        summary_lines = []
        
        for msg in recent_messages:
            speaker = msg['speaker']
            message = msg['message'][:100] + "..." if len(msg['message']) > 100 else msg['message']
            summary_lines.append(f"{speaker}: {message}")
        
        return "\n".join(summary_lines)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get conversation performance statistics."""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            'total_messages': len(self.character.conversation_history),
            'average_response_time': avg_response_time,
            'error_count': self.error_count,
            'current_mood': self.character.current_mood,
            'active_ai_backend': 'local' if self.config.use_local_ai else 'cloud'
        }
    
    async def regenerate_response(self) -> Optional[str]:
        """Regenerate the last AI response."""
        if len(self.character.conversation_history) < 2:
            return None
        
        # Get the last user message
        last_user_message = None
        for msg in reversed(self.character.conversation_history):
            if msg['speaker'] == 'User':
                last_user_message = msg['message']
                break
        
        if last_user_message:
            # Remove last AI response from history
            if self.character.conversation_history and self.character.conversation_history[-1]['speaker'] == self.character.name:
                self.character.conversation_history.pop()
            
            # Generate new response
            return await self.process_input(last_user_message)
        
        return None
    
    async def shutdown(self):
        """Shutdown the conversation manager."""
        try:
            # Save conversation history if needed
            # Could implement conversation persistence here
            
            logger.info("Conversation manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during conversation manager shutdown: {e}")
