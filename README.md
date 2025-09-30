# 🤖 VRM AI Chatbot - Your Personal 3D AI Companion

A sophisticated desktop AI companion featuring 3D VRM avatars, voice interaction, and personality-driven conversations. Built with Python 3.11+ and modern technologies.

## ✨ Features

### 🎨 **Visual & Interactive**

- **3D VRM Avatar Rendering** - Full VRM 1.0 support with MToon shaders
- **Transparent Desktop Overlay** - Borderless window that sits on your desktop
- **Real-time Facial Animation** - Lip sync, eye tracking, and emotional expressions
- **Windows Integration** - Click-through, always-on-top, and system tray support

### 🧠 **AI & Conversation**

- **Hybrid AI Backend** - Local models (Ollama) + Cloud APIs (OpenAI, Claude, Gemini)
- **Personality System** - Customizable traits, background stories, and speaking styles
- **Context Awareness** - Maintains conversation history and emotional state
- **Multi-language Support** - Configurable language and regional settings

### 🎤 **Voice & Audio**

- **Advanced TTS** - pyttsx3, Azure Speech, OpenAI voices, Google Cloud TTS
- **Speech Recognition** - Whisper, Azure STT, Google Cloud STT, real-time processing
- **Voice Customization** - Pitch, speed, volume, accent control
- **Lip Sync Animation** - Phoneme-based mouth movement

### 📞 **Communication**

- **LiveKit Integration** - Professional video calling capabilities
- **System Integration** - File operations, automation, hotkeys
- **Multi-platform** - Windows (full features), macOS/Linux (basic)

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (3.11.13 recommended)
- **uv** package manager (recommended) or pip
- **4GB+ RAM** (8GB+ recommended)
- **Graphics card** with OpenGL 3.3+ support

### Installation

#### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/coff33ninja/vrm-ai-chatbot.git
cd vrm-ai-chatbot

# Create virtual environment with uv
uv venv --python 3.11
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
uv pip install -r requirements.txt
# OR run the installer
python install.py

# Configure environment
# Edit .env with your API keys (optional for local AI)

# Run the application
python main.py
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/coff33ninja/vrm-ai-chatbot.git
cd vrm-ai-chatbot

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env with your API keys

# Run the application
python main.py
```

### 2. Configuration

The `.env` file is created automatically from `.env.template`. Add your API keys:

```env
# AI Services (choose one or more)
OPENAI_API_KEY=sk-your_key_here
ANTHROPIC_API_KEY=sk-ant-your_key_here
GEMINI_API_KEY=your_gemini_key_here

# Voice Services (optional)
AZURE_SPEECH_KEY=your_azure_key
AZURE_SPEECH_REGION=eastus

# Google Cloud Services (optional)
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# LiveKit (optional)
LIVEKIT_API_KEY=your_livekit_key
LIVEKIT_API_SECRET=your_livekit_secret
```

### 3. Add VRM Model

1. Download a VRM model:

   - [VRoid Hub](https://hub.vroid.com/) (Free)
   - [Booth.pm](https://booth.pm/) (Commercial)
   - Create your own with [VRoid Studio](https://vroid.com/studio)

2. Place the `.vrm` file in `assets/models/`

3. Update `configs/config.yaml`:
   ```yaml
   graphics:
     default_model: "assets/models/your_character.vrm"
   ```

### 4. Run

```bash
python main.py
```

## 🎯 Usage Guide

### Basic Interaction

- **Start Conversation**: Simply speak or type
- **Move Character**: Click and drag anywhere on the character
- **Show/Hide**: Double-click or press `Esc`
- **Settings**: Right-click for context menu

### Voice Commands

```
"Hello" - Start conversation
"How are you?" - Chat naturally
"Change your voice" - Switch TTS voice
"Become more cheerful" - Adjust personality
"Goodbye" - End conversation
```

### Hotkeys

- `Ctrl+Shift+A` - Toggle visibility
- `Ctrl+Shift+V` - Start video call
- `Ctrl+Shift+S` - Open settings
- `F11` - Toggle fullscreen
- `T` - Cycle transparency levels

## 🔧 Configuration

### AI Backend Options

#### Local AI (Recommended for Privacy)

```bash
# Install Ollama
# Download from https://ollama.ai/

# Pull models
ollama pull llama3.1:8b     # Balanced quality/speed
ollama pull mistral:7b      # Alternative option
ollama pull codellama:7b    # For coding help
```

Configure in `configs/config.yaml`:

```yaml
ai:
  use_local_ai: true
  local_model: "llama3.1:8b"
```

#### Cloud AI (Best Quality)

```yaml
ai:
  use_local_ai: false
  # Set API keys in .env file
```

#### Google Gemini (Fast & Cost-Effective)

```yaml
ai:
  use_local_ai: false
  # Gemini will be tried first, then OpenAI, then Claude
  # Set GEMINI_API_KEY in .env file
```

### Voice Configuration

#### Built-in TTS (Default)

```yaml
voice:
  tts_engine: "pyttsx3"
  tts_voice: "default" # or specific system voice
  tts_rate: 200
  tts_volume: 0.9
```

#### Premium Voice Services

```yaml
voice:
  # Azure (high quality, many languages)
  tts_engine: "azure"

  # OpenAI (very natural, English focus)
  tts_engine: "openai"
  tts_voice: "nova"  # alloy, echo, fable, onyx, nova, shimmer

  # Google Cloud (high quality, many languages, cost-effective)
  tts_engine: "gemini"
  tts_voice: "en-US-Neural2-F"  # or other Google voices
```

### Character Personality

Create custom personalities in `data/characters/`:

```yaml
# my_character.yaml
name: "Luna"
personality_traits:
  - name: "friendly"
    intensity: 0.8
  - name: "curious"
    intensity: 0.9
  - name: "helpful"
    intensity: 0.7

background_story: "I'm Luna, your AI research assistant. I love learning about new topics and helping with creative projects!"

speaking_style: "enthusiastic and supportive"

interests:
  - "science and technology"
  - "creative writing"
  - "problem solving"
  - "learning new things"

voice_settings:
  pitch: 0.1 # Slightly higher pitch
  speed: 1.1 # Slightly faster speech
  volume: 0.9
```

## 📁 Project Structure

```
vrm-ai-chatbot/
├── .venv/                 # Virtual environment
├── src/                   # Source code
│   ├── core/             # Core application
│   ├── ai/               # AI conversation
│   ├── voice/            # Voice synthesis
│   ├── graphics/         # 3D rendering
│   ├── models/           # Data models
│   ├── gui/              # GUI components
│   ├── integrations/     # System integrations
│   └── utils/            # Utilities
├── assets/               # Asset files
│   ├── models/           # VRM models
│   ├── shaders/          # OpenGL shaders
│   ├── textures/         # Textures
│   └── voices/           # Voice files
├── configs/              # Configuration
├── data/                 # Data files
│   ├── characters/       # Character profiles
│   └── conversations/    # Chat history
├── logs/                 # Application logs
├── tests/                # Test files
├── dev-tools/            # Development scripts
│   ├── project_analysis.py
│   ├── project_setup.py
│   ├── renderer_setup.py
│   ├── voice_ai_setup.py
│   └── final_setup.py
├── main.py               # Application entry point
├── install.py            # Installation script
├── launch.bat            # Windows launcher
├── tests/                # Test files
│   ├── test_basic.py     # Basic environment test
│   ├── test_components.py # Component verification test
│   ├── test_gemini.py    # Gemini integration test
│   └── run_tests.py      # Test runner
├── requirements.txt      # Dependencies
└── requirements-core.txt # Core dependencies only
```

## 🏗️ Architecture

### Core Components

```
src/
├── core/              # Application framework
│   ├── application.py # Main app orchestration
│   ├── config.py      # Configuration management
│   └── event_bus.py   # Event system
├── ai/                # Conversation management
│   └── conversation.py # AI response generation
├── voice/             # Audio processing
│   └── synthesis.py   # Text-to-speech
├── graphics/          # 3D rendering
│   ├── renderer.py    # VRM/OpenGL rendering
│   ├── window.py      # Window management
│   └── shader_loader.py # Shader loading
├── models/            # Data structures
│   └── character.py   # Character with personality
├── gui/               # GUI components
│   └── splash.py      # Splash screen
├── integrations/      # System integrations
│   └── __init__.py
└── utils/             # Shared utilities
    ├── event_bus.py   # Component communication
    ├── logger.py      # Logging system
    ├── config.py      # Config utilities
    ├── shader_loader.py # Shader utilities
    └── splash.py      # Splash utilities
```

### Event-Driven Design

The application uses an async event bus for communication:

- `speech_recognized` → AI processes input
- `ai_response` → Voice synthesis speaks
- `speech_started` → Character lip sync begins
- `window_closed` → Graceful shutdown

### Rendering Pipeline

1. **VRM Loading** - Parse .vrm files with glTF structure
2. **Mesh Processing** - Extract geometry, materials, bones
3. **Shader Compilation** - MToon and standard PBR shaders
4. **Animation Update** - Blend shapes, bone transforms
5. **Frame Render** - OpenGL drawing with transparency

## 🎨 Customization

### Creating Custom Characters

1. **Design in VRoid Studio** (free)
2. **Export as VRM** with proper settings
3. **Configure personality** in YAML file
4. **Test and iterate** voice and behavior

### Advanced Scripting

The system supports Python plugins:

```python
# plugins/my_plugin.py
async def on_conversation_start(character, user_input):
    """Custom behavior when conversation starts"""
    if "weather" in user_input.lower():
        # Add weather data to context
        weather = await get_weather()
        character.context["weather"] = weather

# Register plugin
app.register_plugin("conversation_start", on_conversation_start)
```

### Custom Shaders

Add custom shaders to `assets/shaders/`:

```glsl
// custom.vert - Vertex shader
// custom.frag - Fragment shader
```

## 🔒 Privacy & Security

### Data Protection

- **Local AI**: All conversations stay on your device
- **No Telemetry**: No usage data is collected
- **API Keys**: Stored locally in `.env` file
- **VRM Models**: Processed entirely locally

### Network Usage

- **Local Mode**: No internet required after setup
- **Cloud Mode**: Only API calls to configured services
- **Optional Features**: LiveKit, cloud TTS clearly marked

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
uv pip install -r requirements.txt  # Includes dev dependencies

# Run all tests
python tests/run_tests.py

# Run individual tests
python tests/test_components.py
python tests/test_basic.py
python tests/test_gemini.py

# Code formatting (included in requirements)
black src/
flake8 src/
mypy src/
```

### Development Scripts

The `dev-tools/` folder contains the setup scripts used to generate this project:

- `project_analysis.py` - Technical analysis and feasibility study
- `project_setup.py` - Core application framework generator
- `renderer_setup.py` - 3D graphics and VRM rendering system
- `voice_ai_setup.py` - Voice synthesis and AI components
- `final_setup.py` - Final components and assets

These scripts have already been run and are kept for reference and regeneration if needed.

## 🛠️ Troubleshooting

### Common Issues

#### Installation Problems

```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Using pip (fallback)
pip install -r requirements.txt

# If specific packages fail
pip install PyOpenGL PyOpenGL_accelerate
pip install pygame sounddevice
pip install pywin32  # Windows only
```

#### Performance Issues

1. **Lower graphics settings**:

   ```yaml
   graphics:
     target_fps: 30
     antialiasing: false
     window_width: 640
     window_height: 480
   ```

2. **Use local AI**: Faster than cloud APIs
3. **Close other applications**: Free up system resources
4. **Update graphics drivers**: Ensure OpenGL support

#### VRM Loading Errors

- Verify `.vrm` file is not corrupted
- Check file permissions
- Try with included default model first
- Ensure model follows VRM 1.0 standard

#### Voice Issues

- **No audio**: Check system audio settings
- **Wrong voice**: Configure `tts_voice` in settings
- **API errors**: Verify keys in `.env` file
- **Microphone**: Check permissions and device selection

### Debug Mode

Enable detailed logging:

```yaml
debug: true
log_level: "DEBUG"
```

Check logs in `logs/` directory for detailed error information.

## 🤝 Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/
```

### Areas for Contribution

- Additional VRM features (VRM 2.0, advanced materials)
- More AI backends (local transformers, specialized models)
- Enhanced animations (gesture recognition, emotion detection)
- Platform support (macOS, Linux improvements)
- Performance optimizations
- Documentation and tutorials

## 📄 License & Credits

### Project License

This project is released under the MIT License. See LICENSE file for details.

### VRM Models

- Check individual model licenses before use
- Many VRoid Hub models are CC0 (public domain)
- Commercial models may have usage restrictions

### Dependencies

- **PyOpenGL**: 3D graphics rendering
- **ModernGL**: Modern OpenGL wrapper
- **pygltflib**: VRM/glTF file parsing
- **pyttsx3**: Text-to-speech synthesis
- **OpenAI/Anthropic**: AI conversation APIs
- **LiveKit**: Real-time communication
- **Many others**: See `requirements.txt`

## 🌟 Showcase

Share your creations:

- Tag `#VRMAIChatbot` on social media
- Submit screenshots to the gallery
- Share custom characters and personalities
- Contribute improvements and features

## 📞 Support

- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Documentation**: Check SETUP.md for detailed guides
- **Community**: Join our Discord server

---

**Made with ❤️ by the VRM AI Chatbot community**

_Bringing AI companions to life with personality, voice, and 3D avatars_
