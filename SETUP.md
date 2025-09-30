# VRM AI Chatbot - Setup Guide

Welcome to your personal VRM AI Chatbot! This guide will help you set up and configure your 3D AI companion.

## Installation

### 1. Python Environment

#### Option A: Using uv (Recommended)
```bash
# Install uv if not already installed
# Visit: https://docs.astral.sh/uv/getting-started/installation/

# Ensure Python 3.11+ is available
python --version

# Create virtual environment with uv
uv venv --python 3.11

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies with uv
uv pip install -r requirements.txt
```

#### Option B: Using pip
```bash
# Ensure Python 3.11+ is installed
python --version

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Quick Installation Script

```bash
# Run the automated installer
python install.py
```

This script will:
- Check Python version compatibility
- Install all dependencies
- Create necessary directories
- Set up configuration files
- Provide setup guidance

## Testing Your Setup

After installation, test your setup:

```bash
# Test basic environment
python test_basic.py

# Test all components
python test_components.py

# Test specific imports
python -c "import src.core.application; print('✅ Core modules loaded')"

# Test dependencies
python -c "import numpy, openai, moderngl; print('✅ Dependencies working')"

# Run the application (should show splash screen)
python main.py
```

### Expected Test Results

Both test scripts should show:
- ✅ Python version: 3.11.x
- ✅ All dependencies installed
- ✅ Project structure complete
- ✅ All components ready
- ✅ Configuration files exist

## Configuration

### Basic Configuration

The `.env` file is created automatically from `.env.template`. Edit it to add your API keys:

```env
# AI Configuration
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Voice Services (optional)
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=your_region

# Google Cloud Services (optional)
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# LiveKit (optional)
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_secret
LIVEKIT_SERVER_URL=wss://your-livekit-server.com
```

### Add VRM Models

1. Download VRM models from:
   - [VRoid Hub](https://hub.vroid.com/)
   - [Booth.pm](https://booth.pm/en/browse/3D%20Models)
   - [DOVA-SYNDROME](https://dova-s.jp/_contents/license/)

2. Place VRM files in: `assets/models/`

3. Update config to point to your model:
   ```yaml
   graphics:
     default_model: "assets/models/your_character.vrm"
   ```

### Run the Application

```bash
python main.py
```

## 📋 Detailed Setup

### Local AI Setup (Recommended)

For privacy and cost savings, set up local AI:

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai/)

2. **Download Models**:
   ```bash
   ollama pull llama3.1:8b        # Good balance of quality/speed
   ollama pull codellama:7b       # For coding assistance
   ollama pull mistral:7b         # Alternative model
   ```

3. **Configure**: Set `use_local_ai: true` in your config

### Cloud AI Setup

For maximum quality, configure cloud AI services:

#### OpenAI (Recommended)
1. Get API key from [OpenAI Platform](https://platform.openai.com/)
2. Add to `.env`: `OPENAI_API_KEY=sk-...`
3. Set `use_local_ai: false` in config

#### Anthropic Claude
1. Get API key from [Anthropic Console](https://console.anthropic.com/)
2. Add to `.env`: `ANTHROPIC_API_KEY=sk-ant-...`

#### Google Gemini (Recommended)
1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add to `.env`: `GEMINI_API_KEY=your_key_here`
3. Fast, cost-effective, and high-quality responses

### Voice Configuration

#### Local Voice (pyttsx3) - Default
- Works out of the box
- Uses system voices
- Configure in `configs/config.yaml`:
  ```yaml
  voice:
    tts_engine: "pyttsx3"
    tts_voice: "default"  # or specific voice name
    tts_rate: 200
    tts_volume: 0.9
  ```

#### Azure Speech Services (Premium)
1. Create Azure Speech resource
2. Get key and region from Azure portal
3. Configure:
   ```yaml
   voice:
     tts_engine: "azure"
     azure_speech_key: "your_key"
     azure_speech_region: "your_region"
   ```

#### OpenAI TTS (High Quality)
1. Use your OpenAI API key
2. Configure:
   ```yaml
   voice:
     tts_engine: "openai"
     tts_voice: "nova"  # or alloy, echo, fable, onyx, shimmer
   ```

#### Google Cloud TTS (Cost-Effective)
1. Create a Google Cloud project and enable Text-to-Speech API
2. Create a service account and download the JSON key file
3. Set `GOOGLE_APPLICATION_CREDENTIALS` in `.env` to the path of your JSON key
4. Configure:
   ```yaml
   voice:
     tts_engine: "gemini"
     tts_voice: "en-US-Neural2-F"  # or other Google voices
     language_code: "en-US"
   ```

### Character Customization

Create custom character profiles in `data/characters/`:

```yaml
# data/characters/my_character.yaml
name: "Aria"
personality_traits:
  - "friendly"
  - "helpful" 
  - "curious"
  - "empathetic"
background_story: "I'm your helpful AI companion, always here to chat and assist!"
speaking_style: "warm and conversational"
interests:
  - "technology"
  - "art"
  - "music"
  - "science"

voice_settings:
  pitch: 0.0
  speed: 1.0
  volume: 1.0
  accent: "neutral"
```

### Advanced Features

#### LiveKit Video Calling
1. Sign up at [LiveKit Cloud](https://livekit.io/)
2. Get API credentials
3. Configure in `.env` and enable in config
4. Use hotkey `Ctrl+Shift+V` to start video call

#### System Integration
Enable system integration for:
- Window management
- File operations
- System monitoring
- Hotkey support

```yaml
system:
  enable_system_integration: true
  hotkey_toggle: "ctrl+shift+a"
  always_on_top: true
  click_through: false
```

## 🎮 Usage

### Basic Controls
- **Double-click**: Show/hide character
- **Right-click**: Context menu
- **Drag**: Move character around screen
- **Esc**: Hide character
- **F11**: Toggle fullscreen
- **T**: Toggle transparency

### Voice Commands
- Say "Hello" to start conversation
- The character listens when you speak
- Supports natural conversation

### Hotkeys
- `Ctrl+Shift+A`: Toggle character visibility
- `Ctrl+Shift+V`: Start video call (if enabled)
- `Ctrl+Shift+S`: Open settings
- `Ctrl+Shift+Q`: Quit application

## 🔧 Troubleshooting

### Common Issues

#### "No module named 'OpenGL'"
```bash
pip install PyOpenGL PyOpenGL_accelerate
```

#### "Failed to initialize renderer"
- Update graphics drivers
- Check OpenGL support: [OpenGL Extensions Viewer](https://www.realtech-vr.com/home/glview)

#### "Azure Speech key error"
- Verify key and region in Azure portal
- Check network connection
- Ensure quota is not exceeded

#### "VRM model not loading"
- Check file path in config
- Ensure VRM file is valid
- Try with default models first

#### Poor Performance
- Reduce graphics settings:
  ```yaml
  graphics:
    target_fps: 30
    antialiasing: false
    window_width: 640
    window_height: 480
  ```
- Use local AI instead of cloud
- Close other applications

### Getting Help

1. Check logs in `logs/` directory
2. Enable debug mode: `debug: true` in config
3. Search issues on GitHub
4. Create new issue with logs

## Project Structure

After setup, your project should look like this:

```
vrm-ai-chatbot/
├── .venv/                 # Virtual environment (created during setup)
├── src/
│   ├── core/              # Core application logic
│   │   ├── __init__.py
│   │   ├── application.py # Main application class
│   │   ├── config.py      # Configuration management
│   │   └── event_bus.py   # Event system
│   ├── ai/                # AI conversation management
│   │   ├── __init__.py
│   │   └── conversation.py
│   ├── voice/             # Voice synthesis and recognition
│   │   ├── __init__.py
│   │   └── synthesis.py
│   ├── graphics/          # 3D rendering and VRM
│   │   ├── __init__.py
│   │   ├── renderer.py    # VRM renderer
│   │   ├── window.py      # Window management
│   │   └── shader_loader.py
│   ├── models/            # Data models
│   │   ├── __init__.py
│   │   └── character.py
│   ├── gui/               # GUI components
│   │   ├── __init__.py
│   │   └── splash.py      # Splash screen
│   ├── integrations/      # System integrations
│   │   └── __init__.py
│   └── utils/             # Utility functions
│       ├── __init__.py
│       ├── config.py
│       ├── event_bus.py
│       ├── logger.py
│       ├── shader_loader.py
│       └── splash.py
├── assets/
│   ├── models/            # VRM character models (add your .vrm files here)
│   ├── shaders/           # OpenGL shaders
│   │   ├── mtoon.vert
│   │   ├── mtoon.frag
│   │   ├── standard.vert
│   │   └── standard.frag
│   ├── textures/          # Additional textures
│   └── voices/            # Custom voice files
├── configs/
│   └── config.yaml        # Main configuration
├── data/
│   ├── characters/        # Character personality files
│   │   └── luna.yaml      # Sample character
│   └── conversations/     # Conversation history
├── logs/                  # Application logs
├── tests/                 # Test files
├── dev-tools/             # Development scripts (reference only)
│   ├── README.md          # Development tools documentation
│   ├── project_analysis.py
│   ├── project_setup.py
│   ├── renderer_setup.py
│   ├── voice_ai_setup.py
│   └── final_setup.py
├── .env                   # Environment variables
├── .env.template          # Environment template
├── main.py               # Application entry point
├── install.py            # Installation script
├── launch.bat            # Windows launcher
├── test_basic.py         # Basic environment test
├── test_components.py    # Component verification test
├── requirements.txt      # Python dependencies
├── requirements-core.txt # Core dependencies only
└── README.md             # Project documentation
```

## 🎨 Customization

### Creating Custom Personalities

1. Edit character traits in config or character files
2. Adjust AI system prompts
3. Customize voice settings
4. Create unique background stories

### Modifying Appearance

1. Use different VRM models
2. Adjust lighting and materials
3. Customize window transparency
4. Change animations and expressions

### Advanced Scripting

The system supports Python plugins for:
- Custom AI behaviors
- Special effects
- System integrations
- Voice processing

## Development Tools

The `dev-tools/` folder contains setup scripts that were used to generate this project:

- **project_analysis.py** - Technical analysis and feasibility study
- **project_setup.py** - Generated core application framework
- **renderer_setup.py** - Generated 3D graphics and VRM rendering
- **voice_ai_setup.py** - Generated voice synthesis and AI components
- **final_setup.py** - Generated final components and assets

**Note**: These scripts have already been executed during project creation. They are kept for reference and can be used to regenerate components if needed.

⚠️ **Warning**: Running these scripts will overwrite existing files!

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Reinstall dependencies
uv pip install -r requirements.txt
# OR
pip install -r requirements.txt
```

**Virtual Environment Issues**
```bash
# If using uv, ensure it's installed
# Visit: https://docs.astral.sh/uv/getting-started/installation/

# Recreate virtual environment
rm -rf .venv  # or rmdir /s .venv on Windows
uv venv --python 3.11
```

**OpenGL Issues**
- Update graphics drivers
- Ensure OpenGL 3.3+ support
- Try running with integrated graphics if discrete GPU fails

**Voice Synthesis Issues**
- Windows: Ensure SAPI voices are installed
- macOS: Check System Preferences > Accessibility > Speech
- Linux: Install espeak or festival

**Permission Issues**
- Run as administrator (Windows)
- Check file permissions
- Ensure antivirus isn't blocking files

**Unicode/Encoding Issues**
- Ensure your terminal supports UTF-8
- On Windows, use Windows Terminal or PowerShell
- Check that .env file is saved with UTF-8 encoding

## Next Steps

Once setup is complete:

1. **Verify Installation**: Run `python test_components.py` to ensure everything is working
2. **Add VRM Models**: Place .vrm files in `assets/models/`
3. **Configure Character**: Edit `data/characters/luna.yaml`
4. **Set API Keys**: Add your keys to `.env` (optional for local AI)
5. **Install Local AI** (Optional): Install Ollama from https://ollama.ai/ and run `ollama pull llama3.1:8b`
6. **Test Run**: Execute `python main.py`
7. **Customize**: Modify settings in `configs/config.yaml`

## Project Status

✅ **Project is fully set up and ready to use!**

- Virtual environment created with Python 3.11.13
- All dependencies installed successfully
- Complete project structure generated
- All components tested and verified
- Development tools organized in `dev-tools/`
- Ready for customization and deployment

For detailed usage instructions, see the main README.md file.

## 🔒 Privacy & Security

- Local AI keeps conversations private
- No data sent to servers when using local mode
- VRM models stored locally
- Optional cloud services are clearly marked

## 📄 License

This project is open source. Check individual VRM model licenses for usage rights.

---

Enjoy your new AI companion! 🤖✨
