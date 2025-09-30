# Development Tools

This folder contains the setup scripts that were used to generate the VRM AI Chatbot project structure and components.

## Scripts Overview

### 📊 `project_analysis.py`
- **Purpose**: Technical analysis and feasibility study
- **Generated**: `vrm_ai_project_analysis.json`
- **Contains**: Library recommendations, architecture analysis, complexity assessment

### 🏗️ `project_setup.py` 
- **Purpose**: Core application framework generator
- **Generated**: 
  - `main.py` - Application entry point
  - `src/core/application.py` - Main application class
  - `src/core/config.py` - Configuration system
  - Project directory structure

### 🎨 `renderer_setup.py`
- **Purpose**: 3D graphics and VRM rendering system
- **Generated**:
  - `src/graphics/renderer.py` - VRM renderer with MToon shaders
  - `src/graphics/window.py` - Transparent window system
  - `src/models/character.py` - Character model system

### 🎤 `voice_ai_setup.py`
- **Purpose**: Voice synthesis and AI conversation components
- **Generated**:
  - `src/voice/synthesis.py` - Multi-backend TTS system
  - `src/ai/conversation.py` - AI conversation manager
  - `src/utils/event_bus.py` - Event system
  - `src/utils/logger.py` - Logging utilities
  - `SETUP.md` - Detailed setup documentation

### 🎯 `final_setup.py`
- **Purpose**: Final components and assets
- **Generated**:
  - `src/gui/splash.py` - Splash screen
  - `assets/shaders/*.vert|.frag` - OpenGL shaders
  - `data/characters/luna.yaml` - Sample character
  - `install.py` - Installation script
  - `launch.bat` - Windows launcher

## Usage

These scripts have already been run and generated all the project components. They are kept here for:

- **Documentation** - Understanding how the project was built
- **Regeneration** - Re-creating components if needed
- **Reference** - Building similar projects
- **Maintenance** - Understanding the architecture

## Running Scripts

If you need to regenerate components:

```bash
# Activate virtual environment first
.venv/Scripts/activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Run individual scripts
python dev-tools/project_analysis.py
python dev-tools/project_setup.py
python dev-tools/renderer_setup.py
python dev-tools/voice_ai_setup.py
python dev-tools/final_setup.py
```

⚠️ **Warning**: Running these scripts will overwrite existing files!

## Project Status

✅ All scripts have been executed successfully
✅ All components have been generated
✅ Project is ready for use

The main application is now ready to run with `python main.py`