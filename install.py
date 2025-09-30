#!/usr/bin/env python3
"""
VRM AI Chatbot Installation Script
Automated setup for dependencies and configuration.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 11):
        print("❌ Python 3.11 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def install_dependencies():
    """Install Python dependencies."""
    print("📦 Assuming dependencies are installed (e.g., via 'uv pip install'). Skipping.")
    return True

def setup_directories():
    """Create necessary directories."""
    print("📁 Setting up directories...")
    
    directories = [
        "data/characters",
        "data/conversations", 
        "logs",
        "assets/models",
        "assets/voices",
        "assets/textures"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")

def create_env_file():
    """Create .env file from template if it doesn't exist."""
    if not Path('.env').exists():
        if Path('.env.template').exists():
            print("📝 Creating .env file from template...")
            import shutil
            shutil.copy('.env.template', '.env')
            print("✅ .env file created - please add your API keys")
        else:
            print("⚠️  No .env template found - you'll need to create .env manually")
    else:
        print("✅ .env file already exists")

def check_optional_dependencies():
    """Check for optional dependencies and suggest installation."""
    print("🔍 Checking optional dependencies...")
    
    optional_deps = {
        'pygame': 'Audio playback support',
        'opencv-python': 'Enhanced computer vision',
        'librosa': 'Advanced audio processing',
        'sounddevice': 'Audio device management'
    }
    
    missing = []
    for dep, description in optional_deps.items():
        try:
            __import__(dep.replace('-', '_'))
        except ImportError:
            missing.append((dep, description))
    
    if missing:
        print("⚠️  Optional dependencies not found:")
        for dep, desc in missing:
            print(f"   - {dep}: {desc}")
        print("   Install with: pip install " + " ".join([dep for dep, _ in missing]))
    else:
        print("✅ All optional dependencies found")

def download_ollama():
    """Provide instructions for Ollama installation."""
    print("🤖 Local AI Setup (Ollama):")
    print("   1. Download Ollama from: https://ollama.ai/")
    print("   2. Install and run: ollama pull llama3.1:8b")
    print("   3. This enables private, local AI conversations")

def main():
    """Main installation process."""
    print("🚀 VRM AI Chatbot Installation")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Create .env file
    create_env_file()
    
    # Check optional dependencies
    check_optional_dependencies()
    
    # Ollama instructions
    download_ollama()
    
    print("\n🎉 Installation complete!")
    print("\n📋 Next steps:")
    print("   1. Add your API keys to .env file")
    print("   2. Place VRM models in assets/models/")
    print("   3. Run: python main.py")
    print("\n📖 See SETUP.md for detailed configuration guide")

if __name__ == "__main__":
    main()
