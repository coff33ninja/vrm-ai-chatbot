#!/usr/bin/env python3
"""
VRM AI Chatbot - Main Application Entry Point
A desktop AI companion with VRM avatar support, voice interaction, and system integration.

Features:
- 3D VRM avatar rendering with real-time animation
- Transparent desktop overlay window
- Local and cloud AI integration
- Voice synthesis and speech recognition
- LiveKit video calling support
- Windows system integration
- Customizable personality traits

Author: VRM AI Chatbot Project
Version: 1.0.0
Python: 3.11+
"""

import sys
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import core modules
from src.core.application import VRMAIApplication
from src.core.config import Config, load_config
from src.utils.logger import setup_logging
from src.gui.splash import SplashScreen

def check_python_version():
    """Ensure compatible Python version."""
    if sys.version_info < (3, 11):
        print("❌ Python 3.11 or higher is required!")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'numpy', 'OpenGL', 'moderngl', 'tkinter', 'pyttsx3', 
        'speech_recognition', 'openai', 'livekit', 'pygltflib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'OpenGL':
                import OpenGL.GL
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nPlease install missing packages with:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    print("✅ All required dependencies are installed")

def main():
    """Main application entry point."""
    print("🤖 VRM AI Chatbot - Starting Application...")
    print("=" * 50)
    
    # Check system requirements
    check_python_version()
    check_dependencies()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting VRM AI Chatbot Application")
    
    try:
        # Load configuration
        config = load_config()
        logger.info(f"Configuration loaded: {config.app_name}")
        
        # Show splash screen
        splash = SplashScreen()
        splash.show()
        
        # Initialize and run the main application
        app = VRMAIApplication(config)

        # Hook splash to event bus so it can receive real progress updates
        try:
            # Subscribe to startup_progress events
            def _on_startup_progress(progress, status):
                # Called on event bus; forward to splash
                try:
                    splash.update_progress(progress, status)
                except Exception:
                    pass

            # The application creates its own EventBus on init; we'll subscribe
            app.event_bus.subscribe("startup_progress", _on_startup_progress)
        except Exception:
            # If event bus isn't ready yet, continue; splash will still show simulated progress
            pass
        
        # Start the application event loop
        asyncio.run(app.run())
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"❌ Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
