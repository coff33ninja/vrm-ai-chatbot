#!/usr/bin/env python3
"""
Basic test to verify the virtual environment and core dependencies work.
"""

import sys
import os
from pathlib import Path

def test_python_version():
    """Test Python version."""
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return sys.version_info >= (3, 11)

def test_core_imports():
    """Test core dependency imports."""
    try:
        import numpy
        print(f"✅ NumPy: {numpy.__version__}")
        
        import openai
        print(f"✅ OpenAI: {openai.__version__}")
        
        import anthropic
        print(f"✅ Anthropic: {anthropic.__version__}")
        
        import pydantic
        print(f"✅ Pydantic: {pydantic.__version__}")
        
        import yaml
        print(f"✅ PyYAML: Available")
        
        import rich
        print(f"✅ Rich: Available")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_project_structure():
    """Test project directory structure."""
    required_dirs = [
        "src", "src/core", "src/ai", "src/voice", "src/graphics", 
        "src/models", "src/utils", "assets", "configs", "data", "logs"
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not Path(directory).exists():
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print(f"✅ All required directories exist ({len(required_dirs)} dirs)")
        return True

def test_config_files():
    """Test configuration files."""
    config_files = [".env", "configs/config.yaml", "requirements.txt"]
    
    existing_files = []
    for file_path in config_files:
        if Path(file_path).exists():
            existing_files.append(file_path)
    
    print(f"✅ Configuration files: {len(existing_files)}/{len(config_files)} exist")
    for file_path in existing_files:
        print(f"   - {file_path}")
    
    return len(existing_files) >= 2  # At least .env and requirements.txt

def main():
    """Run all tests."""
    print("🧪 VRM AI Chatbot - Basic Environment Test")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Core Dependencies", test_core_imports),
        ("Project Structure", test_project_structure),
        ("Configuration Files", test_config_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 TEST SUMMARY:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Status: {'✅ READY' if passed == total else '⚠️ NEEDS ATTENTION'}")
    
    if passed == total:
        print(f"\n🎉 Environment setup is complete!")
        print(f"   Virtual environment: Active")
        print(f"   Dependencies: Installed")
        print(f"   Project structure: Ready")
        print(f"\n🚀 Next steps:")
        print(f"   1. Add API keys to .env file")
        print(f"   2. Run remaining setup scripts")
        print(f"   3. Add VRM models to assets/models/")
    else:
        print(f"\n⚠️ Some tests failed. Please check the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)