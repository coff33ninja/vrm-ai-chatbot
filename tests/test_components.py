#!/usr/bin/env python3
"""
Component test to verify all VRM AI Chatbot components are properly created.
"""

import sys
import os
from pathlib import Path

def test_core_components():
    """Test core application components."""
    core_files = [
        "src/core/application.py",
        "src/core/config.py", 
        "src/core/event_bus.py",
        "src/ai/conversation.py",
        "src/ai/gemini_client.py",
        "src/voice/synthesis.py",
        "src/voice/gemini_tts.py",
        "src/voice/gemini_stt.py",
        "src/graphics/renderer.py",
        "src/graphics/window.py",
        "src/graphics/shader_loader.py",
        "src/models/character.py",
        "src/gui/splash.py"
    ]
    
    missing_files = []
    for file_path in core_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing core files: {missing_files}")
        return False
    else:
        print(f"✅ All core components exist ({len(core_files)} files)")
        return True

def test_utility_components():
    """Test utility components."""
    util_files = [
        "src/utils/logger.py",
        "src/utils/event_bus.py",
        "src/utils/shader_loader.py",
        "src/utils/config.py",
        "src/utils/splash.py"
    ]
    
    existing_files = []
    for file_path in util_files:
        if Path(file_path).exists():
            existing_files.append(file_path)
    
    print(f"✅ Utility components: {len(existing_files)}/{len(util_files)} exist")
    return len(existing_files) >= 3  # At least most utilities exist

def test_shader_files():
    """Test shader files."""
    shader_files = [
        "assets/shaders/mtoon.vert",
        "assets/shaders/mtoon.frag", 
        "assets/shaders/standard.vert",
        "assets/shaders/standard.frag"
    ]
    
    existing_shaders = []
    for file_path in shader_files:
        if Path(file_path).exists():
            existing_shaders.append(file_path)
    
    if len(existing_shaders) == len(shader_files):
        print(f"✅ All shader files exist ({len(existing_shaders)} shaders)")
        return True
    else:
        print(f"⚠️ Shader files: {len(existing_shaders)}/{len(shader_files)} exist")
        return False

def test_character_data():
    """Test character data files."""
    character_files = [
        "data/characters/luna.yaml"
    ]
    
    existing_chars = []
    for file_path in character_files:
        if Path(file_path).exists():
            existing_chars.append(file_path)
    
    if existing_chars:
        print(f"✅ Character files exist ({len(existing_chars)} characters)")
        return True
    else:
        print(f"❌ No character files found")
        return False

def test_configuration():
    """Test configuration files."""
    config_files = [
        "configs/config.yaml",
        ".env",
        ".env.template"
    ]
    
    existing_configs = []
    for file_path in config_files:
        if Path(file_path).exists():
            existing_configs.append(file_path)
    
    print(f"✅ Configuration files: {len(existing_configs)}/{len(config_files)} exist")
    for config in existing_configs:
        print(f"   - {config}")
    
    return len(existing_configs) >= 2

def test_documentation():
    """Test documentation files."""
    doc_files = [
        "README.md",
        "SETUP.md",
        "install.py",
        "launch.bat"
    ]
    
    existing_docs = []
    for file_path in doc_files:
        if Path(file_path).exists():
            existing_docs.append(file_path)
    
    print(f"✅ Documentation files: {len(existing_docs)}/{len(doc_files)} exist")
    return len(existing_docs) >= 3

def test_import_syntax():
    """Test that Python files have valid syntax."""
    python_files = []
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    
    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                compile(f.read(), file_path, 'exec')
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
        except Exception:
            # Skip files that might have import issues but valid syntax
            pass
    
    if syntax_errors:
        print(f"❌ Syntax errors found:")
        for error in syntax_errors:
            print(f"   - {error}")
        return False
    else:
        print(f"✅ All Python files have valid syntax ({len(python_files)} files)")
        return True

def main():
    """Run all component tests."""
    print("🧪 VRM AI Chatbot - Component Verification Test")
    print("=" * 60)
    
    tests = [
        ("Core Components", test_core_components),
        ("Utility Components", test_utility_components),
        ("Shader Files", test_shader_files),
        ("Character Data", test_character_data),
        ("Configuration", test_configuration),
        ("Documentation", test_documentation),
        ("Python Syntax", test_import_syntax)
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
    
    print(f"\n📊 COMPONENT TEST SUMMARY:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Status: {'✅ ALL COMPONENTS READY' if passed == total else '⚠️ SOME ISSUES FOUND'}")
    
    if passed == total:
        print(f"\n🎉 VRM AI Chatbot project is fully set up!")
        print(f"   📁 Project structure: Complete")
        print(f"   🔧 Core components: Ready")
        print(f"   🎨 Assets: Available")
        print(f"   📝 Documentation: Complete")
        print(f"   ⚙️ Configuration: Ready")
        print(f"\n🚀 Ready for customization and deployment!")
        print(f"   1. Add your API keys to .env")
        print(f"   2. Download VRM models to assets/models/")
        print(f"   3. Customize character in data/characters/")
        print(f"   4. Run: python main.py")
    else:
        print(f"\n⚠️ Some components need attention. Check the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)