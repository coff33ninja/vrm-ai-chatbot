#!/usr/bin/env python3
"""
Test script for Gemini AI and voice integration.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_gemini_ai():
    """Test Gemini AI integration."""
    print("🧪 Testing Gemini AI Integration...")
    
    try:
        from src.ai.gemini_client import create_gemini_client
        
        # Check if API key is available
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("⚠️  GEMINI_API_KEY not found in environment")
            return False
        
        # Create client
        client = create_gemini_client(api_key)
        if not client:
            print("❌ Failed to create Gemini client")
            return False
        
        # Test conversation
        client.start_chat(system_prompt="You are a helpful AI assistant. Keep responses brief.")
        response = await client.generate_response("Hello! How are you today?")
        
        if response:
            print(f"✅ Gemini AI Response: {response[:100]}...")
            return True
        else:
            print("❌ No response from Gemini AI")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Gemini AI test failed: {e}")
        return False

async def test_gemini_tts():
    """Test Gemini TTS integration."""
    print("\n🧪 Testing Google Cloud TTS Integration...")
    
    try:
        from src.voice.gemini_tts import create_gemini_tts
        
        # Check if credentials are available
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            print("⚠️  GOOGLE_APPLICATION_CREDENTIALS not found in environment")
            return False
        
        if not Path(credentials_path).exists():
            print(f"⚠️  Credentials file not found: {credentials_path}")
            return False
        
        # Create TTS client
        tts_client = create_gemini_tts(credentials_path=credentials_path)
        if not tts_client:
            print("❌ Failed to create Google TTS client")
            return False
        
        # Test synthesis
        audio_file = await tts_client.synthesize_speech("Hello, this is a test of Google Cloud Text-to-Speech.")
        
        if audio_file and Path(audio_file).exists():
            print(f"✅ Google TTS synthesis successful: {audio_file}")
            # Clean up
            Path(audio_file).unlink(missing_ok=True)
            return True
        else:
            print("❌ Google TTS synthesis failed")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Google TTS test failed: {e}")
        return False

async def test_gemini_stt():
    """Test Gemini STT integration."""
    print("\n🧪 Testing Google Cloud STT Integration...")
    
    try:
        from src.voice.gemini_stt import create_gemini_stt
        
        # Check if credentials are available
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            print("⚠️  GOOGLE_APPLICATION_CREDENTIALS not found in environment")
            return False
        
        if not Path(credentials_path).exists():
            print(f"⚠️  Credentials file not found: {credentials_path}")
            return False
        
        # Create STT client
        stt_client = create_gemini_stt(credentials_path=credentials_path)
        if not stt_client:
            print("❌ Failed to create Google STT client")
            return False
        
        # Test getting audio devices
        devices = stt_client.get_audio_devices()
        if devices:
            print(f"✅ Google STT initialized with {len(devices)} audio devices")
            return True
        else:
            print("⚠️  No audio devices found, but STT client created successfully")
            return True
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Google STT test failed: {e}")
        return False

async def main():
    """Run all Gemini tests."""
    print("🚀 VRM AI Chatbot - Gemini Integration Test")
    print("=" * 50)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    tests = [
        ("Gemini AI", test_gemini_ai),
        ("Google Cloud TTS", test_gemini_tts),
        ("Google Cloud STT", test_gemini_stt)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 GEMINI TEST SUMMARY:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Status: {'✅ READY' if passed >= 1 else '❌ NEEDS SETUP'}")
    
    if passed >= 1:
        print(f"\n🎉 Gemini integration is working!")
        print(f"   You can now use Google's AI and voice services")
        print(f"   Configure in configs/config.yaml:")
        print(f"   - Set tts_engine: 'gemini' for Google TTS")
        print(f"   - Set stt_engine: 'gemini' for Google STT")
        print(f"   - Gemini AI will be used automatically if API key is set")
    else:
        print(f"\n⚠️  Gemini integration needs setup:")
        print(f"   1. Get Gemini API key from https://makersuite.google.com/app/apikey")
        print(f"   2. Add GEMINI_API_KEY to .env file")
        print(f"   3. For voice services, set up Google Cloud credentials")
        print(f"   4. Add GOOGLE_APPLICATION_CREDENTIALS to .env file")
    
    return passed >= 1

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)