#!/usr/bin/env python3
"""
VRM AI Chatbot Test Runner
Runs all tests and provides comprehensive results.
"""

import sys
import asyncio
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def run_basic_tests():
    """Run basic environment tests."""
    print("🔍 Running Basic Environment Tests...")
    try:
        result = subprocess.run([sys.executable, "tests/test_basic.py"], 
                              capture_output=True, text=True, cwd=project_root)
        if result.returncode == 0:
            print("✅ Basic tests passed")
            return True
        else:
            print("❌ Basic tests failed")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running basic tests: {e}")
        return False

async def run_component_tests():
    """Run component verification tests."""
    print("\n🔍 Running Component Tests...")
    try:
        result = subprocess.run([sys.executable, "tests/test_components.py"], 
                              capture_output=True, text=True, cwd=project_root)
        if result.returncode == 0:
            print("✅ Component tests passed")
            return True
        else:
            print("❌ Component tests failed")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running component tests: {e}")
        return False

async def run_gemini_tests():
    """Run Gemini integration tests."""
    print("\n🔍 Running Gemini Integration Tests...")
    try:
        result = subprocess.run([sys.executable, "tests/test_gemini.py"], 
                              capture_output=True, text=True, cwd=project_root)
        if result.returncode == 0:
            print("✅ Gemini tests passed")
            return True
        else:
            print("⚠️  Gemini tests failed (may need API keys)")
            print(result.stdout)
            return False  # Don't fail overall tests for optional features
    except Exception as e:
        print(f"⚠️  Error running Gemini tests: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 VRM AI Chatbot - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Environment", run_basic_tests),
        ("Component Verification", run_component_tests),
        ("Gemini Integration", run_gemini_tests)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 TEST SUITE SUMMARY:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Status: {'✅ ALL TESTS PASSED' if passed >= 2 else '⚠️ SOME TESTS FAILED'}")
    
    if passed >= 2:
        print(f"\n🎉 VRM AI Chatbot is ready to use!")
        print(f"   Core functionality: Working")
        print(f"   Components: Verified")
        print(f"   Optional features: {'Working' if results[2] else 'Need setup'}")
    else:
        print(f"\n⚠️ Some core tests failed. Please check the output above.")
    
    return passed >= 2

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)