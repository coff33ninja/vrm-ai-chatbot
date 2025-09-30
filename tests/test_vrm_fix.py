"""
Quick test to verify VRM loading fixes
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import asyncio
from src.models.character import Character

async def test_vrm_loading():
    print("=" * 60)
    print("Testing VRM Loading Fix")
    print("=" * 60)
    
    # Create a character
    character = Character(name="Test Character")
    
    # Load VRM
    vrm_path = Path("E:/SCRIPTS/vrm-ai-chatbot/assets/models/AvatarSample_D.vrm")
    print(f"\n📂 Loading VRM: {vrm_path.name}")
    
    try:
        await character.load_vrm_model(vrm_path)
        print(f"✅ VRM loaded successfully!")
        
        # Check binary_blob
        if hasattr(character.vrm_data, 'binary_blob') and character.vrm_data.binary_blob:
            print(f"✅ binary_blob present: {len(character.vrm_data.binary_blob)} bytes")
        else:
            print(f"❌ binary_blob missing or None")
            return False
        
        # Check buffers
        if character.vrm_data.buffers:
            print(f"✅ Buffers: {len(character.vrm_data.buffers)}")
        
        # Check accessors
        if character.vrm_data.accessors:
            print(f"✅ Accessors: {len(character.vrm_data.accessors)}")
            
            # Try to access first accessor's buffer
            try:
                accessor = character.vrm_data.accessors[0]
                if accessor.bufferView is not None:
                    buffer_view = character.vrm_data.bufferViews[accessor.bufferView]
                    buffer = character.vrm_data.buffers[buffer_view.buffer]
                    
                    # Try to get buffer data
                    if character.vrm_data.binary_blob:
                        test_offset = buffer_view.byteOffset or 0
                        test_data = character.vrm_data.binary_blob[test_offset:test_offset+10]
                        print(f"✅ Successfully accessed buffer data: {len(test_data)} bytes sample")
                    else:
                        print(f"⚠️ Cannot test buffer access: binary_blob is None")
            except Exception as e:
                print(f"❌ Error accessing buffer: {e}")
                return False
        
        print(f"\n{'='*60}")
        print(f"✅ All tests passed!")
        print(f"{'='*60}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading VRM: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_vrm_loading())
    sys.exit(0 if success else 1)
