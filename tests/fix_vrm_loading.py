"""
VRM Loading Fix Test Script
This script tests and fixes the VRM loading issue.
"""

import sys
from pathlib import Path
from pygltflib import GLTF2

def test_vrm_loading(vrm_path: Path):
    """Test VRM loading and diagnose buffer issues."""
    print(f"\n{'='*60}")
    print(f"Testing VRM file: {vrm_path.name}")
    print(f"{'='*60}\n")
    
    if not vrm_path.exists():
        print(f"❌ File not found: {vrm_path}")
        return False
    
    try:
        # Load VRM file
        print("📂 Loading VRM file...")
        vrm_data = GLTF2().load(str(vrm_path))
        print("✅ VRM file loaded successfully")
        
        # Check binary_blob
        print(f"\n📊 Binary blob check:")
        if hasattr(vrm_data, 'binary_blob'):
            if vrm_data.binary_blob:
                print(f"  ✅ binary_blob exists: {len(vrm_data.binary_blob)} bytes")
            else:
                print(f"  ⚠️  binary_blob is None or empty")
        else:
            print(f"  ⚠️  No binary_blob attribute")
        
        # Check buffers
        print(f"\n📦 Buffer information:")
        print(f"  Total buffers: {len(vrm_data.buffers) if vrm_data.buffers else 0}")
        
        if vrm_data.buffers:
            for idx, buffer in enumerate(vrm_data.buffers):
                print(f"\n  Buffer {idx}:")
                print(f"    Length: {buffer.byteLength if hasattr(buffer, 'byteLength') else 'N/A'}")
                print(f"    URI: {buffer.uri if hasattr(buffer, 'uri') else 'None'}")
                
                if hasattr(buffer, 'uri') and buffer.uri:
                    if buffer.uri.startswith('data:'):
                        print(f"    Type: Data URI (embedded)")
                    else:
                        print(f"    Type: External file")
                        # Check if external file exists
                        ext_path = vrm_path.parent / buffer.uri
                        if ext_path.exists():
                            print(f"    ✅ External file exists: {ext_path}")
                        else:
                            print(f"    ❌ External file missing: {ext_path}")
                else:
                    print(f"    Type: GLB binary (no URI)")
        
        # Check accessors
        print(f"\n🔍 Accessor information:")
        print(f"  Total accessors: {len(vrm_data.accessors) if vrm_data.accessors else 0}")
        
        if vrm_data.accessors:
            # Test first few accessors
            for idx in range(min(3, len(vrm_data.accessors))):
                accessor = vrm_data.accessors[idx]
                print(f"\n  Accessor {idx}:")
                print(f"    bufferView: {accessor.bufferView}")
                print(f"    type: {accessor.type}")
                print(f"    componentType: {accessor.componentType}")
                print(f"    count: {accessor.count}")
        
        # Check buffer views
        print(f"\n📋 BufferView information:")
        print(f"  Total bufferViews: {len(vrm_data.bufferViews) if vrm_data.bufferViews else 0}")
        
        if vrm_data.bufferViews:
            for idx in range(min(3, len(vrm_data.bufferViews))):
                bv = vrm_data.bufferViews[idx]
                print(f"\n  BufferView {idx}:")
                print(f"    buffer: {bv.buffer}")
                print(f"    byteOffset: {bv.byteOffset}")
                print(f"    byteLength: {bv.byteLength}")
        
        # Try to get data from first buffer
        print(f"\n🔬 Testing buffer data extraction:")
        try:
            if vrm_data.buffers and len(vrm_data.buffers) > 0:
                buffer = vrm_data.buffers[0]
                buffer_uri = getattr(buffer, 'uri', None)
                
                print(f"  Attempting to get data from buffer 0...")
                buffer_data = vrm_data.get_data_from_buffer_uri(buffer_uri)
                
                if buffer_data:
                    print(f"  ✅ Successfully extracted buffer data: {len(buffer_data)} bytes")
                else:
                    print(f"  ❌ Buffer data is None")
        except Exception as e:
            print(f"  ❌ Error extracting buffer data: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading VRM: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test the default VRM file
    default_vrm = Path("E:/SCRIPTS/vrm-ai-chatbot/assets/models/AvatarSample_D.vrm")
    
    if len(sys.argv) > 1:
        vrm_path = Path(sys.argv[1])
    else:
        vrm_path = default_vrm
    
    success = test_vrm_loading(vrm_path)
    
    if success:
        print(f"\n✅ Test completed successfully!")
    else:
        print(f"\n❌ Test failed!")
    
    print(f"\n{'='*60}\n")
