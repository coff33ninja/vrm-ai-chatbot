"""
Clear VRM cache to force fresh loading
Useful for testing or when encountering cache-related issues
"""
import os
import shutil
from pathlib import Path

def clear_cache():
    """Clear all VRM caches"""
    print("🧹 VRM Cache Cleaner")
    print("=" * 60)
    
    cache_locations = []
    
    # User cache directory
    try:
        user_cache = Path.home() / '.cache' / 'vrm_ai_chatbot'
        if user_cache.exists():
            cache_locations.append(('User Cache', user_cache))
    except Exception:
        pass
    
    # Repo cache directory
    try:
        repo_cache = Path(__file__).parent / '.vrm_cache'
        if repo_cache.exists():
            cache_locations.append(('Repo Cache', repo_cache))
    except Exception:
        pass
    
    if not cache_locations:
        print("✅ No cache directories found - nothing to clean")
        return
    
    print(f"\nFound {len(cache_locations)} cache location(s):\n")
    
    for name, path in cache_locations:
        print(f"📂 {name}: {path}")
        
        try:
            # Count files
            files = list(path.glob('*'))
            file_count = len(files)
            
            # Calculate size
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            size_mb = total_size / (1024 * 1024)
            
            print(f"   Files: {file_count}")
            print(f"   Size: {size_mb:.2f} MB")
            
            # Ask for confirmation
            response = input(f"\n   Delete this cache? (y/N): ").strip().lower()
            
            if response == 'y':
                shutil.rmtree(path)
                print(f"   ✅ Deleted!")
            else:
                print(f"   ⏭️  Skipped")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()
    
    print("=" * 60)
    print("✅ Cache cleaning complete!")

if __name__ == "__main__":
    try:
        clear_cache()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
