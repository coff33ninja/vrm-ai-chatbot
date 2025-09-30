"""Test VRM loading with detailed diagnostics"""
import sys
import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_vrm_load.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)

async def test_vrm_load():
    """Test VRM model loading"""
    try:
        logger.info("Starting VRM load test...")
        
        # Import Character class
        from src.models.character import Character
        
        # Create character
        character = Character(name="Test Character")
        logger.info(f"Character created: {character.name}")
        
        # Load VRM model
        model_path = Path("assets/models/AvatarSample_D.vrm")
        logger.info(f"Loading VRM from: {model_path}")
        logger.info(f"File exists: {model_path.exists()}")
        logger.info(f"File size: {model_path.stat().st_size if model_path.exists() else 'N/A'} bytes")
        
        await character.load_vrm_model(model_path)
        
        logger.info(f"VRM loaded successfully!")
        logger.info(f"Has vrm_data: {character.vrm_data is not None}")
        
        # Safe binary_blob check
        blob_size = 0
        has_blob = False
        if character.vrm_data:
            try:
                if hasattr(character.vrm_data, 'binary_blob'):
                    blob_attr = getattr(character.vrm_data, 'binary_blob')
                    if isinstance(blob_attr, (bytes, bytearray, memoryview)):
                        has_blob = True
                        blob_size = len(blob_attr)
                        logger.info(f"Has binary_blob: {has_blob}")
                        logger.info(f"Binary blob size: {blob_size} bytes")
                    else:
                        logger.warning(f"binary_blob exists but is not bytes: {type(blob_attr)}")
            except Exception as e:
                logger.error(f"Error checking binary_blob: {e}")
        
        if not has_blob:
            logger.error("❌ binary_blob was not set properly!")
            return False
        
        # Test renderer  
        logger.info("Testing renderer...")
        from src.graphics.window import TransparentWindow
        from src.graphics.renderer import VRMRenderer
        
        window = TransparentWindow(width=800, height=600)
        renderer = VRMRenderer(window)
        
        await renderer.initialize()
        logger.info("Renderer initialized")
        
        await renderer.load_character(character)
        logger.info("Character loaded in renderer!")
        
        logger.info("✅ All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = asyncio.run(test_vrm_load())
    sys.exit(0 if success else 1)
