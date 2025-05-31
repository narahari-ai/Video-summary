#!/usr/bin/env python3
import sys
import logging
from pathlib import Path
import yaml
from app.transcriber import TranscriberFactory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    config_path = Path("configs/model_profiles.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

def test_whisper():
    logger.info("Testing Whisper model...")
    try:
        config = load_config()["whisper_base"]
        transcriber = TranscriberFactory.create("whisper_base", config)
        logger.info("✓ Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading Whisper model: {str(e)}")
        return False
    return True

def test_vosk():
    logger.info("Testing Vosk model...")
    try:
        config = load_config()["vosk"]
        transcriber = TranscriberFactory.create("vosk", config)
        logger.info("✓ Vosk model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading Vosk model: {str(e)}")
        return False
    return True

def main():
    whisper_ok = test_whisper()
    vosk_ok = test_vosk()
    
    if whisper_ok and vosk_ok:
        logger.info("All models are working correctly!")
        return 0
    else:
        logger.error("Some models failed to load.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
