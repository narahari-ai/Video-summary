#!/usr/bin/env python3
import os
import sys
import logging
import yaml
from pathlib import Path
from app.video_loader import VideoLoader
from app.audio_extractor import AudioExtractor
from app.transcriber import TranscriberFactory
from app.summarizer import SummarizerFactory
from app.mindmap_generator import MindMapGenerator
from app.note_generator import NoteGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    with open("configs/model_profiles.yaml") as f:
        model_profiles = yaml.safe_load(f)
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    return model_profiles, config

def test_video_loading(video_path):
    logger.info(f"Testing video loading for: {video_path}")
    try:
        loader = VideoLoader()
        metadata = loader.load_metadata(video_path)
        logger.info(f"✓ Video loaded successfully - Duration: {metadata.get('duration')}s")
        return True
    except Exception as e:
        logger.error(f"Error loading video: {str(e)}")
        return False

def test_audio_extraction(video_path, audio_path):
    logger.info(f"Testing audio extraction for: {video_path}")
    try:
        extractor = AudioExtractor()
        extractor.extract_audio(video_path, audio_path)
        if os.path.exists(audio_path):
            logger.info("✓ Audio extracted successfully")
            return True
        else:
            logger.error("Audio file was not created")
            return False
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        return False

def test_transcription(audio_path, model_profiles):
    results = {}
    
    # Test Whisper
    logger.info("Testing Whisper transcription...")
    try:
        whisper_config = model_profiles["whisper_base"]
        transcriber = TranscriberFactory.create("whisper_base", whisper_config)
        transcript = transcriber.transcribe(audio_path)
        logger.info("✓ Whisper transcription successful")
        results["whisper"] = True
    except Exception as e:
        logger.error(f"Error in Whisper transcription: {str(e)}")
        results["whisper"] = False

    # Test Vosk
    logger.info("Testing Vosk transcription...")
    try:
        vosk_config = model_profiles["vosk"]
        transcriber = TranscriberFactory.create("vosk", vosk_config)
        transcript = transcriber.transcribe(audio_path)
        logger.info("✓ Vosk transcription successful")
        results["vosk"] = True
    except Exception as e:
        logger.error(f"Error in Vosk transcription: {str(e)}")
        results["vosk"] = False

    return results

def test_summarization(text, model_profiles):
    results = {}
    
    # Test BART
    logger.info("Testing BART summarization...")
    try:
        bart_config = model_profiles["bart"]
        summarizer = SummarizerFactory.create("bart", bart_config)
        summary = summarizer.summarize(text)
        logger.info("✓ BART summarization successful")
        results["bart"] = True
    except Exception as e:
        logger.error(f"Error in BART summarization: {str(e)}")
        results["bart"] = False

    # Test T5
    logger.info("Testing T5 summarization...")
    try:
        t5_config = model_profiles["t5"]
        summarizer = SummarizerFactory.create("t5", t5_config)
        summary = summarizer.summarize(text)
        logger.info("✓ T5 summarization successful")
        results["t5"] = True
    except Exception as e:
        logger.error(f"Error in T5 summarization: {str(e)}")
        results["t5"] = False

    return results

def main():
    # Load configurations
    model_profiles, config = load_config()
    
    # Get video files
    video_dir = Path("data/videos")
    video_files = [f for f in video_dir.glob("*.mp4") if not f.name.endswith(".Zone.Identifier")]
    
    if not video_files:
        logger.error("No video files found in data/videos/")
        return 1

    all_tests_passed = True
    
    for video_file in video_files:
        logger.info(f"\nTesting with video: {video_file.name}")
        
        # Create test output paths
        audio_path = f"data/audios/test_{video_file.stem}.wav"
        
        # Test video loading
        if not test_video_loading(str(video_file)):
            all_tests_passed = False
            continue
            
        # Test audio extraction
        if not test_audio_extraction(str(video_file), audio_path):
            all_tests_passed = False
            continue
            
        # Test transcription
        trans_results = test_transcription(audio_path, model_profiles)
        if not all(trans_results.values()):
            all_tests_passed = False
            
        # Test summarization with a sample text
        sample_text = "This is a test text for summarization. It needs to be long enough to be meaningful. " * 5
        sum_results = test_summarization(sample_text, model_profiles)
        if not all(sum_results.values()):
            all_tests_passed = False
            
        # Clean up test audio file
        try:
            os.remove(audio_path)
        except:
            pass
    
    if all_tests_passed:
        logger.info("\n✓ All tests completed successfully!")
        return 0
    else:
        logger.error("\n✗ Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
