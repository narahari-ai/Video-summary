#!/usr/bin/env python3
import os
import yaml
import argparse
from typing import Dict

from app.video_loader import VideoLoader
from app.audio_extractor import AudioExtractor
from app.transcriber import Transcriber
from app.model_selector import ModelSelector
from app.utils import setup_logging, get_video_name, validate_config

def transcribe_video(config: Dict, video_source: str, model_name: str):
    """Run only the transcription pipeline"""
    # Initialize model selector
    model_selector = ModelSelector()
    
    # Validate model
    if not model_selector.validate_model(model_name, "transcription"):
        raise ValueError(f"Invalid transcription model: {model_name}")
    
    # Get video name for logging
    video_name = get_video_name(video_source)
    logger = setup_logging(video_name)
    
    try:
        # Load video
        logger.info(f"Loading video from {video_source}")
        video_loader = VideoLoader(config)
        video_path = video_loader.load(video_source)
        
        # Extract audio
        logger.info("Extracting audio")
        audio_extractor = AudioExtractor()
        audio_path = audio_extractor.extract(video_path)
        
        # Transcribe
        logger.info(f"Transcribing with {model_name}")
        transcriber = Transcriber(model_selector.models)
        transcript_path = transcriber.transcribe(audio_path, model_name)
        
        logger.info(f"Transcription complete! Output saved to: {transcript_path}")
        return transcript_path
        
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Video Transcription")
    parser.add_argument("--config", default="configs/config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--video", required=True,
                      help="Path to video file or URL")
    parser.add_argument("--model", required=True,
                      help="Name of transcription model to use")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Run transcription
    transcribe_video(config, args.video, args.model)

if __name__ == "__main__":
    main()
