#!/usr/bin/env python3
import os
import yaml
import argparse
from typing import Dict
import time

from app.video_loader import VideoLoader
from app.audio_extractor import AudioExtractor
from app.transcriber import Transcriber
from app.summarizer import Summarizer
from app.mindmap_generator import MindMapGenerator
from app.note_generator import NoteGenerator
from app.model_selector import ModelSelector
from app.utils import setup_logging, get_video_name, validate_config, clean_output_directories

def validate_file_output(file_path: str, min_size_bytes: int = 100) -> bool:
    """Validate that a file exists and has minimum content"""
    if not os.path.exists(file_path):
        return False
    return os.path.getsize(file_path) >= min_size_bytes

def load_config(config_path: str) -> Dict:
    """Load and validate configuration"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    if not validate_config(config):
        raise ValueError("Invalid configuration file")
        
    return config

def process_video(config: Dict, video_source: str):
    """Process video through the entire pipeline"""
    # Initialize model selector
    model_selector = ModelSelector()
    
    # Get video name for logging
    video_name = get_video_name(video_source)
    logger = setup_logging(video_name)
    
    try:
        # Load video
        logger.info(f"Loading video from {video_source}")
        video_loader = VideoLoader(config)
        video_path = video_loader.load(video_source)
        if not validate_file_output(video_path):
            raise ValueError("Video loading failed")
        
        # Extract audio
        logger.info("Extracting audio")
        audio_extractor = AudioExtractor()
        audio_path = audio_extractor.extract(video_path)
        if not validate_file_output(audio_path):
            raise ValueError("Audio extraction failed")
        
        # Transcribe with each model
        logger.info("Starting transcription")
        transcriber = Transcriber(model_selector.models)
        transcripts = {}
        
        for model_name in config["models"]["transcription"]:
            logger.info(f"Transcribing with {model_name}")
            transcript_path = transcriber.transcribe(audio_path, model_name)
            
            # Validate transcript
            if not validate_file_output(transcript_path, min_size_bytes=500):  # Expect reasonable transcript size
                logger.error(f"Transcript from {model_name} seems too short")
                continue
                
            transcripts[model_name] = transcript_path
            logger.info(f"Transcription with {model_name} completed successfully")
            
        if not transcripts:
            raise ValueError("All transcription attempts failed")
            
        # Generate summaries
        logger.info("Generating summaries")
        summarizer = Summarizer(model_selector.models)
        summaries = {}
        
        for transcript_model, transcript_path in transcripts.items():
            for summary_model in config["models"]["summarization"]:
                logger.info(f"Summarizing {transcript_model} transcript with {summary_model}")
                summary_path = summarizer.summarize(transcript_path, summary_model)
                
                # Validate summary
                if not validate_file_output(summary_path, min_size_bytes=200):  # Expect reasonable summary size
                    logger.error(f"Summary using {summary_model} seems too short")
                    continue
                    
                summaries[f"{transcript_model}_{summary_model}"] = summary_path
                logger.info(f"Summarization with {summary_model} completed successfully")
        
        if not summaries:
            raise ValueError("All summarization attempts failed")
                
        # Generate mind maps and notes
        mindmap_generator = MindMapGenerator()
        note_generator = NoteGenerator()
        
        for model_combo, summary_path in summaries.items():
            try:
                # Generate mind map
                logger.info(f"Generating mind map for {model_combo}")
                mindmap_path = mindmap_generator.generate(summary_path)
                if not validate_file_output(mindmap_path):
                    logger.error(f"Mind map generation failed for {model_combo}")
                
                # Generate notes and FAQs
                logger.info(f"Generating notes and FAQs for {model_combo}")
                notes_path, faq_path = note_generator.generate(summary_path)
                if not (validate_file_output(notes_path) and validate_file_output(faq_path)):
                    logger.error(f"Notes/FAQ generation failed for {model_combo}")
            except Exception as e:
                logger.error(f"Error processing {model_combo}: {str(e)}")
                continue
            
        logger.info("Processing complete!")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Video Intelligence Assistant")
    parser.add_argument("--config", default="configs/config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--video", required=True,
                      help="Path to video file or URL")
    parser.add_argument("--clean", choices=['none', 'video', 'all'], default='none',
                      help="Clean output files: 'none' (default), 'video' (clean only this video's outputs), 'all' (clean all outputs)")
    
    args = parser.parse_args()
    video_name = get_video_name(args.video)
    
    # Clean outputs if requested
    if args.clean == 'all':
        clean_output_directories()
    elif args.clean == 'video':
        clean_output_directories(video_name)
    
    # Load configuration
    config = load_config(args.config)
    
    # Process video
    process_video(config, args.video)

if __name__ == "__main__":
    main()
