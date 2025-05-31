#!/usr/bin/env python3
import os
import yaml
import argparse
from typing import Dict

from app.summarizer import Summarizer
from app.model_selector import ModelSelector
from app.utils import setup_logging, get_video_name

def summarize_transcript(config: Dict, transcript_path: str, model_name: str):
    """Run only the summarization pipeline"""
    # Initialize model selector
    model_selector = ModelSelector()
    
    # Validate model
    if not model_selector.validate_model(model_name, "summarization"):
        raise ValueError(f"Invalid summarization model: {model_name}")
    
    # Get video name for logging
    video_name = get_video_name(transcript_path)
    logger = setup_logging(video_name)
    
    try:
        # Summarize
        logger.info(f"Summarizing with {model_name}")
        summarizer = Summarizer(model_selector.models)
        summary_path = summarizer.summarize(transcript_path, model_name)
        
        logger.info(f"Summarization complete! Output saved to: {summary_path}")
        return summary_path
        
    except Exception as e:
        logger.error(f"Error during summarization: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Transcript Summarization")
    parser.add_argument("--config", default="configs/config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--transcript", required=True,
                      help="Path to transcript file")
    parser.add_argument("--model", required=True,
                      help="Name of summarization model to use")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Run summarization
    summarize_transcript(config, args.transcript, args.model)

if __name__ == "__main__":
    main()
