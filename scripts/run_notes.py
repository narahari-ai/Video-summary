#!/usr/bin/env python3
import os
import yaml
import argparse

from app.mindmap_generator import MindMapGenerator
from app.note_generator import NoteGenerator
from app.utils import setup_logging, get_video_name

def generate_notes(summary_path: str):
    """Run the notes and mind map generation pipeline"""
    # Get video name for logging
    video_name = get_video_name(summary_path)
    logger = setup_logging(video_name)
    
    try:
        # Generate mind map
        logger.info("Generating mind map")
        mindmap_generator = MindMapGenerator()
        mindmap_path = mindmap_generator.generate(summary_path)
        logger.info(f"Mind map saved to: {mindmap_path}")
        
        # Generate notes and FAQs
        logger.info("Generating notes and FAQs")
        note_generator = NoteGenerator()
        notes_path, faq_path = note_generator.generate(summary_path)
        
        logger.info(f"Notes saved to: {notes_path}")
        logger.info(f"FAQs saved to: {faq_path}")
        
        return mindmap_path, notes_path, faq_path
        
    except Exception as e:
        logger.error(f"Error during notes generation: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Notes and Mind Map Generation")
    parser.add_argument("--summary", required=True,
                      help="Path to summary file")
    
    args = parser.parse_args()
    
    # Run notes generation
    generate_notes(args.summary)

if __name__ == "__main__":
    main()
