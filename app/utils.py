import os
import logging
from datetime import datetime
from typing import Dict, Optional
import shutil
from pathlib import Path
import re
import traceback

def setup_logging(video_name: str) -> logging.Logger:
    """Set up logging for the current video processing run"""
    # Create logs directory if it doesn't exist
    log_dir = "data/outputs/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(f"video_ai_{video_name}")
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Create unique log file for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{video_name}_{timestamp}.log")
    error_log_file = os.path.join(log_dir, f"{video_name}_{timestamp}_error.log")
    
    # Create file handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    error_file_handler = logging.FileHandler(error_log_file)
    error_file_handler.setLevel(logging.ERROR)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatters
    standard_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    error_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n%(exc_info)s\n')
    
    # Set formatters
    file_handler.setFormatter(standard_formatter)
    error_file_handler.setFormatter(error_formatter)
    console_handler.setFormatter(standard_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(error_file_handler)
    logger.addHandler(console_handler)
    
    return logger

def error_handler(logger: logging.Logger):
    """Decorator for handling and logging errors"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                tb = traceback.format_exc()
                logger.error(error_msg, extra={'traceback': tb})
                raise
        return wrapper
    return decorator

def get_video_name(video_path: str) -> str:
    """Extract video name from path without extension"""
    return os.path.splitext(os.path.basename(video_path))[0]

def validate_config(config: Dict) -> bool:
    """Validate configuration file structure"""
    required_keys = ["video_source", "models"]
    if not all(key in config for key in required_keys):
        return False
        
    if "models" in config:
        if not isinstance(config["models"], dict):
            return False
        if not all(key in config["models"] for key in ["transcription", "summarization"]):
            return False
            
    return True

def clean_output_directories(video_name: Optional[str] = None):
    """Clean output directories for a specific video or all outputs if no video specified"""
    output_dirs = [
        "data/outputs/transcripts",
        "data/outputs/summaries",
        "data/outputs/mindmaps",
        "data/outputs/notes",
        "data/outputs/faqs",
        "data/audios"
    ]
    
    for dir_path in output_dirs:
        if not os.path.exists(dir_path):
            continue
            
        if video_name:
            # Remove only files related to the specific video
            for file in os.listdir(dir_path):
                if file.startswith(video_name):
                    os.remove(os.path.join(dir_path, file))
        else:
            # Remove all files in directory
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)

def get_unique_filename(base_path: str) -> str:
    """Generate a unique filename by appending a counter if file exists"""
    if not os.path.exists(base_path):
        return base_path
        
    directory = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)
    
    counter = 1
    while os.path.exists(base_path):
        # Check if name already has a counter
        match = re.match(r"^(.+)_(\d+)$", name)
        if match:
            name = match.group(1)
            
        new_name = f"{name}_{counter}{ext}"
        base_path = os.path.join(directory, new_name)
        counter += 1
    
    return base_path
