#!/usr/bin/env python3
import os
import sys
import torch
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import whisper
import wget
import zipfile
import shutil
import traceback

# Enable debug mode for transformers
os.environ['TRANSFORMERS_VERBOSITY'] = 'debug'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_dir_if_not_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directory exists: {path}")

def download_whisper_model(model_name="base", models_dir="models/whisper"):
    """Download and save Whisper model locally"""
    try:
        logger.info(f"Downloading Whisper {model_name} model...")
        create_dir_if_not_exists(models_dir)
        model = whisper.load_model(model_name)
        save_path = os.path.join(models_dir, f"{model_name}.pt")
        torch.save(model.state_dict(), save_path)
        logger.info(f"Whisper {model_name} model saved to {save_path}")
    except Exception as e:
        logger.error(f"Error downloading Whisper model: {str(e)}")
        raise

def download_vosk_model(models_dir="models/vosk"):
    """Download and extract Vosk model"""
    try:
        logger.info("Downloading Vosk model...")
        create_dir_if_not_exists(models_dir)
        model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
        zip_path = os.path.join(models_dir, "vosk-model-small-en-us.zip")
        
        if not os.path.exists(zip_path):
            logger.info(f"Downloading from {model_url}")
            wget.download(model_url, zip_path)
            logger.info("\nDownload complete")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(models_dir)
        
        # Remove the zip file
        os.remove(zip_path)
        logger.info("Vosk model downloaded and extracted")
    except Exception as e:
        logger.error(f"Error downloading Vosk model: {str(e)}")
        raise

def download_transformer_model(model_name, save_dir):
    """Download and save transformer model locally"""
    try:
        logger.info(f"Downloading {model_name} model...")
        create_dir_if_not_exists(save_dir)
        
        # Download tokenizer and model
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Downloading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Save them locally
        logger.info(f"Saving to {save_dir}")
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)
        logger.info(f"{model_name} model saved to {save_dir}")
    except Exception as e:
        logger.error(f"Error downloading transformer model: {str(e)}")
        raise

def main():
    try:
        logger.info("Starting model downloads...")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, "models")

        # Ask which model to download
        print("\nAvailable models to download:")
        print("1. Whisper (base)")
        print("2. Vosk")
        print("3. BART")
        print("4. T5")
        print("5. Pegasus")
        print("6. GPT-2")
        print("7. MT5")
        print("8. All models")
        
        choice = input("\nEnter the number of the model to download (1-8): ")
        
        if choice in ['1', '8']:
            whisper_dir = os.path.join(models_dir, "whisper")
            download_whisper_model("base", whisper_dir)
        
        if choice in ['2', '8']:
            vosk_dir = os.path.join(models_dir, "vosk")
            download_vosk_model(vosk_dir)
        
        if choice in ['3', '8']:
            bart_dir = os.path.join(models_dir, "bart")
            download_transformer_model("facebook/bart-large-cnn", bart_dir)
        
        if choice in ['4', '8']:
            t5_dir = os.path.join(models_dir, "t5")
            download_transformer_model("t5-base", t5_dir)

        if choice in ['5', '8']:
            pegasus_dir = os.path.join(models_dir, "pegasus")
            download_transformer_model("google/pegasus-xsum", pegasus_dir)

        if choice in ['6', '8']:
            gpt2_dir = os.path.join(models_dir, "gpt2")
            download_transformer_model("gpt2", gpt2_dir)

        if choice in ['7', '8']:
            mt5_dir = os.path.join(models_dir, "mt5")
            download_transformer_model("google/mt5-small", mt5_dir)
        
        logger.info("Selected models downloaded successfully!")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
