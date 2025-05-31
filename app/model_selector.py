import os
from typing import Dict, List
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import whisper

class ModelSelector:
    def __init__(self, config_path: str = "configs/model_profiles.yaml"):
        self.config_path = config_path
        self.models = self._load_config()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.base_dir, "models")

    def _load_config(self) -> Dict:
        """Load model configurations from YAML file"""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def get_transcription_models(self) -> List[str]:
        """Get list of available transcription models"""
        return [name for name, config in self.models.items() 
                if config["type"] == "transcription"]

    def get_summarization_models(self) -> List[str]:
        """Get list of available summarization models"""
        return [name for name, config in self.models.items() 
                if config["type"] == "summarization"]

    def get_model_config(self, model_name: str) -> Dict:
        """Get configuration for specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in config")
        return self.models[model_name]

    def validate_model(self, model_name: str, task_type: str) -> bool:
        """Validate if model exists and supports given task"""
        if model_name not in self.models:
            return False
        return self.models[model_name]["type"] == task_type

    def load_whisper_model(self, model_name: str):
        """Load local Whisper model"""
        model_path = os.path.join(self.models_dir, "whisper", f"{model_name}.pt")
        if not os.path.exists(model_path):
            raise ValueError(f"Whisper model not found at {model_path}. Please run download_models.py first.")
        model = whisper.load_model(model_name)
        model.load_state_dict(torch.load(model_path))
        return model

    def load_transformer_model(self, model_type: str):
        """Load local transformer model (BART or T5)"""
        model_dir = os.path.join(self.models_dir, model_type)
        if not os.path.exists(model_dir):
            raise ValueError(f"Model not found at {model_dir}. Please run download_models.py first.")
        
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        return model, tokenizer

    def load_vosk_model(self):
        """Load local Vosk model"""
        model_dir = os.path.join(self.models_dir, "vosk")
        if not os.path.exists(model_dir):
            raise ValueError(f"Vosk model not found at {model_dir}. Please run download_models.py first.")
        return model_dir  # Vosk will load the model from this directory
