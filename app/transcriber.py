import os
from typing import Dict, Optional
import whisper
from vosk import Model, KaldiRecognizer
import wave
import json
import logging
from pathlib import Path
from .utils import get_unique_filename

class BaseTranscriber:
    def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError

class WhisperTranscriber(BaseTranscriber):
    def __init__(self, config: Dict):
        self.model = whisper.load_model(config["size"])
        self.device = config.get("device", "cuda")
        self.language = config.get("language", "en")
        self.chunk_length = config.get("chunk_length", 30 * 60)  # 30 minutes in seconds
        self.logger = logging.getLogger(__name__)

    def transcribe(self, audio_path: str) -> str:
        try:
            self.logger.info(f"Loading audio file: {audio_path}")
            
            # Load audio file
            audio = whisper.load_audio(audio_path)
            duration = len(audio) / whisper.audio.SAMPLE_RATE
            self.logger.info(f"Audio duration: {duration/60:.2f} minutes")
            
            # Transcribe with safe defaults
            result = self.model.transcribe(
                audio_path,
                language=self.language,
                task="transcribe",
                verbose=True,
                initial_prompt="This is a lecture or educational video transcription.",
                condition_on_previous_text=True,
                word_timestamps=False  # Disable word timestamps to avoid CUDA error
            )
            
            transcript = result["text"]
            
            # Log some stats
            word_count = len(transcript.split())
            self.logger.info(f"Transcription complete. Word count: {word_count}")
            
            # Validate transcript
            if not transcript:
                raise ValueError("Empty transcription result")
                
            if word_count < 10:
                error_msg = f"Transcription seems too short (only {word_count} words)"
                self.logger.error(error_msg)
                
                # Try one more time with different parameters
                self.logger.info("Retrying transcription with different parameters...")
                result = self.model.transcribe(
                    audio_path,
                    language=self.language,
                    task="transcribe",
                    verbose=True,
                    beam_size=5,
                    best_of=5
                )
                transcript = result["text"]
                word_count = len(transcript.split())
                
                if word_count < 10:
                    raise ValueError(error_msg)
                    
            self.logger.info("Transcription successful")
            return transcript
            
        except Exception as e:
            self.logger.error(f"Error in Whisper transcription: {str(e)}")
            raise

class VoskTranscriber(BaseTranscriber):
    def __init__(self, config: Dict):
        model_path = config["model_path"]
        if not os.path.exists(model_path):
            raise ValueError(f"Vosk model not found at {model_path}")
        self.model = Model(model_path)

    def transcribe(self, audio_path: str) -> str:
        wf = wave.open(audio_path, "rb")
        rec = KaldiRecognizer(self.model, wf.getframerate())
        
        transcription = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if "text" in result and result["text"]:
                    transcription.append(result["text"])
        
        # Get final bits of audio
        final_result = json.loads(rec.FinalResult())
        if "text" in final_result and final_result["text"]:
            transcription.append(final_result["text"])
            
        return " ".join(transcription)

class TranscriberFactory:
    @staticmethod
    def create(model_name: str, config: Dict) -> BaseTranscriber:
        if model_name == "whisper_base":
            return WhisperTranscriber(config)
        elif model_name == "vosk":
            return VoskTranscriber(config)
        else:
            raise ValueError(f"Unsupported transcription model: {model_name}")

class Transcriber:
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = "data/outputs/transcripts"
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def transcribe(self, audio_path: str, model_name: str) -> str:
        """
        Transcribe audio using specified model and save result
        
        Args:
            audio_path: Path to audio file
            model_name: Name of the model to use
            
        Returns:
            str: Path to saved transcript
        """
        transcriber = TranscriberFactory.create(model_name, self.config[model_name])
        transcript = transcriber.transcribe(audio_path)
        
        # Save transcript with unique filename
        audio_filename = os.path.basename(audio_path)
        transcript_filename = f"{os.path.splitext(audio_filename)[0]}_{model_name}.txt"
        base_output_path = os.path.join(self.output_dir, transcript_filename)
        output_path = get_unique_filename(base_output_path)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcript)
            
        self.logger.info(f"Saved transcript to {output_path}")
        return output_path
