import os
from typing import Dict, List
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    PegasusForConditionalGeneration,
    GPT2LMHeadModel,
    MT5ForConditionalGeneration
)
import torch
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
nltk.download('punkt')

class BaseSummarizer:
    def summarize(self, text: str) -> str:
        raise NotImplementedError

class TransformerSummarizer(BaseSummarizer):
    def __init__(self, config: Dict):
        print(f"Initializing TransformerSummarizer with config: {config}")
        self.model_path = config["model_path"]
        self.device = torch.device(config.get("device", "cuda"))
        self.max_length = config.get("max_length", 1024)
        self.min_length = config.get("min_length", 128)
        
        print(f"Loading tokenizer from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        print(f"Loading model from {self.model_path}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(self.device)
        print("Model loaded successfully")

    def _chunk_text(self, text: str, max_chunk_size: int = 1024) -> List[str]:
        # Split on periods and new lines for basic sentence tokenization
        sentences = [s.strip() for s in text.replace('\n', '. ').split('.') if s.strip()]
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            if current_length + sentence_tokens > max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def summarize(self, text: str) -> str:
        chunks = self._chunk_text(text)
        summaries = []
        
        for chunk in chunks:
            inputs = self.tokenizer.encode(
                "summarize: " + chunk,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True
            ).to(self.device)
            
            summary_ids = self.model.generate(
                inputs,
                max_length=self.max_length,
                min_length=self.min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        
        return " ".join(summaries)

class BartSummarizer(TransformerSummarizer):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_path).to(self.device)

class T5Summarizer(TransformerSummarizer):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_path).to(self.device)

class PegasusSummarizer(TransformerSummarizer):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model = PegasusForConditionalGeneration.from_pretrained(self.model_path).to(self.device)

class GPT2Summarizer(BaseSummarizer):
    def __init__(self, config: Dict):
        self.model_path = config["model_path"]
        self.device = torch.device(config.get("device", "cuda"))
        self.max_length = config.get("max_length", 1024)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_path).to(self.device)

    def summarize(self, text: str) -> str:
        # GPT-2 specific summarization with prompt
        prompt = f"Please summarize the following text:\n\n{text}\n\nSummary:"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        summary_ids = self.model.generate(
            inputs,
            max_length=self.max_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.split("Summary:")[1].strip()

class MT5Summarizer(TransformerSummarizer):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model = MT5ForConditionalGeneration.from_pretrained(self.model_path).to(self.device)

class SummarizerFactory:
    @staticmethod
    def create(model_name: str, config: Dict) -> BaseSummarizer:
        if model_name == "bart":
            return BartSummarizer(config)
        elif model_name == "t5":
            return T5Summarizer(config)
        elif model_name == "pegasus":
            return PegasusSummarizer(config)
        elif model_name == "gpt2":
            return GPT2Summarizer(config)
        elif model_name == "mt5":
            return MT5Summarizer(config)
        else:
            raise ValueError(f"Unsupported summarization model: {model_name}")

class Summarizer:
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = "data/outputs/summaries"
        os.makedirs(self.output_dir, exist_ok=True)

    def summarize(self, transcript_path: str, model_name: str) -> str:
        """
        Summarize transcript using specified model and save result
        
        Args:
            transcript_path: Path to transcript file
            model_name: Name of the model to use
            
        Returns:
            str: Path to saved summary
        """
        # Read transcript
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read()
        
        summarizer = SummarizerFactory.create(model_name, self.config[model_name])
        summary = summarizer.summarize(transcript)
        
        # Save summary
        transcript_filename = os.path.basename(transcript_path)
        summary_filename = f"{os.path.splitext(transcript_filename)[0]}_{model_name}.md"
        output_path = os.path.join(self.output_dir, summary_filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary)
            
        return output_path
