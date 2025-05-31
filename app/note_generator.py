import os
from typing import Dict, List, Tuple
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline

class NoteGenerator:
    def __init__(self):
        self.output_dir = "data/outputs/notes"
        self.faq_dir = "data/outputs/faqs"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.faq_dir, exist_ok=True)
        
        # Initialize pipelines for notes and FAQs
        # Use t5 model which is better at text2text tasks
        self.text2text = pipeline("text2text-generation", model="t5-base", max_length=512)
        self.qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    def generate(self, summary_path: str) -> Tuple[str, str]:
        """
        Generate structured notes and FAQs from summary
        
        Args:
            summary_path: Path to summary file
            
        Returns:
            Tuple[str, str]: Paths to saved notes and FAQ files
        """
        # Read summary
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = f.read()
            
        # Generate structured notes by breaking into sections
        notes = []
        
        # Break text into sentences and group into sections
        sentences = sent_tokenize(summary)
        section_size = max(3, len(sentences) // 5)  # Aim for 5 sections minimum
        
        for i in range(0, len(sentences), section_size):
            section = " ".join(sentences[i:i + section_size])
            # Generate bullet points using text2text generation
            prompt = f"summarize into bullet points: {section}"
            bullet_points = self.text2text(prompt, max_length=150, min_length=30)[0]['generated_text']
            notes.append(bullet_points.replace('\n', '\n- '))
        
        notes_text = "# Summary Notes\n\n" + "\n\n".join(notes)
        
        # Generate FAQs by extracting key topics and generating Q&A pairs
        faqs = ["# Frequently Asked Questions\n\n"]
        topics = self.text2text("extract main topics: " + summary[:1000], max_length=100)[0]['generated_text'].split(',')
        
        for topic in topics[:5]:  # Take top 5 topics
            question = f"What is {topic.strip()}?"
            answer = self.qa(question=question, context=summary)
            if answer['score'] > 0.1:  # Only include if confidence is reasonable
                faqs.append(f"\n### Q: {question}\n{answer['answer']}\n")
                    
        faq_text = "".join(faqs)
        
        # Save outputs
        summary_filename = os.path.basename(summary_path)
        notes_filename = f"{os.path.splitext(summary_filename)[0]}_notes.md"
        faq_filename = f"{os.path.splitext(summary_filename)[0]}_faqs.md"
        
        notes_path = os.path.join(self.output_dir, notes_filename)
        faq_path = os.path.join(self.faq_dir, faq_filename)
        
        with open(notes_path, "w", encoding="utf-8") as f:
            f.write(notes_text)
            
        with open(faq_path, "w", encoding="utf-8") as f:
            f.write(faq_text)
            
        return notes_path, faq_path
