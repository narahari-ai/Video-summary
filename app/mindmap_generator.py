import os
from typing import Dict, List
import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import graphviz
import re

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class MindMapGenerator:
    def __init__(self):
        self.output_dir = "data/outputs/mindmaps"
        os.makedirs(self.output_dir, exist_ok=True)
        self.stop_words = set(stopwords.words('english'))

    def generate(self, summary_path: str) -> str:
        """
        Generate mind map from summary and save as image
        
        Args:
            summary_path: Path to summary file
            
        Returns:
            str: Path to saved mind map image
        """
        # Read summary
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = f.read()
            
        # Extract key concepts and relationships
        concepts = self._extract_concepts(summary)
        relationships = self._build_relationships(concepts)
        
        # Create mind map
        print(f"Creating mindmap from {len(relationships)} relationships")
        mindmap = self._create_graphviz(relationships)
        
        # Save mind map
        summary_filename = os.path.basename(summary_path)
        mindmap_filename = f"{os.path.splitext(summary_filename)[0]}_mindmap"
        print(f"Saving mindmap to {mindmap_filename}")
        output_path = os.path.join(self.output_dir, mindmap_filename)
        
        mindmap.render(output_path, format="png", cleanup=True)
        
        return f"{output_path}.png"

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        print(f"Extracting concepts from text of length: {len(text)}")
        # Split into sentences and normalize
        sentences = sent_tokenize(text.lower())
        print(f"Found {len(sentences)} sentences")
        
        # Extract key phrases and important words
        concepts = set()
        for sentence in sentences:
            # Split into words and clean up
            words = [w for w in word_tokenize(sentence) 
                    if w.isalnum() and w not in self.stop_words and len(w) > 2]
            
            # Look for consecutive capitalized words or important word patterns
            phrase = []
            for i, word in enumerate(words):
                # Keep track of potential concepts
                if (word[0].isupper() or  # Capitalized words
                    re.match(r'^(test|data|value|analysis|hypothesis|probability)', word)):  # Key terms
                    phrase.append(word)
                elif len(phrase) > 0:  # End of phrase
                    if len(phrase) > 1:  # Only keep multi-word phrases
                        concepts.add(" ".join(phrase))
                    phrase = []
            
            # Add any remaining phrases
            if len(phrase) > 1:
                concepts.add(" ".join(phrase))
                
            # Also add any individual important words
            concepts.update([w for w in words if re.match(
                r'^(test|data|value|analysis|hypothesis|probability|normal|distribution)', w)])
        
        return list(concepts)

    def _build_relationships(self, concepts: List[str]) -> List[tuple]:
        """Build relationships between concepts based on word overlap"""
        relationships = []
        
        for i, concept1 in enumerate(concepts):
            words1 = set(word_tokenize(concept1.lower()))
            
            for concept2 in concepts[i+1:]:
                words2 = set(word_tokenize(concept2.lower()))
                
                # If concepts share words or are substrings, create relationship
                if (words1 & words2) or \
                   (concept1.lower() in concept2.lower()) or \
                   (concept2.lower() in concept1.lower()):
                    relationships.append((concept1, concept2))
        
        return relationships

    def _create_graphviz(self, relationships: List[tuple]) -> graphviz.Graph:
        """Create graphviz mind map"""
        dot = graphviz.Graph(
            comment='Mind Map',
            graph_attr={'rankdir': 'LR'}
        )
        
        # Add all nodes and edges
        for source, target in relationships:
            dot.node(source, source)
            dot.node(target, target)
            dot.edge(source, target)
            
        return dot
