"""
NRC VAD Lexicon Integration Module for DimABSA
Provides lexical sentiment features from NRC Valence-Arousal-Dominance Lexicon.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class NRCVADLexicon:
    """
    NRC VAD Lexicon loader and feature extractor.

    The lexicon contains ~55,000 English words with valence, arousal, and dominance scores.
    Original scores are in [-1, 1] range, converted to [1, 9] for DimABSA compatibility.
    """

    def __init__(
        self,
        lexicon_path: Union[str, Path],
        convert_scale: bool = True,
        use_dependency_parsing: bool = True,
        context_window: int = 5
    ):
        """
        Initialize the NRC VAD Lexicon.

        Args:
            lexicon_path: Path to NRC-VAD-Lexicon-v2.1.txt file
            convert_scale: Convert [-1,1] scores to [1,9] scale
            use_dependency_parsing: Use spaCy dependency parsing for context extraction
            context_window: Token window size if not using dependency parsing
        """
        self.lexicon_path = Path(lexicon_path)
        self.convert_scale = convert_scale
        self.use_dependency_parsing = use_dependency_parsing and SPACY_AVAILABLE
        self.context_window = context_window

        # Load lexicon
        self.lexicon: Dict[str, Dict[str, float]] = {}
        self._load_lexicon()

        # Initialize spaCy if available and requested
        self.nlp = None
        if self.use_dependency_parsing:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy model 'en_core_web_sm' not found. "
                      "Using window-based context extraction instead.")
                self.use_dependency_parsing = False

    def _load_lexicon(self):
        """Load and parse the NRC VAD Lexicon file."""
        if not self.lexicon_path.exists():
            raise FileNotFoundError(f"Lexicon file not found: {self.lexicon_path}")

        with open(self.lexicon_path, 'r', encoding='utf-8') as f:
            # Skip header line
            header = f.readline()

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    term = parts[0].lower()
                    valence = float(parts[1])
                    arousal = float(parts[2])
                    dominance = float(parts[3])

                    # Convert scale if requested: [-1, 1] -> [1, 9]
                    if self.convert_scale:
                        valence = self._convert_to_va_scale(valence)
                        arousal = self._convert_to_va_scale(arousal)
                        dominance = self._convert_to_va_scale(dominance)

                    self.lexicon[term] = {
                        'valence': valence,
                        'arousal': arousal,
                        'dominance': dominance
                    }

        print(f"Loaded {len(self.lexicon)} terms from NRC VAD Lexicon")

    def _convert_to_va_scale(self, score: float) -> float:
        """
        Convert NRC lexicon score from [-1, 1] to [1, 9] scale.

        Formula: new_score = 4 * old_score + 5
        - -1 -> 1 (lowest)
        - 0 -> 5 (neutral)
        - 1 -> 9 (highest)

        Args:
            score: Original score in [-1, 1]

        Returns:
            Converted score in [1, 9]
        """
        return 4 * score + 5

    def get_word_va(self, word: str) -> Optional[Tuple[float, float]]:
        """
        Get valence and arousal scores for a single word.

        Args:
            word: Word to look up

        Returns:
            Tuple of (valence, arousal) or None if not found
        """
        word = word.lower().strip()
        if word in self.lexicon:
            entry = self.lexicon[word]
            return entry['valence'], entry['arousal']
        return None

    def get_word_vad(self, word: str) -> Optional[Tuple[float, float, float]]:
        """
        Get valence, arousal, and dominance scores for a single word.

        Args:
            word: Word to look up

        Returns:
            Tuple of (valence, arousal, dominance) or None if not found
        """
        word = word.lower().strip()
        if word in self.lexicon:
            entry = self.lexicon[word]
            return entry['valence'], entry['arousal'], entry['dominance']
        return None

    def _tokenize_simple(self, text: str) -> List[str]:
        """Simple whitespace and punctuation tokenization."""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()

    def _get_related_words_dependency(
        self,
        text: str,
        aspect: str
    ) -> List[Tuple[str, float]]:
        """
        Get syntactically related words using dependency parsing.

        Args:
            text: Full sentence
            aspect: Target aspect term

        Returns:
            List of (word, weight) tuples where weight indicates relationship strength
        """
        if not self.nlp:
            return self._get_related_words_window(text, aspect)

        doc = self.nlp(text.lower())
        aspect_tokens = set(self._tokenize_simple(aspect))

        related_words = []

        # Find aspect tokens in the document
        aspect_token_indices = set()
        for i, token in enumerate(doc):
            if token.text in aspect_tokens:
                aspect_token_indices.add(i)

        # Collect related words based on dependency relations
        for i, token in enumerate(doc):
            if i in aspect_token_indices:
                continue

            weight = 0.0

            # Check if token is syntactically related to aspect
            for asp_idx in aspect_token_indices:
                asp_token = doc[asp_idx]

                # Direct dependency relation
                if token.head == asp_token or asp_token.head == token:
                    weight = max(weight, 1.0)

                # Sibling relation (same head)
                elif token.head == asp_token.head:
                    weight = max(weight, 0.8)

                # Two-hop relation
                elif token.head.head == asp_token or asp_token.head.head == token:
                    weight = max(weight, 0.5)

            # Also consider distance-based weight
            if weight == 0.0:
                min_dist = min(abs(i - asp_idx) for asp_idx in aspect_token_indices)
                if min_dist <= self.context_window:
                    weight = 1.0 / (1.0 + min_dist)

            if weight > 0:
                related_words.append((token.text, weight))

        return related_words

    def _get_related_words_window(
        self,
        text: str,
        aspect: str
    ) -> List[Tuple[str, float]]:
        """
        Get related words using simple window-based approach.

        Args:
            text: Full sentence
            aspect: Target aspect term

        Returns:
            List of (word, weight) tuples with distance-based weights
        """
        tokens = self._tokenize_simple(text)
        aspect_tokens = set(self._tokenize_simple(aspect))

        # Find aspect positions
        aspect_positions = []
        for i, token in enumerate(tokens):
            if token in aspect_tokens:
                aspect_positions.append(i)

        if not aspect_positions:
            # Aspect not found, use all tokens with equal weight
            return [(token, 0.5) for token in tokens if token not in aspect_tokens]

        related_words = []
        for i, token in enumerate(tokens):
            if token in aspect_tokens:
                continue

            # Calculate minimum distance to any aspect token
            min_dist = min(abs(i - pos) for pos in aspect_positions)

            if min_dist <= self.context_window:
                # Weight decreases with distance
                weight = 1.0 / (1.0 + min_dist)
                related_words.append((token, weight))

        return related_words

    def extract_features(
        self,
        text: str,
        aspect: str
    ) -> torch.Tensor:
        """
        Extract 8-dimensional lexicon feature vector for aspect in text.

        Features:
        0. context_v: Weighted average valence of context words
        1. context_a: Weighted average arousal of context words
        2. aspect_v: Average valence of aspect words
        3. aspect_a: Average arousal of aspect words
        4. weighted_v: Distance-weighted valence (closer words have more influence)
        5. weighted_a: Distance-weighted arousal
        6. max_v: Maximum valence in context
        7. max_a: Maximum arousal in context

        Args:
            text: Full sentence
            aspect: Target aspect term

        Returns:
            Float tensor of shape [8] with lexicon features
        """
        # Initialize features with neutral values
        neutral = 5.0  # Neutral value on [1,9] scale
        features = [neutral] * 8

        # Get related words with weights
        if self.use_dependency_parsing and self.nlp:
            related_words = self._get_related_words_dependency(text, aspect)
        else:
            related_words = self._get_related_words_window(text, aspect)

        # Collect context VA scores
        context_v_scores = []
        context_a_scores = []
        context_weights = []

        for word, weight in related_words:
            va = self.get_word_va(word)
            if va is not None:
                context_v_scores.append(va[0])
                context_a_scores.append(va[1])
                context_weights.append(weight)

        # Get aspect VA scores
        aspect_tokens = self._tokenize_simple(aspect)
        aspect_v_scores = []
        aspect_a_scores = []

        for word in aspect_tokens:
            va = self.get_word_va(word)
            if va is not None:
                aspect_v_scores.append(va[0])
                aspect_a_scores.append(va[1])

        # Compute features
        if context_v_scores:
            # Simple average
            features[0] = np.mean(context_v_scores)  # context_v
            features[1] = np.mean(context_a_scores)  # context_a

            # Weighted average
            total_weight = sum(context_weights)
            features[4] = sum(v * w for v, w in zip(context_v_scores, context_weights)) / total_weight
            features[5] = sum(a * w for a, w in zip(context_a_scores, context_weights)) / total_weight

            # Max values
            features[6] = max(context_v_scores)  # max_v
            features[7] = max(context_a_scores)  # max_a

        if aspect_v_scores:
            features[2] = np.mean(aspect_v_scores)  # aspect_v
            features[3] = np.mean(aspect_a_scores)  # aspect_a

        return torch.tensor(features, dtype=torch.float32)

    def extract_batch_features(
        self,
        texts: List[str],
        aspects: List[str]
    ) -> torch.Tensor:
        """
        Extract lexicon features for a batch of text-aspect pairs.

        Args:
            texts: List of sentences
            aspects: List of aspect terms

        Returns:
            Float tensor of shape [batch_size, 8]
        """
        features = []
        for text, aspect in zip(texts, aspects):
            features.append(self.extract_features(text, aspect))
        return torch.stack(features)


class LexiconOnlyPredictor:
    """
    Baseline model that predicts VA scores using only lexicon features.
    Uses distance-weighted averaging with configurable context/aspect blending.
    """

    def __init__(
        self,
        lexicon: NRCVADLexicon,
        context_weight: float = 0.3,
        aspect_weight: float = 0.7
    ):
        """
        Initialize lexicon-only predictor.

        Args:
            lexicon: NRCVADLexicon instance
            context_weight: Weight for context words in final prediction
            aspect_weight: Weight for aspect words in final prediction
        """
        self.lexicon = lexicon
        self.context_weight = context_weight
        self.aspect_weight = aspect_weight

        # Ensure weights sum to 1
        total = context_weight + aspect_weight
        self.context_weight /= total
        self.aspect_weight /= total

    def predict(self, text: str, aspect: str) -> Tuple[float, float]:
        """
        Predict VA scores for an aspect in text.

        Args:
            text: Full sentence
            aspect: Target aspect term

        Returns:
            Tuple of (valence, arousal) predictions
        """
        features = self.lexicon.extract_features(text, aspect)

        # Blend context and aspect scores
        # features: [context_v, context_a, aspect_v, aspect_a, weighted_v, weighted_a, max_v, max_a]
        context_v = features[4].item()  # Use weighted scores
        context_a = features[5].item()
        aspect_v = features[2].item()
        aspect_a = features[3].item()

        # Weighted combination
        pred_v = self.context_weight * context_v + self.aspect_weight * aspect_v
        pred_a = self.context_weight * context_a + self.aspect_weight * aspect_a

        # Clamp to valid range
        pred_v = max(1.0, min(9.0, pred_v))
        pred_a = max(1.0, min(9.0, pred_a))

        return pred_v, pred_a

    def predict_batch(
        self,
        texts: List[str],
        aspects: List[str]
    ) -> Tuple[List[float], List[float]]:
        """
        Predict VA scores for a batch.

        Args:
            texts: List of sentences
            aspects: List of aspect terms

        Returns:
            Tuple of (valence_list, arousal_list)
        """
        v_preds = []
        a_preds = []

        for text, aspect in zip(texts, aspects):
            v, a = self.predict(text, aspect)
            v_preds.append(v)
            a_preds.append(a)

        return v_preds, a_preds


def create_lexicon(
    lexicon_dir: Union[str, Path] = "NRC-VAD-Lexicon-v2.1",
    use_dependency_parsing: bool = False
) -> NRCVADLexicon:
    """
    Factory function to create NRCVADLexicon with common settings.

    Args:
        lexicon_dir: Path to NRC-VAD-Lexicon directory
        use_dependency_parsing: Whether to use spaCy for context extraction

    Returns:
        Configured NRCVADLexicon instance
    """
    lexicon_dir = Path(lexicon_dir)
    lexicon_file = lexicon_dir / "NRC-VAD-Lexicon-v2.1.txt"

    if not lexicon_file.exists():
        # Try alternate location
        lexicon_file = lexicon_dir / "Unigrams" / "unigrams-NRC-VAD-Lexicon-v2.1.txt"

    return NRCVADLexicon(
        lexicon_path=lexicon_file,
        convert_scale=True,
        use_dependency_parsing=use_dependency_parsing
    )


if __name__ == "__main__":
    # Test lexicon loading and feature extraction
    lexicon_path = Path("NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt")

    if lexicon_path.exists():
        print("Testing NRC VAD Lexicon integration...")
        lexicon = NRCVADLexicon(lexicon_path, use_dependency_parsing=False)

        # Test individual word lookup
        test_words = ["good", "bad", "excellent", "terrible", "neutral", "laptop", "food"]
        print("\nWord VA scores:")
        for word in test_words:
            va = lexicon.get_word_va(word)
            if va:
                print(f"  {word}: V={va[0]:.2f}, A={va[1]:.2f}")
            else:
                print(f"  {word}: not found")

        # Test feature extraction
        test_cases = [
            ("The food was absolutely amazing and delicious!", "food"),
            ("The laptop battery drains too quickly.", "battery"),
            ("Average service, nothing special.", "service"),
        ]

        print("\nFeature extraction:")
        for text, aspect in test_cases:
            features = lexicon.extract_features(text, aspect)
            print(f"\n  Text: {text}")
            print(f"  Aspect: {aspect}")
            print(f"  Features: {features.numpy().round(2)}")

        # Test lexicon-only predictor
        print("\nLexicon-only predictions:")
        predictor = LexiconOnlyPredictor(lexicon)
        for text, aspect in test_cases:
            v, a = predictor.predict(text, aspect)
            print(f"  '{aspect}': V={v:.2f}, A={a:.2f}")
    else:
        print(f"Lexicon file not found: {lexicon_path}")
