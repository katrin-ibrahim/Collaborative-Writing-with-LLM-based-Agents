"""
Enhanced Text Preprocessor for Wikipedia Content
Handles robust text cleaning, tokenization, and normalization for BM25 search.
"""

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Install with: pip install nltk")


class EnhancedTextPreprocessor:
    """
    Enhanced text preprocessor for Wikipedia content with robust cleaning and tokenization.
    """

    def __init__(self, language: str = "english", use_stemming: bool = True):
        """
        Initialize the text preprocessor.

        Args:
            language: Language for stop words (default: english)
            use_stemming: Whether to apply stemming
        """
        self.language = language
        self.use_stemming = use_stemming
        self.stemmer = PorterStemmer() if use_stemming and NLTK_AVAILABLE else None

        # Initialize NLTK components if available
        if NLTK_AVAILABLE:
            self._ensure_nltk_data()
            try:
                self.stop_words = set(stopwords.words(language))
            except:
                logger.warning(
                    f"Stop words for {language} not available, using empty set"
                )
                self.stop_words = set()
        else:
            # Fallback stop words for English
            self.stop_words = {
                "a",
                "an",
                "and",
                "are",
                "as",
                "at",
                "be",
                "by",
                "for",
                "from",
                "has",
                "he",
                "in",
                "is",
                "it",
                "its",
                "of",
                "on",
                "that",
                "the",
                "to",
                "was",
                "will",
                "with",
                "the",
                "this",
                "but",
                "they",
                "have",
                "had",
                "what",
                "said",
                "each",
                "which",
                "she",
                "do",
                "how",
                "their",
                "if",
                "up",
                "out",
                "many",
                "then",
                "them",
                "these",
                "so",
                "some",
                "her",
                "would",
                "make",
                "like",
                "into",
                "him",
                "time",
                "two",
                "more",
                "go",
                "no",
                "way",
                "could",
                "my",
                "than",
                "first",
                "been",
                "call",
                "who",
                "oil",
                "sit",
                "now",
                "find",
                "down",
                "day",
                "did",
                "get",
                "come",
                "made",
                "may",
                "part",
            }

        # Wikipedia-specific patterns to remove
        self.wiki_patterns = [
            # Citations and references
            r"\[\d+\]",  # [1], [2], etc.
            r"\[citation needed\]",
            r"\[clarification needed\]",
            r"\[when\?\]",
            r"\[who\?\]",
            r"\[according to whom\?\]",
            # Navigation elements
            r"Category:.*",
            r"File:.*",
            r"Image:.*",
            r"Media:.*",
            # Templates and infoboxes
            r"\{\{[^}]*\}\}",
            # External links
            r"https?://[^\s]+",
            r"www\.[^\s]+",
            # HTML remnants
            r"<[^>]+>",
            r"&[a-zA-Z]+;",  # HTML entities
            # Special Wikipedia markup
            r"\[\[([^|\]]+\|)?([^\]]+)\]\]",  # Wiki links - keep the display text
            r"'''([^']+)'''",  # Bold - keep content
            r"''([^']+)''",  # Italic - keep content
        ]

        # Compile patterns for efficiency
        self.compiled_patterns = [
            (re.compile(pattern), replacement)
            for pattern, replacement in [
                (r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2"),  # Wiki links -> display text
                (r"'''([^']+)'''", r"\1"),  # Bold -> content
                (r"''([^']+)''", r"\1"),  # Italic -> content
            ]
        ]

        self.removal_patterns = [
            re.compile(pattern) for pattern in self.wiki_patterns[:-3]
        ]

    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded."""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download("punkt", quiet=True)

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download("stopwords", quiet=True)

    def clean_wikipedia_content(self, text: str) -> str:
        """
        Clean Wikipedia-specific markup and formatting.

        Args:
            text: Raw Wikipedia article text

        Returns:
            Cleaned text with markup removed
        """
        if not text:
            return ""

        # Apply replacement patterns first (preserve content)
        for pattern, replacement in self.compiled_patterns:
            text = pattern.sub(replacement, text)

        # Remove unwanted patterns
        for pattern in self.removal_patterns:
            text = pattern.sub("", text)

        # Clean up multiple whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n+", " ", text)

        return text.strip()

    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words with proper handling of special cases.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        if not text:
            return []

        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text.lower())
            except:
                # Fallback to simple split if NLTK fails
                tokens = text.lower().split()
        else:
            # Simple tokenization fallback
            # Remove punctuation but preserve numbers and special chars in entity names
            text = re.sub(r"[^\w\s]", " ", text.lower())
            tokens = text.split()

        return tokens

    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """
        Remove stop words while preserving important terms.

        Args:
            tokens: List of tokens

        Returns:
            Filtered tokens
        """
        filtered_tokens = []

        for token in tokens:
            # Keep tokens that are:
            # - Not stop words
            # - Numbers (important for dates, years, etc.)
            # - Short acronyms (AFL, NBA, etc.)
            # - Mixed alphanumeric (2022_afl, etc.)
            if (
                token not in self.stop_words
                or token.isdigit()
                or (len(token) <= 4 and token.isupper())
                or any(char.isdigit() for char in token)
            ):
                filtered_tokens.append(token)

        return filtered_tokens

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming to tokens.

        Args:
            tokens: List of tokens

        Returns:
            Stemmed tokens
        """
        if not self.use_stemming or not self.stemmer:
            return tokens

        stemmed_tokens = []
        for token in tokens:
            # Don't stem numbers, short acronyms, or mixed alphanumeric
            if (
                token.isdigit()
                or (len(token) <= 4 and token.isupper())
                or any(char.isdigit() for char in token)
            ):
                stemmed_tokens.append(token)
            else:
                try:
                    stemmed_tokens.append(self.stemmer.stem(token))
                except:
                    stemmed_tokens.append(token)

        return stemmed_tokens

    def preprocess_text(self, text: str) -> List[str]:
        """
        Complete preprocessing pipeline for text.

        Args:
            text: Raw text to preprocess

        Returns:
            List of processed tokens
        """
        # Step 1: Clean Wikipedia markup
        cleaned_text = self.clean_wikipedia_content(text)

        # Step 2: Tokenize
        tokens = self.tokenize_text(cleaned_text)

        # Step 3: Filter tokens (remove very short/long, non-alphabetic)
        tokens = [
            token
            for token in tokens
            if len(token) >= 2
            and len(token) <= 50
            and (token.isalnum() or any(char.isalpha() for char in token))
        ]

        # Step 4: Remove stop words
        tokens = self.remove_stop_words(tokens)

        # Step 5: Apply stemming
        tokens = self.stem_tokens(tokens)

        return tokens

    def preprocess_query(self, query: str) -> List[str]:
        """
        Preprocess search query using same pipeline as documents.

        Args:
            query: Search query

        Returns:
            Processed query tokens
        """
        # Handle common query patterns
        query = query.replace("_", " ")  # Handle underscores in entity names
        query = re.sub(r"([a-z])([A-Z])", r"\1 \2", query)  # Split camelCase

        return self.preprocess_text(query)

    def get_stats(self) -> dict:
        """Get preprocessor statistics."""
        return {
            "language": self.language,
            "use_stemming": self.use_stemming,
            "nltk_available": NLTK_AVAILABLE,
            "stop_words_count": len(self.stop_words),
            "wiki_patterns_count": len(self.wiki_patterns),
        }
