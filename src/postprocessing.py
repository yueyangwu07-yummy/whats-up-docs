"""
Post-processing functions for summary refinement.

This module provides functions to improve summary quality by:
- Removing duplicate sentences
- Adjusting summary length
- Ensuring coherence between sentences
"""

import logging
import re
from typing import List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def _get_sentence_model() -> Optional[SentenceTransformer]:
    """
    Lazy load sentence transformer model.
    
    Returns:
        SentenceTransformer model or None if not available
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        LOGGER.warning(
            "sentence-transformers not available. "
            "Install it with: pip install sentence-transformers"
        )
        return None
    
    try:
        # Use a lightweight model for similarity computation
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as exc:
        LOGGER.exception("Failed to load sentence transformer model: %s", exc)
        return None


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Simple sentence splitting using regex
    # Split on period, exclamation, question mark followed by space or end
    sentences = re.split(r'[.!?]+\s+', text.strip())
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def count_words(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text: Input text
        
    Returns:
        Word count
    """
    return len(text.split())


def remove_duplicate_sentences(
    summary: str,
    similarity_threshold: float = 0.85,
    model: Optional[SentenceTransformer] = None,
) -> str:
    """
    Remove semantically duplicate sentences from summary.
    
    Uses sentence-transformers to compute semantic similarity.
    If sentence-transformers is not available, falls back to exact string matching.
    
    Args:
        summary: Input summary text
        similarity_threshold: Similarity threshold for considering sentences as duplicates (0-1)
        model: Optional pre-loaded SentenceTransformer model
        
    Returns:
        Summary with duplicate sentences removed
    """
    if not summary.strip():
        return summary
    
    sentences = split_into_sentences(summary)
    if len(sentences) <= 1:
        return summary
    
    # Try to use sentence transformers
    if model is None:
        model = _get_sentence_model()
    
    if model is not None:
        try:
            # Compute embeddings for all sentences
            embeddings = model.encode(sentences, convert_to_numpy=True)
            
            # Find duplicate sentences
            keep_indices = [0]  # Always keep first sentence
            for i in range(1, len(sentences)):
                is_duplicate = False
                # Check similarity with all previously kept sentences
                for j in keep_indices:
                    # Compute cosine similarity
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    if similarity >= similarity_threshold:
                        is_duplicate = True
                        LOGGER.debug(
                            "Removing duplicate sentence %d (similarity: %.3f with sentence %d)",
                            i, similarity, j
                        )
                        break
                
                if not is_duplicate:
                    keep_indices.append(i)
            
            filtered_sentences = [sentences[i] for i in keep_indices]
            result = ". ".join(filtered_sentences)
            if result and not result.endswith(('.', '!', '?')):
                result += "."
            return result
            
        except Exception as exc:
            LOGGER.warning("Error in semantic duplicate detection: %s. Using fallback.", exc)
    
    # Fallback: exact string matching
    LOGGER.debug("Using exact string matching for duplicate detection")
    seen = set()
    filtered_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        if sentence_lower not in seen:
            seen.add(sentence_lower)
            filtered_sentences.append(sentence)
    
    result = ". ".join(filtered_sentences)
    if result and not result.endswith(('.', '!', '?')):
        result += "."
    return result


def adjust_length(
    summary: str,
    min_words: int = 150,
    max_words: int = 220,
) -> str:
    """
    Adjust summary length to be within specified word count range.
    
    If summary is too short, attempts to expand by keeping more sentences.
    If summary is too long, truncates to fit within max_words.
    
    Args:
        summary: Input summary text
        min_words: Minimum word count (default: 150)
        max_words: Maximum word count (default: 220)
        
    Returns:
        Adjusted summary
    """
    if not summary.strip():
        return summary
    
    word_count = count_words(summary)
    
    # If within range, return as is
    if min_words <= word_count <= max_words:
        LOGGER.debug("Summary length (%d words) is within range", word_count)
        return summary
    
    sentences = split_into_sentences(summary)
    
    # If too long, truncate
    if word_count > max_words:
        LOGGER.debug("Summary too long (%d words), truncating to %d", word_count, max_words)
        truncated = []
        current_words = 0
        for sentence in sentences:
            sentence_words = count_words(sentence)
            if current_words + sentence_words > max_words:
                break
            truncated.append(sentence)
            current_words += sentence_words
        
        result = ". ".join(truncated)
        if result and not result.endswith(('.', '!', '?')):
            result += "."
        return result
    
    # If too short, return as is (can't expand without original text)
    # In practice, this would require the original document
    LOGGER.debug("Summary too short (%d words), but cannot expand without original text", word_count)
    return summary


def ensure_coherence(summary: str) -> str:
    """
    Ensure coherence between first and last sentences.
    
    Checks if the summary flows well from start to end.
    If the last sentence seems disconnected, attempts to improve it.
    
    Args:
        summary: Input summary text
        
    Returns:
        Summary with improved coherence
    """
    if not summary.strip():
        return summary
    
    sentences = split_into_sentences(summary)
    if len(sentences) <= 1:
        return summary
    
    # Check if last sentence starts with common transition words
    last_sentence = sentences[-1].strip()
    first_sentence = sentences[0].strip()
    
    # Common transition words that indicate good flow
    good_transitions = [
        "therefore", "thus", "consequently", "in conclusion", "in summary",
        "overall", "in sum", "finally", "in brief", "to summarize"
    ]
    
    # Check if last sentence seems disconnected
    last_lower = last_sentence.lower()
    has_good_transition = any(
        last_lower.startswith(transition) for transition in good_transitions
    )
    
    # If last sentence doesn't have a good transition and seems disconnected,
    # try to improve it
    if not has_good_transition and len(sentences) >= 3:
        # Check if first and last sentences share common words (simple coherence check)
        first_words = set(first_sentence.lower().split())
        last_words = set(last_sentence.lower().split())
        common_words = first_words.intersection(last_words)
        
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "this", "that", "these", "those", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had", "do",
            "does", "did", "will", "would", "could", "should", "may", "might"
        }
        common_words = common_words - stop_words
        
        # If no meaningful common words, the summary might lack coherence
        if len(common_words) == 0:
            LOGGER.debug("Summary may lack coherence, but keeping as is")
            # Could add a concluding phrase, but that might change meaning
            # For now, just return as is
    
    return summary


def postprocess_summary(
    summary: str,
    remove_duplicates: bool = True,
    adjust_len: bool = True,
    ensure_coh: bool = True,
    min_words: int = 150,
    max_words: int = 220,
    similarity_threshold: float = 0.85,
    model: Optional[SentenceTransformer] = None,
) -> str:
    """
    Main post-processing function that applies all refinement steps.
    
    Args:
        summary: Input summary text
        remove_duplicates: Whether to remove duplicate sentences
        adjust_len: Whether to adjust summary length
        ensure_coh: Whether to check coherence
        min_words: Minimum word count for length adjustment
        max_words: Maximum word count for length adjustment
        similarity_threshold: Similarity threshold for duplicate detection
        model: Optional pre-loaded SentenceTransformer model
        
    Returns:
        Post-processed summary
    """
    if not summary.strip():
        return summary
    
    result = summary
    
    # Step 1: Remove duplicates
    if remove_duplicates:
        LOGGER.debug("Removing duplicate sentences")
        result = remove_duplicate_sentences(
            result,
            similarity_threshold=similarity_threshold,
            model=model,
        )
    
    # Step 2: Adjust length
    if adjust_len:
        LOGGER.debug("Adjusting summary length")
        result = adjust_length(result, min_words=min_words, max_words=max_words)
    
    # Step 3: Ensure coherence
    if ensure_coh:
        LOGGER.debug("Ensuring coherence")
        result = ensure_coherence(result)
    
    return result.strip()

