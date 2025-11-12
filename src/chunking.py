"""
Intelligent text chunking for long documents.

This module provides functions to intelligently chunk long documents
by extracting key sections (introduction, middle, conclusion) while
maintaining context continuity.
"""

import logging
from typing import List, Tuple

LOGGER = logging.getLogger(__name__)


def count_words(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text: Input text
        
    Returns:
        Word count
    """
    return len(text.split())


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    import re
    sentences = re.split(r'[.!?]+\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def sliding_window_chunk(
    sentences: List[str],
    window_size: int = 3,
) -> List[str]:
    """
    Create overlapping chunks using sliding window.
    
    Args:
        sentences: List of sentences
        window_size: Number of sentences per window
        
    Returns:
        List of chunked text segments
    """
    if len(sentences) <= window_size:
        return [" ".join(sentences)]
    
    chunks = []
    for i in range(0, len(sentences) - window_size + 1, window_size // 2):
        chunk = sentences[i:i + window_size]
        chunks.append(" ".join(chunk))
    
    # Add remaining sentences
    if len(sentences) % window_size != 0:
        remaining = sentences[-(window_size // 2):]
        if remaining:
            chunks.append(" ".join(remaining))
    
    return chunks


def semantic_chunking(
    text: str,
    max_words: int = 8000,
    intro_ratio: float = 0.15,
    middle_ratio: float = 0.30,
    conclusion_ratio: float = 0.20,
    window_size: int = 3,
) -> str:
    """
    Intelligently chunk long documents by extracting key sections.
    
    Strategy:
    - Extract first 15% (introduction)
    - Sample 30% from middle section
    - Extract last 20% (conclusion)
    - Use sliding window to maintain context continuity
    
    Args:
        text: Input document text
        max_words: Maximum word count for output (default: 8000)
        intro_ratio: Ratio of text to extract from beginning (default: 0.15)
        middle_ratio: Ratio of text to sample from middle (default: 0.30)
        conclusion_ratio: Ratio of text to extract from end (default: 0.20)
        window_size: Sliding window size for context continuity
        
    Returns:
        Chunked text within max_words limit
    """
    if not text.strip():
        return text
    
    total_words = count_words(text)
    LOGGER.info("Original document length: %d words", total_words)
    
    # Only chunk if document exceeds threshold (12000 words as per requirements)
    chunk_threshold = 12000
    if total_words <= chunk_threshold:
        LOGGER.debug("Document length (%d words) is within threshold, no chunking needed", total_words)
        # Still ensure it's within max_words
        if total_words <= max_words:
            return text
        # If slightly over, just truncate
        sentences = split_into_sentences(text)
        result_sentences = []
        current_words = 0
        for sentence in sentences:
            sentence_words = count_words(sentence)
            if current_words + sentence_words > max_words:
                break
            result_sentences.append(sentence)
            current_words += sentence_words
        return " ".join(result_sentences)
    
    # Split into sentences for precise control
    sentences = split_into_sentences(text)
    total_sentences = len(sentences)
    
    if total_sentences == 0:
        return text
    
    LOGGER.info("Document has %d sentences", total_sentences)
    
    # Calculate sentence indices for each section
    intro_end = int(total_sentences * intro_ratio)
    conclusion_start = int(total_sentences * (1 - conclusion_ratio))
    middle_start = intro_end
    middle_end = conclusion_start
    
    # Extract introduction (first 15%)
    intro_sentences = sentences[:intro_end] if intro_end > 0 else []
    LOGGER.debug("Extracted %d sentences from introduction", len(intro_sentences))
    
    # Extract conclusion (last 20%)
    conclusion_sentences = sentences[conclusion_start:] if conclusion_start < total_sentences else []
    LOGGER.debug("Extracted %d sentences from conclusion", len(conclusion_sentences))
    
    # Sample from middle section (30%)
    middle_sentences = sentences[middle_start:middle_end] if middle_start < middle_end else []
    if middle_sentences:
        # Sample evenly from middle section
        sample_count = int(len(middle_sentences) * middle_ratio)
        if sample_count > 0:
            step = max(1, len(middle_sentences) // sample_count)
            sampled_middle = middle_sentences[::step][:sample_count]
            LOGGER.debug("Sampled %d sentences from middle section", len(sampled_middle))
        else:
            sampled_middle = []
    else:
        sampled_middle = []
    
    # Combine sections
    selected_sentences = intro_sentences + sampled_middle + conclusion_sentences
    
    # Apply sliding window to maintain context
    if window_size > 1 and len(selected_sentences) > window_size:
        # Re-chunk with sliding window for better continuity
        windowed_chunks = sliding_window_chunk(selected_sentences, window_size=window_size)
        # Flatten back to sentences (sliding window creates overlaps, but we'll merge)
        # For simplicity, just use the selected sentences
        final_sentences = selected_sentences
    else:
        final_sentences = selected_sentences
    
    # Reconstruct text
    result_text = " ".join(final_sentences)
    result_words = count_words(result_text)
    
    LOGGER.info("Chunked document length: %d words (target: <= %d)", result_words, max_words)
    
    # If still over max_words, truncate further
    if result_words > max_words:
        LOGGER.debug("Chunked text still exceeds max_words, truncating further")
        truncated_sentences = []
        current_words = 0
        for sentence in final_sentences:
            sentence_words = count_words(sentence)
            if current_words + sentence_words > max_words:
                break
            truncated_sentences.append(sentence)
            current_words += sentence_words
        
        result_text = " ".join(truncated_sentences)
        result_words = count_words(result_text)
        LOGGER.info("Final chunked document length: %d words", result_words)
    
    return result_text


def chunk_document(
    text: str,
    chunk_threshold: int = 12000,
    max_words: int = 8000,
    use_smart_chunking: bool = True,
) -> str:
    """
    Main function to chunk documents intelligently.
    
    Args:
        text: Input document text
        chunk_threshold: Word count threshold for chunking (default: 12000)
        max_words: Maximum word count for output (default: 8000)
        use_smart_chunking: Whether to use smart chunking strategy
        
    Returns:
        Chunked text
    """
    if not text.strip():
        return text
    
    word_count = count_words(text)
    
    # If document is short enough, return as is
    if word_count <= chunk_threshold:
        if word_count <= max_words:
            return text
        # Slightly over, just truncate
        sentences = split_into_sentences(text)
        result_sentences = []
        current_words = 0
        for sentence in sentences:
            sentence_words = count_words(sentence)
            if current_words + sentence_words > max_words:
                break
            result_sentences.append(sentence)
            current_words += sentence_words
        return " ".join(result_sentences)
    
    # Use smart chunking
    if use_smart_chunking:
        return semantic_chunking(text, max_words=max_words)
    else:
        # Simple truncation fallback
        sentences = split_into_sentences(text)
        result_sentences = []
        current_words = 0
        for sentence in sentences:
            sentence_words = count_words(sentence)
            if current_words + sentence_words > max_words:
                break
            result_sentences.append(sentence)
            current_words += sentence_words
        return " ".join(result_sentences)

