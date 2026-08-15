#full_sentence_extractor.py
import re

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from doxl_ai_terminal.data_structure.docs import DocData


@dataclass
class ExtractedSentence:
    paragraph: int
    sentence_index: int
    sentence: str
    source_line: int
    match_score: Optional[float] = None


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences (full stop, ?, !)"""
    # Handles Mr. Mrs. Dr. etc. without breaking
    pattern = r'(?<!\b(?:Mr|Mrs|Dr|Sr|Jr|vs|etc|Prof|Inc|Ltd))\.\s+|\?\s+|!\s+'

    parts = re.split(pattern, text)
    return [s.strip() for s in parts if s.strip()]


def get_full_text_by_paragraph(doc_data: DocData) -> Dict[int, str]:
    """Combine all lines of each paragraph into one full text"""
    para_map: Dict[int, List[str]] = {}

    for line in doc_data.lines:
        if line.paragraph not in para_map:
            para_map[line.paragraph] = []
        para_map[line.paragraph].append(line.line_str)

    # Join lines with space to form full paragraph text
    return {p: " ".join(lines) for p, lines in para_map.items()}


def _find_source_line(doc_data: DocData, paragraph: int, sentence: str) -> int:
    """Find which line the sentence belongs to"""
    for line in doc_data.lines:
        if line.paragraph == paragraph and sentence[:20].lower() in line.line_str.lower():
            return line.line
    return -1


def _is_match(chunk: str, sentence: str) -> bool:
    """Check if chunk content overlaps with the sentence"""
    chunk_lower = chunk.lower()
    sentence_lower = sentence.lower()

    # Direct containment
    if chunk_lower in sentence_lower:
        return True

    # Partial overlap — at least 60% of chunk words found in sentence
    chunk_words = set(chunk_lower.split())
    sentence_words = set(sentence_lower.split())
    overlap = chunk_words & sentence_words

    if len(chunk_words) > 0 and len(overlap) / len(chunk_words) >= 0.6:
        return True

    return False


def extract_full_sentences(doc_data: DocData, query: str) -> List[ExtractedSentence]:
    """
    Find all sentences in DocData that contain the query.
    Returns complete sentences (start to full stop).
    """
    para_texts = get_full_text_by_paragraph(doc_data)
    results = []

    for p_num, full_text in para_texts.items():
        sentences = split_into_sentences(full_text)

        for s_index, sentence in enumerate(sentences, start=1):
            if query.lower() in sentence.lower():

                # Ensure it ends with punctuation
                if not sentence.endswith(('.', '?', '!')):
                    sentence = sentence + '.'

                results.append(ExtractedSentence(
                    paragraph=p_num,
                    sentence_index=s_index,
                    sentence=sentence,
                    source_line=_find_source_line(doc_data, p_num, sentence),
                ))

    return results


def extract_from_retrieved_chunk(doc_data: DocData, chunk_text: str) -> List[ExtractedSentence]:
    """
    Given a retrieved chunk (possibly partial),
    find and return the complete sentence(s) from DocData.
    """
    para_texts = get_full_text_by_paragraph(doc_data)
    results = []

    # Take key phrases from the chunk to match
    chunk_words = chunk_text.lower().split()
    # Use first 5 words as search key (enough to locate)
    search_key = " ".join(chunk_words[:5]) if len(chunk_words) >= 5 else chunk_text.lower()

    for p_num, full_text in para_texts.items():
        if search_key not in full_text.lower():
            continue

        sentences = split_into_sentences(full_text)

        for s_index, sentence in enumerate(sentences, start=1):
            # Check if any part of the chunk matches this sentence
            if _is_match(chunk_text, sentence):

                if not sentence.endswith(('.', '?', '!')):
                    sentence = sentence + '.'

                results.append(ExtractedSentence(
                    paragraph=p_num,
                    sentence_index=s_index,
                    sentence=sentence,
                    source_line=_find_source_line(doc_data, p_num, sentence),
                ))

    return results


def extract_with_context(doc_data: DocData, chunk_text: str, context_window: int = 1) -> List[Dict]:
    """
    Extract the full sentence + surrounding sentences.

    context_window: how many sentences before/after to include
    """
    para_texts = get_full_text_by_paragraph(doc_data)
    results = []

    for p_num, full_text in para_texts.items():
        sentences = split_into_sentences(full_text)

        for s_index, sentence in enumerate(sentences):
            if not _is_match(chunk_text, sentence):
                continue

            # Get surrounding sentences
            start = max(0, s_index - context_window)
            end = min(len(sentences), s_index + context_window + 1)

            before = sentences[start:s_index]
            after = sentences[s_index + 1:end]

            if not sentence.endswith(('.', '?', '!')):
                sentence = sentence + '.'

            results.append({
                "paragraph": p_num,
                "before": [s + '.' if not s.endswith(('.', '?', '!')) else s for s in before],
                "matched_sentence": sentence,
                "after": [s + '.' if not s.endswith(('.', '?', '!')) else s for s in after],
                "full_context": " ".join(
                    [s + '.' if not s.endswith(('.', '?', '!')) else s
                     for s in sentences[start:end]]
                ),
            })

    return results