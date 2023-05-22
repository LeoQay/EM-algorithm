from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np

import xml.etree.ElementTree as XmlElementTree


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    with open(filename, 'r') as file:
        text = file.read().replace('&', '&amp;')

    sentence_pairs = []
    alignments = []

    root = XmlElementTree.fromstring(text)

    tags = {'english', 'czech', 'sure', 'possible'}
    for sentence in root:
        parsed = {tag: sentence.find(tag).text for tag in tags}
        for key in ['english', 'czech']:
            val = parsed[key]
            if val is None:
                parsed[key] = []
            else:
                parsed[key] = val.split()
        sentence_pairs.append(SentencePair(parsed['english'], parsed['czech']))
        for key in ['sure', 'possible']:
            val = parsed[key]
            if val is None:
                parsed[key] = []
            else:
                parsed[key] = [tuple(map(int, pair.split('-'))) for pair in val.split()]
        alignments.append(LabeledAlignment(parsed['sure'], parsed['possible']))

    return sentence_pairs, alignments


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff -- natural number -- most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language
        
    Tip: 
        Use cutting by freq_cutoff independently in src and target.
        Moreover in both cases of freq_cutoff (None or not None) - you may get a different size of the dictionary

    """
    def make_dict(iterable):
        count = Counter(iterable)
        if freq_cutoff is not None:
            words = {
                pair[0]: idx
                for idx, pair in enumerate(count.most_common(freq_cutoff))
            }
        else:
            words = {key: idx for idx, key in enumerate(count)}
        return words

    def iterate_source():
        for sentence_pair in sentence_pairs:
            for source in sentence_pair.source:
                yield source

    def iterate_target():
        for sentence_pair in sentence_pairs:
            for target in sentence_pair.target:
                yield target

    return make_dict(iterate_source()), make_dict(iterate_target())


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    result = []
    for pair in sentence_pairs:
        source_arr = np.array([source_dict[source] for source in pair.source if source in source_dict], dtype=np.int32)
        target_arr = np.array([target_dict[target] for target in pair.target if target in target_dict], dtype=np.int32)
        if len(source_arr) > 0 and len(target_arr) > 0:
            result.append(TokenizedSentencePair(source_arr, target_arr))

    return result
