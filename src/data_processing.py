from typing import List, Dict, Tuple
from collections import Counter
import torch

try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
    with open(infile, "r") as file:
        lines: List[str] = file.read().splitlines()

    examples: List[SentimentExample] = []

    for full_line in lines:
        line: List[str] = full_line.split("\t")
        info: str = " ".join(line[:-1])
        label: str = line[-1]
        tokenized: List[str] = tokenize(info)
        examples.append(SentimentExample(tokenized, int(label)))
    return examples


def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    vocab: Dict[str, int] = {}

    for example in examples:
        for word in example._words:
            if not word in vocab.keys():
                vocab[word] = len(vocab.keys())

    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """
    bow: torch.Tensor = torch.zeros(len(vocab))

    contador: Counter = Counter(text)
    for word in contador.keys():
        if word in vocab.keys():
            bow[vocab[word]] = contador[word]

    if binary:
        bow[bow>0] = 1

    return bow
