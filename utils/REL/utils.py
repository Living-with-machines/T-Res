import numpy as np
from typing import List, Any, Optional, Tuple


def flatten_list_of_lists(
    list_of_lists: List[List[Any]],
) -> Tuple[List[Any], List[int]]:
    """
    Flatten a list of lists for input to torch.nn.EmbeddingBag.

    Args:
        list_of_lists (List[List[Any]]): A list of lists to be flattened.

    Returns:
        tuple: A tuple containing the flattened list and the offsets.

    Example:
        >>> list_of_lists = [[1, 2, 3], [4, 5], [6]]
        >>> print(flatten_list_of_lists(list_of_lists))
        ([1, 2, 3, 4, 5, 6], [3, 5])
    """
    list_of_lists = [[]] + list_of_lists
    offsets = np.cumsum([len(x) for x in list_of_lists])[:-1]
    flatten = sum(list_of_lists[1:], [])
    return flatten, offsets


def make_equal_len(
    lists: List[List[Any]], fill_in: Optional[int] = 0, to_right: Optional[bool] = True
) -> Tuple[List[Any], List[float]]:
    """
    Make lists of equal length by padding or truncating.

    Args:
        lists (list): A list of lists to be made of equal length.
        fill_in (int, optional): The value used for padding. Defaults to ``0``.
        to_right (bool, optional): Whether to pad or truncate to the right.
            Defaults to ``True``.

    Returns:
        tuple: A tuple containing the lists of equal length and the mask.

    Example:
        >>> lists = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        >>> print(make_equal_len(lists))
        ([[1, 2, 3], [4, 5, 0], [6, 7, 8]], [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    Note:
        The mask indicates the original length of each list with ``1.0`` values
        and the padded/truncated parts with ``0.0`` values.
    """
    lens = [len(l) for l in lists]
    max_len = max(1, max(lens))
    if to_right:
        eq_lists = [l + [fill_in] * (max_len - len(l)) for l in lists]
        mask = [[1.0] * l + [0.0] * (max_len - l) for l in lens]
    else:
        eq_lists = [[fill_in] * (max_len - len(l)) + l for l in lists]
        mask = [[0.0] * (max_len - l) + [1.0] * l for l in lens]
    return eq_lists, mask


def is_important_word(s: str) -> bool:
    """
    Check if a word is important. An important word is not a stopword, a
    number, or has a length of 1.

    Args:
        s (str): The word to be checked.

    Returns:
        bool: True if the word is important, False otherwise.

    Example:
        >>> print(is_important_word("apple"))
        True
    """
    try:
        if len(s) <= 3 or s.lower() in STOPWORDS:
            return False
        float(s)
        return False
    except:
        return True


STOPWORDS = {
    "a",
    "about",
    "above",
    "across",
    "after",
    "afterwards",
    "again",
    "against",
    "all",
    "almost",
    "alone",
    "along",
    "already",
    "also",
    "although",
    "always",
    "am",
    "among",
    "amongst",
    "amoungst",
    "amount",
    "an",
    "and",
    "another",
    "any",
    "anyhow",
    "anyone",
    "anything",
    "anyway",
    "anywhere",
    "are",
    "around",
    "as",
    "at",
    "back",
    "be",
    "became",
    "because",
    "become",
    "becomes",
    "becoming",
    "been",
    "before",
    "beforehand",
    "behind",
    "being",
    "below",
    "beside",
    "besides",
    "between",
    "beyond",
    "both",
    "bottom",
    "but",
    "by",
    "call",
    "can",
    "cannot",
    "cant",
    "dont",
    "co",
    "con",
    "could",
    "couldnt",
    "cry",
    "de",
    "describe",
    "detail",
    "do",
    "done",
    "down",
    "due",
    "during",
    "each",
    "eg",
    "eight",
    "either",
    "eleven",
    "else",
    "elsewhere",
    "empty",
    "enough",
    "etc",
    "even",
    "ever",
    "every",
    "everyone",
    "everything",
    "everywhere",
    "except",
    "few",
    "fifteen",
    "fify",
    "fill",
    "find",
    "fire",
    "first",
    "five",
    "for",
    "former",
    "formerly",
    "forty",
    "found",
    "four",
    "from",
    "front",
    "full",
    "further",
    "get",
    "give",
    "go",
    "had",
    "has",
    "hasnt",
    "have",
    "he",
    "hence",
    "her",
    "here",
    "hereafter",
    "hereby",
    "herein",
    "hereupon",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "however",
    "hundred",
    "i",
    "ie",
    "if",
    "in",
    "inc",
    "indeed",
    "interest",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "keep",
    "last",
    "latter",
    "latterly",
    "least",
    "less",
    "ltd",
    "made",
    "many",
    "may",
    "me",
    "meanwhile",
    "might",
    "mill",
    "mine",
    "more",
    "moreover",
    "most",
    "mostly",
    "move",
    "much",
    "must",
    "my",
    "myself",
    "name",
    "namely",
    "neither",
    "never",
    "nevertheless",
    "next",
    "nine",
    "no",
    "nobody",
    "none",
    "noone",
    "nor",
    "not",
    "nothing",
    "now",
    "nowhere",
    "of",
    "off",
    "often",
    "on",
    "once",
    "one",
    "only",
    "onto",
    "or",
    "other",
    "others",
    "otherwise",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "part",
    "per",
    "perhaps",
    "please",
    "put",
    "rather",
    "re",
    "same",
    "see",
    "seem",
    "seemed",
    "seeming",
    "seems",
    "serious",
    "several",
    "she",
    "should",
    "show",
    "side",
    "since",
    "sincere",
    "six",
    "sixty",
    "so",
    "some",
    "somehow",
    "someone",
    "something",
    "sometime",
    "sometimes",
    "somewhere",
    "still",
    "such",
    "system",
    "take",
    "ten",
    "than",
    "that",
    "the",
    "their",
    "them",
    "themselves",
    "then",
    "thence",
    "there",
    "thereafter",
    "thereby",
    "therefore",
    "therein",
    "thereupon",
    "these",
    "they",
    "thick",
    "thin",
    "third",
    "this",
    "those",
    "though",
    "three",
    "through",
    "throughout",
    "thru",
    "thus",
    "to",
    "together",
    "too",
    "top",
    "toward",
    "towards",
    "twelve",
    "twenty",
    "two",
    "un",
    "under",
    "until",
    "up",
    "upon",
    "us",
    "very",
    "via",
    "was",
    "we",
    "well",
    "were",
    "what",
    "whatever",
    "when",
    "whence",
    "whenever",
    "where",
    "whereafter",
    "whereas",
    "whereby",
    "wherein",
    "whereupon",
    "wherever",
    "whether",
    "which",
    "while",
    "whither",
    "who",
    "whoever",
    "whole",
    "whom",
    "whose",
    "why",
    "will",
    "with",
    "within",
    "without",
    "would",
    "yet",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "st",
    "years",
    "yourselves",
    "new",
    "used",
    "known",
    "year",
    "later",
    "including",
    "used",
    "end",
    "did",
    "just",
    "best",
    "using",
}
"""A set of common stopwords used for word filtering."""
