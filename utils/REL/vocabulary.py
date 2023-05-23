import re
from typing import Optional

LOWER = False
"""A boolean variable indicating whether token normalization should convert
tokens to lowercase. It is set to ``False``."""

DIGIT_0 = False
"""A boolean variable indicating whether digits should be replaced with
``'0'`` during token normalization. It is set to ``False``."""

UNK_TOKEN = "#UNK#"
"""A string representing the unknown token. It is set to ``"#UNK#"``."""

BRACKETS = {
    "-LCB-": "{",
    "-LRB-": "(",
    "-LSB-": "[",
    "-RCB-": "}",
    "-RRB-": ")",
    "-RSB-": "]",
}
"""A dictionary that maps specific bracket tokens to their corresponding symbols."""


class Vocabulary:
    """
    A class representing a vocabulary object used for storing references to embeddings.
    """

    unk_token = UNK_TOKEN

    def __init__(self):
        self.word2id = {}
        self.idtoword = {}

        self.id2word = []
        self.counts = []
        self.unk_id = 0
        self.first_run = 0

    @staticmethod
    def normalize(
        token: str, lower: Optional[bool] = LOWER, digit_0: Optional[bool] = DIGIT_0
    ) -> str:
        """
        Normalise the given token based on the specified normalisation rules.

        Arguments:
            token (str): The token to be normalized.
            lower (bool): Flag indicating whether token should be converted to
                lowercase. Defaults to ``False``.
            digit_0 (bool): Flag indicating whether digits should be replaced
                with ``'0'`` during normalization. Defaults to ``False``.

        Returns:
            str: The normalized token.

        Example:
            >>> Vocabulary.normalize("Hello, World!", lower=True, digit_0=True)
            'hello, world0'

        Note:
            Special tokens, like the unknown token (``#UNK#``), start token
            (``<s>``), and end token (``</s>``) are not affected by the
            normalisation process.

            Certain bracket tokens are replaced with their corresponding
            symbols defined in the :py:const:`~utils.REL.vocabulary.BRACKETS`
            dictionary.
        """
        if token in [Vocabulary.unk_token, "<s>", "</s>"]:
            return token
        elif token in BRACKETS:
            token = BRACKETS[token]
        else:
            if digit_0:
                token = re.sub("[0-9]", "0", token)

        if lower:
            return token.lower()
        else:
            return token

    def add_to_vocab(self, token: str) -> None:
        """
        Add the given token to the vocabulary.

        Arguments:
            token (str): The token to be added to the vocabulary.

        Returns:
            None.

        Example:
            >>> vocab = Vocabulary()
            >>> vocab.add_to_vocab("apple")
            >>> vocab.add_to_vocab("banana")
            >>> vocab.size()
            2

        Note:
            This method assigns a new ID to the token and updates the
            necessary dictionaries and lists.

            If the token is already present in the vocabulary, it will not be
            added again.
        """
        new_id = len(self.id2word)
        self.id2word.append(token)
        self.word2id[token] = new_id
        self.idtoword[new_id] = token

    def size(self) -> int:
        """
        Get the size of the vocabulary.

        Returns:
            int: The number of unique words in the vocabulary.

        Example:
            >>> vocab = Vocabulary()
            >>> vocab.add_to_vocab("apple")
            >>> vocab.add_to_vocab("banana")
            >>> vocab.size()
            2
        """
        return len(self.id2word)

    def get_id(self, token: str) -> int:
        """
        Get the ID associated with the given token from the vocabulary.

        Args:
            token (str): The token for which to retrieve the ID.

        Returns:
            int: The ID of the token in the vocabulary, or the ID of the
            unknown token if the token is not found.

        Example:
            >>> vocab = Vocabulary()
            >>> vocab.add_to_vocab("apple")
            >>> vocab.add_to_vocab("banana")
            >>> vocab.get_id("apple")
            0
            >>> vocab.get_id("orange")
            1
            >>> vocab.get_id("grape")
            0

        Note:
            This method normalizes the token using the defined normalisation
            rules before retrieving the ID.

            If the normalized token is not present in the vocabulary, the ID
            of the unknown token is returned.
        """
        tok = Vocabulary.normalize(token)
        return self.word2id.get(tok, self.unk_id)
