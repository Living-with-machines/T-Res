import re
from typing import Optional

LOWER = False
DIGIT_0 = False
UNK_TOKEN = "#UNK#"
BRACKETS = {
    "-LCB-": "{",
    "-LRB-": "(",
    "-LSB-": "[",
    "-RCB-": "}",
    "-RRB-": ")",
    "-RSB-": "]",
}


class Vocabulary:
    """
    A class representing a vocabulary object used for storing references to embeddings.

    Credit:
        The code for this class and its methods is taken from the `REL: Radboud
        Entity Linker <https://github.com/informagi/REL/>`_ Github repository. See
        `the original script <https://github.com/informagi/REL/blob/main/src/REL/vocabulary.py>`_
        for more information.

        ::

            Reference:

            @inproceedings{vanHulst:2020:REL,
            author =    {van Hulst, Johannes M. and Hasibi, Faegheh and Dercksen, Koen and Balog, Krisztian and de Vries, Arjen P.},
            title =     {REL: An Entity Linker Standing on the Shoulders of Giants},
            booktitle = {Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
            series =    {SIGIR '20},
            year =      {2020},
            publisher = {ACM}
            }

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
        """
        new_id = len(self.id2word)
        self.id2word.append(token)
        self.word2id[token] = new_id
        self.idtoword[new_id] = token

    def size(self) -> int:
        """
        Get the size of the vocabulary.

        Returns:
            int: The number of words in the vocabulary.
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
        """
        tok = Vocabulary.normalize(token)
        return self.word2id.get(tok, self.unk_id)
