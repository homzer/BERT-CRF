from src.input.tokenizer import Tokenizer
from src.utils import ConfigUtil


class DataProcessor:
    def __init__(self, vocab_file):
        self.tokenizer = Tokenizer(vocab_file)

    def texts2ids(self, texts: list, length: int):
        """
        Tokenize, padding and truncate several texts.
        And convert it to several lists of id.
        :param texts: `list` of `str`
        :param length: `int` the fixed length of text.
        :return: 2-D `list` the ids of tokens.
        """
        return [self.text2ids(text, length) for text in texts]

    def text2ids(self, text: str, length: int):
        """
        Tokenize, padding and truncate a text.
        And convert it to a list of id.
        :param text: str
        :param length: the fixed length of text.
        :return: `list` the ids of tokens.
        """
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.tokens2ids(tokens)
        # Padding
        while len(token_ids) < length:
            token_ids.append(0)
        # Truncate
        if len(token_ids) > length:
            token_ids = token_ids[:length]
        assert len(token_ids) == length
        return token_ids
