import tensorflow
import re
import html.entities
import typing as t
import nltk.sentiment.util

from .base import BaseTokenizer


class PottsTokenizer(BaseTokenizer):
    """
    Tokenizer based on `Christopher Potts' tokenizer <http://sentiment.christopherpotts.net/tokenizing.html>`_, released in 2011.

    This module is released under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License: https://creativecommons.org/licenses/by-nc-sa/3.0/ .
    """

    # noinspection RegExpRepeatedSpace
    # language=pythonregexp
    emoticon_re_string = r"""[<>]?[:;=8][\-o*']?[)\](\[dDpP/:}{@|\\]"""

    emoticon_re = re.compile(emoticon_re_string)

    words_re_string = "(" + "|".join([
        # Emoticons:
        emoticon_re_string
        ,
        # Phone numbers:
        # language=pythonregexp
        r"""(?:[+]?[01][\s.-]*)?(?:[(]?\d{3}[\s.)-]*)?\d{3}[\-\s.]*\d{4}"""
        ,
        # HTML tags:
        # language=pythonregexp
        r"""<[^>]+>"""
        ,
        # Twitter username:
        # language=pythonregexp
        r"""@[\w_]+"""
        ,
        # Twitter hashtags:
        # language=pythonregexp
        r"""#+[\w_]+[\w'_-]*[\w_]+"""
        ,
        # Words with apostrophes or dashes
        # language=pythonregexp
        r"""[a-z][a-z'_-]+[a-z]"""
        ,
        # Numbers, including fractions, decimals
        # language=pythonregexp
        r"""[+-]?\d+(?:[,/.:-]\d+)?"""
        ,
        # Words without apostrophes or dashes
        # language=pythonregexp
        r"""[\w_]+"""
        ,
        # Ellipsis dots
        # language=pythonregexp
        r"""[.](?:\s*[.])+"""
        ,
        # Everything else that isn't whitespace
        # language=pythonregexp
        r"""\S+"""
    ]) + ")"

    words_re = re.compile(words_re_string, re.I)

    # language=pythonregexp
    digit_re_string = r"&#\d+;"

    digit_re = re.compile(digit_re_string)

    # language=pythonregexp
    alpha_re_string = r"&\w+;"

    alpha_re = re.compile(alpha_re_string)

    amp = "&amp;"

    @classmethod
    def __html2string(cls, s: str) -> str:
        """
        Internal metod that seeks to replace all the HTML entities in s with their corresponding characters.
        """
        # First the digits:
        ents = set(cls.digit_re.findall(s))
        if len(ents) > 0:
            for ent in ents:
                entnum = ent[2:-1]
                try:
                    entnum = int(entnum)
                    s = s.replace(ent, chr(entnum))
                except (ValueError, KeyError):
                    pass
        # Now the alpha versions:
        ents = set(cls.alpha_re.findall(s))
        ents = filter((lambda x: x != cls.amp), ents)
        for ent in ents:
            entname = ent[1:-1]
            try:
                s = s.replace(ent, chr(html.entities.name2codepoint[entname]))
            except (ValueError, KeyError):
                pass
            s = s.replace(cls.amp, " and ")
        return s

    def tokenize_plain(self, text: str) -> str:
        # Fix HTML character entitites
        s = self.__html2string(text)
        # Tokenize
        words = self.words_re.findall(s)
        # Possible alter the case, but avoid changing emoticons like :D into :d:
        words = list(map(lambda x: x if self.emoticon_re.search(x) else x.lower(), words))
        # Re-join words
        result = " ".join(words)
        # Return the result
        return result


class PottsTokenizerWithNegation(PottsTokenizer):
    def tokenize_plain(self, text: str) -> str:
        words = super().tokenize_plain(text).split()
        nltk.sentiment.util.mark_negation(words, shallow=True)
        return " ".join(words)


__all__ = (
    "PottsTokenizer",
    "PottsTokenizerWithNegation",
)
