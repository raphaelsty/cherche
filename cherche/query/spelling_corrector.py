import pathlib
import string
import warnings
from typing import List

from .base import _SpellingCorrector

__all__ = ["Norvig"]


class Norvig(_SpellingCorrector):
    """Spelling corrector written by Peter Norvig: [How to Write a Spelling Corrector](https://norvig.com/spell-correct.html)

    Parameters
    ----------
    on
        Fields to use for fitting the spelling corrector on.
    big
        Use the big.txt provided by the Norvig spelling corrector. Contains
        english books from the Gutenberg project.

    Examples
    --------

    >>> from cherche import query, data

    >>> documents = data.load_towns()
    >>> corrector = query.Norvig(on=["title", "article"])
    >>> corrector.add(documents)
    >>> corrector
    Norvig query

    >>> corrector(q="tha citi af Parisa is in Fronce")
    'the city of Paris is in France'

    References
    ----------
    1. [How to Write a Spelling Corrector](https://norvig.com/spell-correct.html)

    """

    def __init__(
        self,
        on: List[str],
        big: bool = False,
    ) -> None:
        super().__init__(on=on)

        if big:
            path_big = pathlib.Path(__file__).parent.parent.joinpath("data/norvig.txt")
            self._update_from_file(path_file=path_big)

    def __call__(self, q: str, **kwargs) -> str:
        """Correct spelling errors in a given query."""
        if len(self.occurrences) == 0:
            warnings.warn("Spelling corrector has not be initialized, dictionary is empty")
            return q
        return " ".join(map(self.correct, q.split(" ")))

    def correct(self, word):
        """Most probable spelling correction for word."""
        return max(self._candidates(word), key=lambda w: self._probability(w))

    def _probability(self, word):
        """Probability of `word`."""
        return self.occurrences[word] / sum(self.occurrences.values())

    def _candidates(self, word):
        """Generate possible spelling corrections for word."""
        return self._known([word]) or self._known(self._edits1(word)) or self._known(self._edits2(word)) or [word]

    def _known(self, words):
        """The subset of `words` that appear in the dictionary."""
        return set(w for w in words if w in self.occurrences)

    def _edits1(self, word):
        """ All edits that are one edit away from `word`. """
        letters = string.ascii_lowercase
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def _edits2(self, word):
        """ All edits that are two edits away from `word`. s"""
        return (e2 for e1 in self._edits1(word) for e2 in self._edits1(e1))


if __name__ == '__main__':
    pass
