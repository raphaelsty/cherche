import collections
import pathlib
import re
import string
import typing

from ..utils import yield_batch_single
from .base import Query

__all__ = ["Norvig"]


class Norvig(Query):
    """Spelling corrector written by Peter Norvig:
    [How to Write a Spelling Corrector](https://norvig.com/spell-correct.html)

    Parameters
    ----------
    on
        Fields to use for fitting the spelling corrector on.

    Examples
    --------

    >>> from cherche import query, data

    >>> documents = data.load_towns()

    >>> corrector = query.Norvig(on = ["title", "article"], lower=True)

    >>> corrector.add(documents)
    Query Norvig
         Vocabulary: 967

    >>> corrector(q="tha citi af Parisa is in Fronce")
    'the city of paris is in france'

    >>> corrector(q=["tha citi af Parisa is in Fronce", "parisa"])
    ['the city of paris is in france', 'paris']

    References
    ----------
    1. [How to Write a Spelling Corrector](https://norvig.com/spell-correct.html)

    """

    def __init__(
        self,
        on: typing.Union[str, typing.List],
        lower: bool = True,
    ) -> None:
        super().__init__(on=on)

        self.occurrences = collections.Counter()
        self.lower = lower

    def __repr__(self) -> str:
        repr = super().__repr__()
        repr += f"\n\t Vocabulary: {len(self.occurrences)}"
        return repr

    def __call__(
        self, q: typing.Union[typing.List[str], str], **kwargs
    ) -> typing.Union[typing.List[str], str]:
        """Correct spelling errors in a given query."""
        queries = []
        for batch in yield_batch_single(q, desc="Spelling-correction"):
            if len(self.occurrences) == 0:
                queries.append(batch)
            else:
                queries.append(" ".join(map(self.correct, batch.split(" "))))
        return queries[0] if isinstance(q, str) else queries

    def correct(self, word: str) -> float:
        """Most probable spelling correction for word."""
        return max(
            self._candidates(word),
            key=lambda w: self._probability(w.lower() if self.lower else w),
        )

    def _probability(self, word: str) -> float:
        """Probability of `word`."""
        return self.occurrences[word] / sum(self.occurrences.values())

    def _candidates(self, word: str) -> set:
        """Generate possible spelling corrections for word."""
        return (
            self._known([word])
            or self._known(self._edits1(word))
            or self._known(self._edits2(word))
            or [word]
        )

    def _known(self, words: str) -> set:
        """The subset of `words` that appear in the dictionary."""
        return set(w for w in words if w in self.occurrences)

    def _edits1(self, word: str) -> set:
        """All edits that are one edit away from `word`."""
        letters = string.ascii_lowercase
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def _edits2(self, word: str) -> set:
        """All edits that are two edits away from `word`. s"""
        return (e2 for e1 in self._edits1(word) for e2 in self._edits1(e1))

    def add(self, documents: typing.Union[typing.List[typing.Dict], str]) -> "Norvig":
        """Fit Nervig spelling corrector."""
        documents = (
            documents
            if isinstance(documents, str)
            else " ".join(
                [
                    " ".join([document.get(field, "") for field in self.on])
                    for document in documents
                ]
            )
        )

        if self.lower:
            documents = documents.lower()

        self.occurrences.update(documents.split(" "))
        return self

    def _update_from_file(self, path_file: str) -> "Norvig":
        """Update dictionary from all words fetched from a raw text file."""
        with open(path_file, "r") as fp:
            self.occurrences.update(re.findall(r"\w+", fp.read().lower()))
        return self

    def reset(self) -> "Norvig":
        """Wipe dictionary."""
        self.occurrences = collections.Counter()
        return self
