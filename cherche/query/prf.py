import typing

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..utils import yield_batch_single
from .base import Query

__all__ = ["PRF"]


class PRF(Query):
    """Pseudo (or blind) Relevance-Feedback module. The Query-Augmentation method applies a fast
    document retrieving method and then extracts keywords from relevant documents. Thus, we have to
    retrieve top words from relevant documents to give a proper augmentation of a given query.

    Parameters
    ----------
    on
        Fields to use for fitting the spelling corrector on.
    tf
        defaults to sklearn.feature_extraction.text.TfidfVectorizer.
        If you want to implement your own tf, it needs to follow the sklearn base API and provides the `transform`
        `fit_transform` and `get_feature_names_out` methods. See sklearn documentation for more information.
    nb_docs
        Number of documents from which to retrieve top-terms.
    nb_terms_per_doc
        Number of terms to extract from each top documents retrieved.

    Examples
    --------

    >>> from cherche import query, data

    >>> documents = data.load_towns()

    >>> prf = query.PRF(
    ...     on=["title", "article"],
    ...     nb_docs=8, nb_terms_per_doc=1,
    ...     documents=documents
    ... )

    >>> prf
    Query PRF
        on       : title, article
        documents: 8
        terms    : 1

    >>> prf(q="Europe")
    'Europe art metro space science bordeaux paris university significance'

    >>> prf(q=["Europe", "Paris"])
    ['Europe art metro space science bordeaux paris university significance', 'Paris received paris club subway billion source tour tournament']

    References
    ----------
    1. [Relevance feedback and pseudo relevance feedback](https://nlp.stanford.edu/IR-book/html/htmledition/relevance-feedback-and-pseudo-relevance-feedback-1.html)
    2. [Blind Feedback](https://en.wikipedia.org/wiki/Relevance_feedback#Blind_feedback)

    """

    def __init__(
        self,
        on: typing.Union[str, list],
        documents: list,
        tf: sklearn.feature_extraction.text.CountVectorizer = TfidfVectorizer(),
        nb_docs: int = 5,
        nb_terms_per_doc: int = 3,
    ) -> None:
        super().__init__(on=on)
        self.nb_docs = nb_docs
        self.nb_terms_per_doc = nb_terms_per_doc
        self.func = tf

        documents = [
            " ".join([doc.get(field, "") for field in self.on]) for doc in documents
        ]

        self.tf = tf
        self.matrix = self.tf.fit_transform(documents)
        self.vocabulary = self.tf.get_feature_names_out()

    def __repr__(self) -> str:
        repr = super().__repr__()
        repr += f"\n\t on       : {', '.join(self.on)}"
        repr += f"\n\t documents: {self.nb_docs}"
        repr += f"\n\t terms    : {self.nb_terms_per_doc}"
        return repr

    def __call__(
        self, q: typing.Union[typing.List[str], str], **kwargs
    ) -> typing.Union[typing.List[str], str]:
        """Augment a given query with new terms."""
        # Extract top terms from the documents wrt. a given query
        queries = []
        for batch in yield_batch_single(q, desc="Query-augmentation"):
            top_terms = self._retrieve_top_terms(q=batch)
            # Augment the query
            batch += " " + " ".join(
                [term for term in top_terms if term not in batch.split(" ")]
            )
            queries.append(batch)
        return queries[0] if isinstance(q, str) else queries

    def _retrieve_top_terms(self, q: str) -> typing.List[str]:
        """Retrieve new terms to augment a given query. Disregard dupes terms."""
        query = self.tf.transform([q])
        # First, retrieve top documents wrt. a given query
        idx_documents = self._retrieve_documents(matrix=self.matrix, query=query)

        # Then, extract a given number of terms from each document
        terms = []
        for idx_document in idx_documents:
            document = self.matrix[idx_document].toarray().squeeze()
            terms.extend(
                self._retrieve_terms(document=document, vocabulary=self.vocabulary)
            )
        return terms

    def _retrieve_documents(self, matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Retrieve pertinents documents given a user query."""
        # Compute ranking score
        results = cosine_similarity(matrix, query).squeeze()

        # Fetching top-documents then sorting by score
        ind = np.argpartition(results, -self.nb_docs)[-self.nb_docs :]
        ind = ind[np.argsort((-results)[ind])]

        return ind

    def _retrieve_terms(
        self, document: np.ndarray, vocabulary: np.ndarray
    ) -> typing.List[str]:
        """Extract top-terms in a given document."""
        ind = np.argpartition(document, -self.nb_terms_per_doc)[
            -self.nb_terms_per_doc :
        ]
        ind = ind[np.argsort((-document)[ind])]
        return vocabulary[ind].tolist()
