import typing
import warnings

import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np

from .base import Query

__all__ = ["PRF"]


class PRF(Query):
    """Pseudo (or blind) Relevance-Feedback module.
    Query-Augmentation method consisting of applying a fast document retrieving method, then extracting keywords from
    top documents.
    The main principle of this method is that the top documents from any working ranking method should give at least
    great results (ie: the user almost always considers the first documents as relevant). Thus, we juste have to
    retrieve top-words from relevant documents to give a proper augmentation of a given query.

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
    >>> prf = query.PRF(on=["title", "article"], nb_docs=8, nb_terms_per_doc=1)
    >>> prf.add(documents)
    >>> prf
    Pseudo Relevance Feedback model
         on: title, article
         nb docs: 8
         nb terms per doc: 1

    >>> prf(q="europe")
    'europe centres metro space art paris bordeaux significance university'


    References
    ----------
    1. [Relevance feedback and pseudo relevance feedback](https://nlp.stanford.edu/IR-book/html/htmledition/relevance-feedback-and-pseudo-relevance-feedback-1.html)
    2. [Blind Feedback](https://en.wikipedia.org/wiki/Relevance_feedback#Blind_feedback)

    """

    def __init__(
        self,
        on: typing.Union[str, list],
        tf: sklearn.feature_extraction.text.CountVectorizer = None,
        nb_docs: int = 5,
        nb_terms_per_doc: int = 3,
    ) -> None:
        super().__init__(on=on)
        self.nb_docs = nb_docs
        self.nb_terms_per_doc = nb_terms_per_doc
        self.documents: typing.List[str] = list()
        self.vocabulary: typing.Set[str] = set()

        if tf is None:
            self.tf = TfidfVectorizer
        else:
            self.tf = tf

    @property
    def type(self):
        return "prf"

    def __repr__(self) -> str:
        repr = "Pseudo Relevance Feedback model"
        repr += f"\n\t on: {', '.join(self.on)}"
        repr += f"\n\t nb docs: {self.nb_docs}"
        repr += f"\n\t nb terms per doc: {self.nb_terms_per_doc}"
        return repr

    def __call__(self, q: str, **kwargs) -> str:
        """Augment a given query with new terms."""
        if len(self.documents) == 0:
            warnings.warn("PRF has not be initialized, no document has been found.")
            return q

        # Quick query cleanup
        q = q.lower()

        # Extract top terms from the documents wrt. a given query
        top_terms = self._retrieve_top_terms(q=q)

        # Augment the query
        q += " " + " ".join([term for term in top_terms if term not in q.split(" ")])

        return q

    def add(self, documents: typing.Union[typing.List[typing.Dict], str]) -> "PRF":
        if isinstance(documents, str):
            self._update_vocabulary(documents)
            self._update_documents(documents)
        elif isinstance(documents, list) and len(documents) > 0:
            if isinstance(documents[0], dict):
                documents = [
                    " ".join([document.get(field, "") for field in self.on])
                    for document in documents
                ]
                for document in documents:
                    self._update_vocabulary(document)
                    self._update_documents(document)
        else:
            raise ValueError(
                f"Unsupported document format for updating PRF internal parameters : {type(documents)}"
            )
        return self

    def _retrieve_top_terms(self, q: str) -> typing.List[str]:
        """Retrieve new terms to augment a given query. Disregard dupes terms."""
        tf = self.tf(vocabulary=self.vocabulary)
        documents = tf.fit_transform(self.documents)
        query = tf.transform([q])
        vocabulary = tf.get_feature_names_out()

        # First, retrieve top documents wrt. a given query
        idx_documents = self._retrieve_documents(documents=documents, query=query)

        # Then, extract a given number of terms from each document
        terms = []
        for idx_document in idx_documents:
            document = documents[idx_document].toarray().squeeze()
            terms.extend(self._retrieve_terms(document=document, vocabulary=vocabulary))
        return terms

    def _retrieve_documents(
        self, documents: np.ndarray, query: np.ndarray
    ) -> np.ndarray:
        """Retrieve pertinents documents given a user query."""
        # Compute ranking score
        results = cosine_similarity(documents, query).squeeze()

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

    def _update_vocabulary(self, document: str) -> None:
        self.vocabulary.update(document.lower().split(" "))

    def _update_documents(self, document: str) -> None:
        self.documents.append(document.lower())
