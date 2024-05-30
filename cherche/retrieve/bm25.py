__all__ = ["BM25"]

import typing

from lenlp import sparse

from .tfidf import TfIdf


class BM25(TfIdf):
    """TfIdf retriever based on cosine similarities.

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Fields to use to match the query to the documents.
    documents
        Documents in TFIdf retriever are static. The retriever must be reseted to index new
        documents.
    k
        Number of documents to retrieve. Default is `None`, i.e all documents that match the query
        will be retrieved.
    tfidf
        TfidfVectorizer class of Sklearn to create a custom TfIdf retriever.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve

    >>> documents = [
    ...     {"id": 0, "title": "Paris", "article": "Eiffel tower"},
    ...     {"id": 1, "title": "Montreal", "article": "Montreal is in Canada."},
    ...     {"id": 2, "title": "Paris", "article": "Eiffel tower"},
    ...     {"id": 3, "title": "Montreal", "article": "Montreal is in Canada."},
    ... ]

    >>> retriever = retrieve.BM25(
    ...     key="id",
    ...     on=["title", "article"],
    ...     documents=documents,
    ... )

    >>> documents = [
    ...     {"id": 4, "title": "Paris", "article": "Eiffel tower"},
    ...     {"id": 5, "title": "Montreal", "article": "Montreal is in Canada."},
    ...     {"id": 6, "title": "Paris", "article": "Eiffel tower"},
    ...     {"id": 7, "title": "Montreal", "article": "Montreal is in Canada."},
    ... ]

    >>> retriever = retriever.add(documents)

    >>> print(retriever(q=["paris", "canada"], k=4))
    [[{'id': 6, 'similarity': 0.5404109029445249},
      {'id': 0, 'similarity': 0.5404109029445249},
      {'id': 2, 'similarity': 0.5404109029445249},
      {'id': 4, 'similarity': 0.5404109029445249}],
     [{'id': 7, 'similarity': 0.3157669764669935},
      {'id': 5, 'similarity': 0.3157669764669935},
      {'id': 3, 'similarity': 0.3157669764669935},
      {'id': 1, 'similarity': 0.3157669764669935}]]

    >>> print(retriever(["unknown", "montreal paris"], k=2))
    [[],
     [{'id': 7, 'similarity': 0.7391866872635209},
      {'id': 5, 'similarity': 0.7391866872635209}]]


    >>> print(retriever(q="paris"))
    [{'id': 6, 'similarity': 0.5404109029445249},
     {'id': 0, 'similarity': 0.5404109029445249},
     {'id': 2, 'similarity': 0.5404109029445249},
     {'id': 4, 'similarity': 0.5404109029445249}]

    References
    ----------
    1. [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
    2. [Python: tf-idf-cosine: to find document similarity](https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity)

    """

    def __init__(
        self,
        key: str,
        on: typing.Union[str, list],
        documents: typing.List[typing.Dict[str, str]] = None,
        count_vectorizer: sparse.BM25Vectorizer = None,
        k: typing.Optional[int] = None,
        batch_size: int = 1024,
        fit: bool = True,
    ) -> None:
        count_vectorizer = (
            sparse.BM25Vectorizer(
                normalize=True, ngram_range=(3, 5), analyzer="char_wb"
            )
            if count_vectorizer is None
            else count_vectorizer
        )

        super().__init__(
            key=key,
            on=on,
            documents=documents,
            tfidf=count_vectorizer,
            k=k,
            batch_size=batch_size,
            fit=fit,
        )
