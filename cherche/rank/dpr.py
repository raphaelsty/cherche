__all__ = ["DPR"]

import typing

from cherche import similarity

from ..similarity import dot
from .base import Ranker


class DPR(Ranker):
    """DPR ranks documents using distinct models to encode the query and
    document contents.

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Fields to use to match the query to the documents.
    encoder
        Encoding function dedicated to documents.
    query_encoder
        Encoding function dedicated to the query.
    k
        Number of documents to reorder. The default value is None, i.e. all documents will be
        reordered and returned.
    path
        Path to the file dedicated to storing the embeddings. The ranker will read this file if it
        already exists to load the embeddings and will update it when documents are added.
    similarity
        Similarity measure to compare documents embeddings and query embedding (similarity.cosine
        or similarity.dot).

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import rank
    >>> from sentence_transformers import SentenceTransformer

    >>> documents = [
    ...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> ranker = rank.DPR(
    ...    key = "id",
    ...    on = ["title", "article"],
    ...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
    ...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
    ...    k = 2,
    ...    path = "test_dpr.pkl"
    ... )

    >>> ranker.add(documents=documents)
    DPR ranker
         key: id
         on: title, article
         k: 2
         similarity: dot
         embeddings stored at: test_dpr.pkl

    >>> print(ranker(q="Paris", documents=[{"id": 0}, {"id": 1}, {"id": 2}]))
    [{'id': 0, 'similarity': 74.02355}, {'id': 1, 'similarity': 68.80651}]

    >>> print(ranker(q="Paris", documents=documents))
    [{'article': 'This town is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'similarity': 74.02355,
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris',
      'author': 'Wiki',
      'id': 1,
      'similarity': 68.80651,
      'title': 'Eiffel tower'}]

    >>> ranker += documents

    >>> print(ranker(q="Paris", documents=[{"id": 0}, {"id": 1}, {"id": 2}]))
    [{'article': 'This town is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'similarity': 74.02355,
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris',
      'author': 'Wiki',
      'id': 1,
      'similarity': 68.80651,
      'title': 'Eiffel tower'}]

    """

    def __init__(
        self,
        key: str,
        on: typing.Union[str, list],
        encoder,
        query_encoder,
        k: int = None,
        path: str = None,
        similarity: similarity = dot,
    ) -> None:
        super().__init__(key=key, on=on, encoder=encoder, k=k, path=path, similarity=similarity)
        self.query_encoder = query_encoder

    def __call__(self, q: str, documents: list, **kwargs) -> list:
        """Encode inputs query and ranks documents based on the similarity between the query and
        the selected field of the documents.

        Parameters
        ----------
        q
            Input query.
        documents
            List of documents to rank.

        """
        if not documents:
            return []

        emb_q = self.query_encoder(q) if q not in self.embeddings else self.embeddings[q]
        emb_documents = self._emb_documents(documents=documents)
        return self._rank(
            similarities=self.similarity(emb_q=emb_q, emb_documents=emb_documents),
            documents=documents,
        )
