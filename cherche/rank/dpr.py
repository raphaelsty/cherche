from __future__ import annotations

__all__ = ["DPR"]


import typing

import more_itertools
import tqdm

from ..similarity import dot
from .base import MemoryStore, Ranker


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
    ... )

    >>> ranker.add(documents=documents)
    DPR ranker
        key: id
        on: title, article
        k: 2
        similarity: dot
        Embeddings pre-computed: 3

    >>> print(ranker.batch(
    ...    q=["Paris", "Montreal"],
    ...    documents={
    ...         0: [{"id": 0}, {"id": 1}, {"id": 2}],
    ...         1: [{"id": 0}, {"id": 1}, {"id": 2}]
    ...    }
    ... ))
    {0: [{'id': 0, 'similarity': 74.02354}, {'id': 1, 'similarity': 68.806526}],
     1: [{'id': 2, 'similarity': 74.88396}, {'id': 1, 'similarity': 60.880127}]}

    >>> print(ranker.batch(
    ...    q=["Paris", "Montreal"],
    ...    batch_size=1,
    ...    documents={
    ...         0: [{"id": 0}, {"id": 1}, {"id": 2}],
    ...         1: [{"id": 0}, {"id": 1}, {"id": 2}]
    ...    }
    ... ))
    {0: [{'id': 0, 'similarity': 74.02354}, {'id': 1, 'similarity': 68.80652}],
     1: [{'id': 2, 'similarity': 74.883965}, {'id': 1, 'similarity': 60.880142}]}

    >>> print(ranker(q="Paris", documents=[{"id": 0}, {"id": 1}, {"id": 2}]))
    [{'id': 0, 'similarity': 74.0235366821289},
     {'id': 1, 'similarity': 68.8065185546875}]

    >>> print(ranker(q="Montreal", documents=[{"id": 0}, {"id": 1}, {"id": 2}]))
    [{'id': 2, 'similarity': 74.88396453857422},
     {'id': 1, 'similarity': 60.88014221191406}]

    >>> print(ranker(q="Paris", documents=documents))
    [{'article': 'This town is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'similarity': 74.0235366821289,
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris',
      'author': 'Wiki',
      'id': 1,
      'similarity': 68.8065185546875,
      'title': 'Eiffel tower'}]

    >>> ranker += documents
    >>> print(ranker(q="Montreal", documents=[{"id": 0}, {"id": 1}, {"id": 2}]))
    [{'article': 'Montreal is in Canada.',
      'author': 'Wiki',
      'id': 2,
      'similarity': 74.88396453857422,
      'title': 'Montreal'},
     {'article': 'Eiffel tower is based in Paris',
      'author': 'Wiki',
      'id': 1,
      'similarity': 60.88014221191406,
      'title': 'Eiffel tower'}]

    """

    def __init__(
        self,
        key: str,
        on: str | list,
        encoder,
        query_encoder,
        k: int | typing.Optionnal = None,
        similarity=dot,
        store=MemoryStore(),
        path: str | typing.Optionnal = None,
    ) -> None:
        super().__init__(
            key=key, on=on, encoder=encoder, k=k, similarity=similarity, store=store
        )
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

        return self.rank(
            query_embedding=self.query_encoder(q),
            documents=documents,
        )

    def batch(
        self, q: typing.List[str], documents: dict, batch_size: int = 64, **kwargs
    ) -> dict:
        """Re-rank batch of documents-queries.

        Parameters
        ----------
        q
            List of queries.
        documents
            Batch of documents.
        batch_size
            Size of the batch.
        """
        rank = {}

        for batch_queries, batch in tqdm.tqdm(
            zip(
                more_itertools.chunked(q, batch_size),
                more_itertools.chunked(documents, batch_size),
            ),
            position=0,
            desc="Ranker batch queries.",
            total=1 + len(q) // batch_size,
        ):
            rank = {
                **rank,
                **self.rank_batch(
                    **{
                        "query_embeddings": self.query_encoder(batch_queries),
                        "batch": {idx: documents[idx] for idx in batch},
                        "n": len(rank),
                    }
                ),
            }

        return rank
