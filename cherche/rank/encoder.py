from __future__ import annotations

__all__ = ["Encoder"]

import typing

import tqdm
import more_itertools

from ..similarity import cosine
from .base import MemoryStore, Ranker


class Encoder(Ranker):
    """SentenceBert Ranker.

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Fields to use to match the query to the documents.
    encoder
        Encoding function dedicated to documents and query.
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

    >>> ranker = rank.Encoder(
    ...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ...    key = "id",
    ...    on = ["title", "article"],
    ...    k = 2,
    ... )

    >>> ranker.add(documents=documents)
    Encoder ranker
        key: id
        on: title, article
        k: 2
        similarity: cosine
        Embeddings pre-computed: 3

    >>> print(ranker(q="Paris", documents=[{"id": 0}, {"id": 1}, {"id": 2}]))
    [{'id': 0, 'similarity': 0.6605139970779419},
     {'id': 1, 'similarity': 0.5142562985420227}]

    >>> print(ranker(q="Montreal", documents=[{"id": 0}, {"id": 1}, {"id": 2}]))
    [{'id': 2, 'similarity': 0.799931526184082},
     {'id': 0, 'similarity': 0.3506389558315277}]

    >>> print(ranker.batch(
    ...    q=["Paris", "Montreal"],
    ...    documents={
    ...         0: [{"id": 0}, {"id": 1}, {"id": 2}],
    ...         1: [{"id": 0}, {"id": 1}, {"id": 2}]
    ...    }
    ... ))

    >>> print(ranker(q="Paris", documents=documents))
    [{'article': 'This town is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'similarity': 0.6605141758918762,
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris',
      'author': 'Wiki',
      'id': 1,
      'similarity': 0.5142566561698914,
      'title': 'Eiffel tower'}]


    >>> ranker += documents

    >>> print(ranker(q="Paris", documents=[{"id": 0}, {"id": 1}, {"id": 2}]))
    [{'article': 'This town is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'similarity': 0.6605141758918762,
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris',
      'author': 'Wiki',
      'id': 1,
      'similarity': 0.5142566561698914,
      'title': 'Eiffel tower'}]

    """

    def __init__(
        self,
        on: str | list,
        key: str,
        encoder,
        k: int | typing.Optionnal = None,
        similarity=cosine,
        store=MemoryStore(),
        path: str | typing.Optionnal = None,
    ) -> None:
        super().__init__(
            key=key,
            on=on,
            encoder=encoder,
            k=k,
            similarity=similarity,
            store=store,
        )

    def __call__(self, q: str, documents: list, **kwargs) -> list:
        """Encode input query and ranks documents based on the similarity between the query and
        the selected field of the documents.

        https://pymilvus.readthedocs.io/en/latest/tutorial.html
        status, documents = client.get_entity_by_id(collection_name, [id_1, id_2])

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
            query_embedding=self.encoder(q),
            documents=documents,
        )

    def batch(self, q: list[str], documents: dict, batch_size: int, **kwargs) -> dict:
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

        for batch in tqdm.tqdm(
            more_itertools.chunked(q, batch_size),
            position=0,
            desc="Retriever batch queries.",
            total=1 + len(q) // batch_size,
        ):

            rank = {
                **rank,
                **self.rank_batch(
                    **{
                        "query_embeddings": self.encoder(batch),
                        "documents": documents,
                        "n": len(rank),
                    }
                ),
            }

        return rank
