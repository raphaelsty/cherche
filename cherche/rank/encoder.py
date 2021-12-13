__all__ = ["Encoder"]

from ..similarity import cosine
from .base import Ranker


class Encoder(Ranker):
    """SentenceBert Ranker.

    Parameters
    ----------
    on
        Field to use to match the query to the documents.
    encoder
        Encoding function dedicated to documents and query.
    k
        Number of documents to retrieve. Default is None, i.e all documents that match the query
        will be retrieved.
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
    ...    {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> ranker = rank.Encoder(
    ...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ...    on = "article",
    ...    k = 2,
    ...    path = "encoder.pkl"
    ... )

    >>> ranker.add(documents=documents)
    Encoder ranker
         on: article
         k: 2
         similarity: cosine
         embeddings stored at: encoder.pkl

    >>> print(ranker(q="Paris", documents=documents))
    [{'article': 'Eiffel tower is based in Paris',
      'author': 'Wiki',
      'similarity': 0.49121392,
      'title': 'Eiffel tower'},
     {'article': 'This town is the capital of France',
      'author': 'Wiki',
      'similarity': 0.44376045,
      'title': 'Paris'}]

    """

    def __init__(
        self, encoder, on: str, k: int = None, path: str = None, similarity=cosine
    ) -> None:
        super().__init__(on=on, encoder=encoder, k=k, path=path, similarity=similarity)

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

        emb_q = self.encoder(q) if q not in self.embeddings else self.embeddings[q]
        emb_documents = [
            self.embeddings[document[self.on]]
            if document[self.on] in self.embeddings
            else self.encoder(document[self.on])
            for document in documents
        ]

        return self._rank(
            similarities=self.similarity(emb_q=emb_q, emb_documents=emb_documents),
            documents=documents,
        )