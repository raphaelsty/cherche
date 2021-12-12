__all__ = ["DPR"]

from ..similarity import dot
from .base import Ranker


class DPR(Ranker):
    """DPR is dedicated to rank documents using distinct models to encode the query and the
    documents contents.

    Parameters
    ----------

        on: Field of the documents to use for ranking.
        encoder: Encoding function to computes embeddings of the documents.
        k: Number of documents to keep.
        path: Path of the file dedicated to store the embeddings as a pickle file.
        similarity: Similarity measure to use i.e similarity.cosine or similarity.dot.

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

    >>> ranker = rank.DPR(
    ...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
    ...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
    ...    on = "article",
    ...    k = 2,
    ...    path = "test_dpr.pkl"
    ... )

    >>> ranker.add(documents=documents)
    DPR ranker
         on: article
         k: 2
         similarity: dot
         embeddings stored at: test_dpr.pkl

    >>> print(ranker(q="Paris", documents=documents, k=2))
    [{'article': 'Eiffel tower is based in Paris',
      'author': 'Wiki',
      'similarity': 69.8168,
      'title': 'Eiffel tower'},
     {'article': 'This town is the capital of France',
      'author': 'Wiki',
      'similarity': 67.30965,
      'title': 'Paris'}]

    """

    def __init__(
        self,
        encoder,
        query_encoder,
        on: str,
        k: int = None,
        path: str = None,
        similarity=dot,
    ) -> None:
        super().__init__(on=on, encoder=encoder, k=k, path=path, similarity=similarity)
        self.query_encoder = query_encoder

    def __call__(self, q: str, documents: list, **kwargs) -> list:
        """Encode inputs query and ranks documents based on the similarity between the query and
        the selected field of the documents.

        Parameters
        ----------

            q: Query.
            documents: List of documents to re-rank.

        """
        if not documents:
            return []

        emb_q = self.query_encoder(q)
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
