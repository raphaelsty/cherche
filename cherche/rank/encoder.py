__all__ = ["Encoder"]

from ..metric import cosine_distance
from .base import Ranker


class Encoder(Ranker):
    """SentenceBert Ranker.

    Parameters
    ----------

        on: Field of the documents to use for ranking.
        encoder: Encoding function to computes embeddings of the documents.
        k: Number of documents to keep.
        path: Path of the file dedicated to store the embeddings as a pickle file.
        metric: Distance / similarity measure to use i.e cherche.metric.cosine_distance or
            cherche.metric.dot_similarity.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import rank
    >>> from sentence_transformers import SentenceTransformer

    >>> ranker = rank.Encoder(
    ...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ...    on = "title",
    ...    k = 2,
    ...    path = "encoder.pkl"
    ... )

    >>> ranker
    Encoder ranker
         on: title
         k: 2
         Metric: cosine_distance
         Embeddings stored at: encoder.pkl

    >>> documents = [
    ...     {"url": "ckb/github.com", "title": "Github library with PyTorch and Transformers .", "date": "10-11-2021"},
    ...     {"url": "mkb/github.com", "title": "Github Library with PyTorch .", "date": "22-11-2021"},
    ...     {"url": "blp/github.com", "title": "Github Library with Pytorch and Transformers .", "date": "22-11-2020"},
    ... ]

    Pre-compute embeddings of documents
    >>> ranker = ranker.add(documents=documents)

    >>> print(ranker(q="Transformers", documents=documents))
    [{'cosine_distance': 0.6396294832229614,
      'date': '10-11-2021',
      'title': 'Github library with PyTorch and Transformers .',
      'url': 'ckb/github.com'},
     {'cosine_distance': 0.6396294832229614,
      'date': '22-11-2020',
      'title': 'Github Library with Pytorch and Transformers .',
      'url': 'blp/github.com'}]

    """

    def __init__(
        self, encoder, on: str, k: int = None, path: str = None, metric=cosine_distance
    ) -> None:
        super().__init__(on=on, encoder=encoder, k=k, path=path, metric=metric)

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

        emb_q = self.encoder(q) if q not in self.embeddings else self.embeddings[q]
        emb_documents = [
            self.embeddings[document[self.on]]
            if document[self.on] in self.embeddings
            else self.encoder(document[self.on])
            for document in documents
        ]

        distances = self.metric(emb_q=emb_q, emb_documents=emb_documents)
        return self._rank(distances=distances, documents=documents)
