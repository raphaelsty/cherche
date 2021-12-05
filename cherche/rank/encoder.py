__all__ = ["Encoder"]

from ..metric import cosine_distance
from .base import Ranker


class Encoder(Ranker):
    """SentenceBert Ranker.

    Parameters
    ----------

        on: Field of the documents to use for ranking.
        encoder: Encoding function to computes embeddings of the documents.
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
    ...    path = "encoder.pkl"
    ... )

    >>> documents = [
    ...     {"url": "ckb/github.com", "title": "Github library with PyTorch and Transformers .", "date": "10-11-2021"},
    ...     {"url": "mkb/github.com", "title": "Github Library with PyTorch .", "date": "22-11-2021"},
    ...     {"url": "blp/github.com", "title": "Github Library with Pytorch and Transformers .", "date": "22-11-2020"},
    ... ]

    >>> ranker = ranker.add(documents=documents)

    >>> print(ranker(q="Transformers", documents=documents, k=2))
    [{'cosine_distance': 0.6396294832229614,
      'date': '10-11-2021',
      'title': 'Github library with PyTorch and Transformers .',
      'url': 'ckb/github.com'},
     {'cosine_distance': 0.6396294832229614,
      'date': '22-11-2020',
      'title': 'Github Library with Pytorch and Transformers .',
      'url': 'blp/github.com'}]

    """

    def __init__(self, encoder, on: str, path: str = None, metric=cosine_distance) -> None:
        super().__init__(on=on, encoder=encoder, path=path, metric=metric)

    def __call__(self, q: str, documents: list, k: int = None) -> list:
        """Encode inputs query and ranks documents based on the similarity between the query and
        the selected field of the documents.

        Parameters
        ----------

            q: Query.
            documents: List of documents to re-rank.
            k: Number of documents to keeps.

        """
        emb_q = self.encoder(q) if q not in self.embeddings else q
        emb_documents = [
            self.embeddings.get(document[self.on], self.encoder(document[self.on]))
            for document in documents
        ]

        distances = self.metric(q=emb_q, documents=emb_documents)
        return self._rank(distances=distances, documents=documents, k=k)
