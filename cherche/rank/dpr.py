__all__ = ["DPR"]

from ..metric import dot_similarity
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
        metric: Distance / similarity measure to use i.e cherche.metric.cosine_distance or
            cherche.metric.dot_similarity.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import rank
    >>> from sentence_transformers import SentenceTransformer

    >>> ranker = rank.DPR(
    ...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
    ...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
    ...    on = "title",
    ...    k = 2,
    ...    path = "dpr.pkl"
    ... )

    >>> ranker
    DPR ranker
         on: title
         k: 2
         Metric: dot_similarity
         Embeddings stored at: dpr.pkl

    >>> documents = [
    ...     {"url": "ckb/github.com", "title": "Github library with PyTorch and Transformers .", "date": "10-11-2021"},
    ...     {"url": "mkb/github.com", "title": "Github Library with PyTorch .", "date": "22-11-2021"},
    ...     {"url": "blp/github.com", "title": "Github Library with Pytorch and Transformers .", "date": "22-11-2020"},
    ... ]

    Pre-compute embeddings of documents
    >>> ranker = ranker.add(documents=documents)

    >>> print(ranker(q="Transformers", documents=documents, k=2))
    [{'date': '10-11-2021',
      'dot_similarity': 54.095573,
      'title': 'Github library with PyTorch and Transformers .',
      'url': 'ckb/github.com'},
     {'date': '22-11-2020',
      'dot_similarity': 54.095573,
      'title': 'Github Library with Pytorch and Transformers .',
      'url': 'blp/github.com'}]

    """

    def __init__(
        self,
        encoder,
        query_encoder,
        on: str,
        k: int = None,
        path: str = None,
        metric=dot_similarity,
    ) -> None:
        super().__init__(on=on, encoder=encoder, k=k, path=path, metric=metric)
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
            self.embeddings.get(document[self.on], self.encoder(document[self.on]))
            for document in documents
        ]

        distances = self.metric(emb_q=emb_q, emb_documents=emb_documents)
        return self._rank(distances=distances, documents=documents)
