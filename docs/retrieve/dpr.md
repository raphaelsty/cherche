# DPR

The `retriever.DPR` model allows the use of framework that encode queries and documents with two distinct models. It is compatible with the [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) DPR models. The encoder pre-computes document embeddings and uses [Faiss](https://github.com/facebookresearch/faiss) to quickly find the documents most similar to the query embedding.

Documents indexed by `retrieve.DPR` can be updated in mini-batch with the `add` method.
This method takes time because the document encoder will pre-compute the document embeddings and store them
in the `pickle` file associated with the `path` parameter. You can speed up the process with a GPU.
When indexing documents, the encoder loads the `pickle` file that contains the embeddings in memory
and updates it with the embeddings of the new documents. It is simply a dictionary with the document
identifier as key the embedding as value.

If you want to deploy this retriever, you should move the pickle file that contains pre-computed
embeddings and all the documents to the target machine or simply use pickle to serialize the
retriever.

```python
>>> from cherche import retrieve
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {
...        "id": 0,
...        "article": "Paris is the capital and most populous city of France",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    },
...    {
...        "id": 1,
...        "article": "Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    },
...    {
...        "id": 2,
...        "article": "The City of Paris is the centre and seat of government of the region and province of Île-de-France .",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    }
... ]

>>> retriever = retrieve.DPR(
...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
...    key = "id",
...    on = ["title", "article"],
...    k = 2,
...    path = "retriever_dpr.pkl"
... )

>>> retriever = retriever.add(documents=documents)

>>> retriever("france")
[{'id': 1, 'similarity': 0.01113}, {'id': 0, 'similarity': 0.01113}]
```

## Map keys to documents

```python
>>> retriever += documents
>>> retriever("france")
[{'id': 1,
  'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.01113},
 {'id': 0,
  'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.01113}]
```

## Custom DPR

You can use your own models within `retrieve.DPR`. They should encodes a list of documents `list[str]` which returns a numpy array with dimensions `(number of documents, embedding size)`. These models should also encode a query (str) and return an embedding of size `(embedding dimension, )`.

Here is an example of how to integrate a custom DPR model:

```python
import numpy as np

from cherche import retrieve
from sentence_transformers import SentenceTransformer

class CustomDPR:

    def __init__(self):
      """Custom DPR retriever."""
      # Document encoder
      self.encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')
      # Query encoder
      self.query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base')

    def documents(self, documents):
      """Documents encoder."""
      return self.encoder.encode(documents)

    def query(self, query):
      """Query encoder."""
      return self.query_encoder.encode(query)

model = CustomDPR()

# Your model should pass these tests, i.e Sentence Bert API.
assert model.documents(["Paris", "France", "Bordeaux"]).shape[0] == 3 
assert isinstance(model.documents(["Paris", "France", "Bordeaux"]), np.ndarray)

assert len(model.documents("Paris").shape) == 1
assert isinstance(model.documents("Paris"), np.ndarray)

assert model.query(["Paris", "France", "Bordeaux"]).shape[0] == 3
assert isinstance(model.query(["Paris", "France", "Bordeaux"]), np.ndarray)

assert len(model.documents("Paris").shape) == 1
assert isinstance(model.query("Paris"), np.ndarray)

retriever = retrieve.DPR(
    encoder = model.documents,
    query_encoder = model.query,
    key = "id",
    on = ["title", "article"],
    k = 2,
    path = "custom_dpr.pkl"
)
```
