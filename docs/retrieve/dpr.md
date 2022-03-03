# DPR

The `retriever.DPR` model uses DPR-based models that encode queries and documents with two distinct models. It is compatible with the [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) DPR models. The encoder pre-computes document embeddings and uses [Faiss](https://github.com/facebookresearch/faiss) to quickly find the documents most similar to the query embedding.

Documents indexed by `retrieve.DPR` can be updated using batch with the `add` method. This method takes time because the document encoder will pre-compute the document embeddings and store them
in the `pickle` file associated with the `path` parameter. We can speed up the process with a GPU. When indexing documents, the encoder loads the `pickle` file that contains the embeddings in memory and updates it with the embeddings of the new documents.

If we want to deploy this retriever, we should rely on Pickle to serialize the retriever.

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

## Retriever DPR on GPU

To speed up the search for the most relevant documents, we can:

- Use the GPU to speed up the DPR model.
- Use the GPU to speed up faiss to retrieve documents.

To use faiss GPU, we need first to install faiss-gpu; we have to update the attribute `tree` of the retriever with the `faiss.index_cpu_to_gpu` method. After that, Faiss GPU significantly speeds up the search.

```sh
pip install faiss-gpu
```

```python
>>> import faiss

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
...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base', device="cuda").encode,
...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base', device="cuda").encode,
...    key = "id",
...    on = ["title", "article"],
...    k = 2,
...    path = "retriever_dpr.pkl"
... )

# 0 is the id of the GPU.
>>> retriever.tree = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, retriever.tree)

>>> retriever.add(documents)

>>> retriever("paris")
[{'id': 0, 'similarity': 0.9025790931437582},
 {'id': 2, 'similarity': 0.8160134832855334}]
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
