# Encoder

The `retriever.Encoder` model encodes queries and documents within a single model. It is compatible
with the [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) models. The
encoder pre-computes document embeddings and uses [Faiss](https://github.com/facebookresearch/faiss)
to quickly find the documents most similar to the query embedding.

Documents indexed by `retrieve.Encoder` can be updated in mini-batch with the `add` method.
This method takes time because the encoder will pre-compute the document embeddings and store them
in the `pickle` file associated with the `path` parameter. You can speed up the process with a GPU.

When indexing documents, the encoder loads the `pickle` file that contains the embeddings in memory
and updates it with the embeddings of the new documents. It is simply a dictionary with the document
identifier as key the embedding as value.

If we want to deploy this retriever, we should move the pickle file that contains pre-computed embeddings and all the documents to the target machine or serialize the retriever as a pickle file.

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

>>> retriever = retrieve.Encoder(
...    key = "id",
...    on = ["title", "article"],
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    k = 2,
...    path = "all-mpnet-base-v2.pkl"
... )

>>> retriever = retriever.add(documents=documents)

>>> retriever("paris")
[{'id': 0, 'similarity': 0.9025790931437582},
 {'id': 2, 'similarity': 0.8160134832855334}]
```

## Retriever Encoder on GPU

To speed up the search for the most relevant documents, we can:

- Use the GPU to speed up the encoder.
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

>>> retriever = retrieve.Encoder(
...    key = "id",
...    on = ["title", "article"],
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda").encode,
...    k = 2,
...    path = "all-mpnet-base-v2.pkl"
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
[{'id': 0,
  'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.9025790931437582},
 {'id': 2,
  'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.8160134832855334}]
```

## Custom Encoder

We can use custom models within `retrieve.Encoder`. It should encode a list of documents `list[str]` and return a NumPy array with dimensions `(number of documents, embedding size)`. This model should also encode a query (str) and return an embedding of size `(embedding dimension, )`. For example, we could use word embeddings to encode documents and queries.

Here is an example of how to integrate a custom encoder:

```python
import numpy as np
from cherche import retrieve
from sentence_transformers import SentenceTransformer

class CustomEncoder:

    def __init__(self):
      """Custom Encoder retriever."""
      self.encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def encode(self, documents):
      """Documents encoder."""
      return self.encoder.encode(documents)

model = CustomEncoder()

# Your model should pass these tests, i.e Sentence Bert API.
assert model.encode(["Paris", "France", "Bordeaux"]).shape[0] == 3 
assert isinstance(model.encode(["Paris", "France", "Bordeaux"]), np.ndarray)

assert len(model.encode("Paris").shape) == 1
assert isinstance(model.encode("Paris"), np.ndarray)

retriever = retrieve.Encoder(
    encoder = model.encode,
    key = "id",
    on = ["title", "article"],
    k = 2,
    path = "custom_encoder.pkl"
)
```
