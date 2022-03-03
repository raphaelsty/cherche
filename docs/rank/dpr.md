# rank.DPR

`rank.DPR` is dedicated to the [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) models
which aims to train two distinct neural networks, one that encodes the query and the other one that
encodes the documents.

## Pre-compute

The `rank.DPR` can pre-compute the set of document embeddings to speed up search in the production environment. To store the embeddings of the ranker, it is necessary to specify the `path` parameter. `rank.Encoder` will load the embeddings from the `path` file when calling the `add` function. A GPU will significantly reduce pre-computing time dedicated to document embeddings.

## Quick Start

```python
>>> from cherche import retrieve, rank
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

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k=30)

>>> ranker = rank.DPR(
...    key = "id",
...    on = ["title", "article"],
...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
...    k = 5,
...    path = "dpr.pkl"
... )

>>> search = retriever + ranker
>>> search.add(documents)
>>> search("France")
[{'id': 0, 'similarity': 72.57037}, {'id': 2, 'similarity': 70.165115}]
```

## Pre-trained DPR

|               Question Encoder               |             Document encoder            |
|:--------------------------------------------:|:---------------------------------------:|
|  facebook-dpr-question_encoder-multiset-base |  facebook-dpr-ctx_encoder-multiset-base |
| facebook-dpr-question_encoder-single-nq-base | facebook-dpr-ctx_encoder-single-nq-base |

## Map index to documents

We can map the documents to the ids retrieved by the pipeline.

```python
>>> search += documents
>>> search(q="France")
[{'id': 0,
  'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 72.57037},
 {'id': 2,
  'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 70.165115}]
```

## Custom DPR

We can train our DPR using the tool of our choice and use it with Cherche. Our DPR should have an API similar to the DPR model of Sentence Transformers models. It should be a function that encodes a list of strings to return a NumPy array with dimensions `(number of documents, embedding size)`. This same function must encode a single string to return an embedding of size `(embedding dimension, )`.

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

retriever = rank.DPR(
    encoder = model.documents,
    query_encoder = model.query,
    key = "id",
    on = ["title", "article"],
    k = 2,
    path = "custom_dpr.pkl"
)
```
