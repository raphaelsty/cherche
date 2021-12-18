# Encoder

The `retriever.Encoder` model allows the use of framework that encode queries and documents with a single model. It is compatible with the [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) models. The encoder pre-computes document embeddings and uses [Faiss](https://github.com/facebookresearch/faiss) to quickly find the documents most similar to the query embedding.

You can use your own model within `retrieve.Encoder`. This model should have an API similar to the Sentence Transformers models. It should have a method which encodes a list of documents `list[str]` which returns a numpy array with dimensions `(number of documents, embedding size)`. This same method must be able to encode a query (str) and return an embedding of size `(1, embedding dimension)`.

```python
>>> from cherche import retrieve
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {
...        "article": "Paris is the capital and most populous city of France",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    },
...    {
...        "article": "Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    },
...    {
...        "article": "The City of Paris is the centre and seat of government of the region and province of Île-de-France .",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    }
... ]

>>> retriever = retrieve.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    on = ["title", "article"],
...    k = 2,
...    path = "retriever_encoder.pkl"
... )

>>> retriever.add(documents=documents)

>>> retriever("france")
```

```python
[{'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.9025790931437582},
 {'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.8160134832855334}]
```
