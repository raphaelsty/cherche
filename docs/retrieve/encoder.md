# Encoder

The `retriever.Encoder` model allows the use of framework that encode queries and documents with a single model. It is compatible with the [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) models. The encoder pre-computes document embeddings and uses [Faiss](https://github.com/facebookresearch/faiss) to quickly find the documents most similar to the query embedding.

You can use your own model within `retrieve.Encoder`. This model should have an API similar to the Sentence Transformers models. It should have a method which encodes a list of documents `list[str]` which returns a numpy array with dimensions `(number of documents, embedding size)`. This same method must be able to encode a query (str) and return an embedding of size `(1, embedding dimension)`.

Documents indexed by `retrieve.Encoder` can be updated in mini-batch with the `add` method.
This method takes time because the encoder will pre-compute the document embeddings and store them
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

>>> retriever = retrieve.Encoder(
...    key = "id",
...    on = ["title", "article"],
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    k = 2,
...    path = "all-mpnet-base-v2.pkl"
... )

>>> retriever = retriever.add(documents=documents)

>>> retriever("france")
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
