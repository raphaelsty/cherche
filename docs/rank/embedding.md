# rank.Embedding

The `rank.Embedding` model utilizes pre-computed embeddings to re-rank documents within the output of the retriever. If you have a custom model that produces its own embeddings and want to re-rank documents accordingly, `rank.Embedding` is the ideal tool for the job.

## Tutorial

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

# Let's use a custom encoder and create our documents embeddings of shape (n_documents, dim_embeddings)
>>> encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
>>> embeddings_documents = encoder.encode([
...    document["article"] for document in documents
... ])

>>> queries = ["paris", "art", "fashion"]

# Queries embeddings of shape (n_queries, dim_embeddings)
>>> embeddings_queries = encoder.encode(queries)

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents)

>>> ranker = rank.Embedding(
...    key = "id",
...    normalize = True,
... )

>>> ranker = ranker.add(
...    documents=documents,
...    embeddings_documents=embeddings_documents,
... )

>>> match = retriever(queries, k=100)

# Re-rank output of retriever
>>> ranker(q=embeddings_queries, documents=match, k=30)
[[{'id': 0, 'similarity': 0.6560695}, # Query 1
  {'id': 1, 'similarity': 0.58203197},
  {'id': 2, 'similarity': 0.5283624}],
 [{'id': 1, 'similarity': 0.1115652}], # Query 2
 [{'id': 1, 'similarity': 0.2555524}, {'id': 2, 'similarity': 0.06398084}]] # Query 3
```

## Map index to documents

We can map the documents to the ids retrieved by the pipeline.

```python
>>> ranker += documents
>>> match = retriever(queries, k=100)
>>> ranker(q=embeddings_queries, documents=match, k=30)
[[{'id': 0,
   'article': 'Paris is the capital and most populous city of France',
   'title': 'Paris',
   'url': 'https://en.wikipedia.org/wiki/Paris',
   'similarity': 0.6560695},
  {'id': 1,
   'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
   'title': 'Paris',
   'url': 'https://en.wikipedia.org/wiki/Paris',
   'similarity': 0.58203197},
  {'id': 2,
   'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
   'title': 'Paris',
   'url': 'https://en.wikipedia.org/wiki/Paris',
   'similarity': 0.5283624}],
 [{'id': 1,
   'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
   'title': 'Paris',
   'url': 'https://en.wikipedia.org/wiki/Paris',
   'similarity': 0.1115652}],
 [{'id': 1,
   'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
   'title': 'Paris',
   'url': 'https://en.wikipedia.org/wiki/Paris',
   'similarity': 0.2555524},
  {'id': 2,
   'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
   'title': 'Paris',
   'url': 'https://en.wikipedia.org/wiki/Paris',
   'similarity': 0.06398084}]]
```
