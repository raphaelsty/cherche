# Retrieve

Retrievers speed up the neural search pipeline by filtering out the majority of documents that are not relevant. Rankers (slower) will then pull up the most relevant documents based on semantic
similarity. The retrievers `retrieve.Elastic`, `retrieve.Meilisearch`, `retrieve.Typesense`, `retrieve.Encoder` and `retrieve.DPR` are the only retrievers in Cherche that are compatible with large corpora. The other retrievers are adapted to small or medium-sized corpora since we will store documents in memory.

`retrieve.Encoder` and `retrieve.DPR` retrievers rely on semantic similarity, unlike the other retrievers, which match exact words.

## Retrievers

Here is the list of available retrievers using Cherche:

- `retrieve.TfIdf`
- `retrieve.BM25L`
- `retrieve.BM25Okapi`
- `retrieve.Elastic`
- `retrieve.Lunr`
- `retrieve.Flash`
- `retrieve.Encoder`
- `retrieve.DPR`
- `retrieve.Fuzz`
- `retrieve.Meilisearch`
- `retrieve.Typesens`

## k and on parameters

The main parameter of retrievers is `on`; it is the field(s) on which the retriever will perform the search. If multiple fields are specified, the retriever will concatenate all fields in the order provided.

The retrievers all have a `k`-parameter which allows selecting the number of documents to retrieve.
The default value is `None`, i.e., the retrievers will retrieve all documents matching the query.
If we choose a value for k, the retriever will only retrieve k top documents that are more likely to match the query.

```python
>>> from cherche import retrieve

>>> documents = [
...    {
...        "id": 0,
...        "article": "Paris is the capital and most populous city of France",
...        "title": "Paris",
...    },
...    {
...        "id": 1,
...        "article": "Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.",
...        "title": "Paris",
...    },
... ]

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], k=30, documents=documents)
```

## Add documents per batch to a retriever

Retrievers store document `keys` to retrieve them later. Some retrievers can index documents keys in mini-batch like `retrieve.Elastic`, `retrieve.Flash` and `retrieve.Encoder`. These retrievers have the `add` method to add documents by batch. The other retrievers do not allow to add documents by batch. Instead, we declare the set of documents when initializing the retriever via the `document` parameter.

|      Retriever     |   Batch   |  Storage  | Corpus size |
|:------------------:|:---------:|:---------:|:-----------:|
|  retrieve.Elastic  |     ✅     | disk     | Large       |
|  retrieve.Meilisearch  |     ✅     | disk | Large       |
|  retrieve.Typesense  |     ✅     | disk   | Large       |  retrieve.Encoder * Milvus |     ✅     | disk   | Large      |
|  retrieve.DPR * Milvus |     ✅     | disk   | Large      |
|  retrieve.Recommend * Milvus |     ✅     | disk | Large      |
|  retrieve.Encoder  |     ✅     | memory   | Medium      |
|  retrieve.DPR  |     ✅     | memory   | Medium      |
|  retrieve.Recommend |     ✅     | memory | Medium      |
|   retrieve.Flash   |     ✅     | memory   | Medium      |
|    retrieve.Fuzz   |     ✅     | memory   | Medium      |
|   retrieve.TfIdf   |     ❌     | memory   | Medium      |
|   retrieve.BM25L   |     ❌     | memory   | Medium      |
| retrieve.BM25Okapi |     ❌     | memory   | Medium      |
|    retrieve.Lunr   |     ❌     | memory   | Medium      |

## Quick start

```python
>>> from cherche import retrieve

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

>>> retriever = retrieve.TfIdf(key="id", on="article", k=30, documents=documents)

>>> retriever("Paris")
[{'id': 0}, {'id': 1}, {'id': 2}]
```

## Matching indexes to documents

It is possible to directly retrieve the content of the documents using the `+` operator between retriever and documents. Documents mapping is helpful if we want to check the retrieved document's content or to accomplish question answering or summarization.

```python
>>> retriever += documents
>>> retriever("Paris")
[{'id': 0,
  'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris'},
 {'id': 1,
  'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris'},
 {'id': 2,
  'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris'}]
```

## Search on multiples fields

Also, we can search with multiple fields to retrieve more documents.

```python
>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k=30)

>>> retriever("Paris")
[{'id': 0}, {'id': 1}, {'id': 2}]
```
