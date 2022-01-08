# Retrieve

Retrievers speed up the neural search pipeline by filtering out the majority of documents that are
not relevant. Rankers (slower) will then be able to pull up the most relevant documents based on semantic
similarity. The retrievers `retrieve.Elastic` and `retrieve.encoder` are the only retrievers in
Cherche that are compatible with large corpora. The other retrievers are adapted to small or
medium size corpora.

## Retrievers

Here is the list of available retrievers using Cherche:

- `retrieve.TfIdf`
- `retrieve.BM25L`
- `retrieve.BM25Okapi`
- `retrieve.Elastic`
- `retrieve.Lunr`
- `retrieve.Flash`
- `retrieve.Encoder`

## k and on parameters

The main parameter of retrievers is `on`. This is the field(s) on which the retriever will perform
the search. If multiple fields are specified, the retriever will concatenate all fields in the
order provided. All the fields defined in `on` must be present in every documents.

The retrievers all have a `k`-parameter which allows to select the number of documents to retrieve.
The default value is `None`, i.e the retrievers will retrieves all documents that match the query.
If you choose a value for k, retriever will only retrieves k top documents that are more likely to 
match the query.

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

Retrievers store document `keys` to retrieve them later. Some retrievers can index documents keys
in mini-batch like `retrieve.Elastic, retrieve.Flash and retrieve.Encoder`. These retrievers have
the `add` method to add documents by batch. The other retrievers do not allow to add documents by
batch. The set of documents must be declared at the initialization of the retriever via the
`document` parameter.

|      Retriever     |   Batch   |
|:------------------:|:---------:|
|  retrieve.Elastic  |     ✅     |
|   retrieve.Flash   |     ✅     |
|  retrieve.Encoder  |     ✅     |
|   retrieve.TfIdf   |     ❌     |
|   retrieve.BM25L   |     ❌     |
| retrieve.BM25Okapi |     ❌     |
|    retrieve.Lunr   |     ❌     |

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

It is possible to directly retrieve the content of the documents using the `+` operator between
retriever and documents. This is useful if you want to see the results of your searches directly.

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

Also we can search on more than one single field to retrieve more documents.

```python
>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k=30)

>>> retriever("Paris")
[{'id': 0}, {'id': 1}, {'id': 2}]
```
