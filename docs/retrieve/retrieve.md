# Retrieve

Retrievers speed up the neural search pipeline by filtering out the majority of documents that are not relevant. Rankers (slower) will then pull up the most relevant documents based on semantic
similarity.

`retrieve.Encoder`, `retrieve.DPR` and `retrieve.Embedding` retrievers rely on semantic similarity, unlike the other retrievers, which match exact words.

## Retrievers

Here is the list of available retrievers using Cherche:

- `retrieve.TfIdf`
- `retrieve.Lunr`
- `retrieve.Flash`
- `retrieve.Fuzz`
- `retrieve.Encoder`
- `retrieve.DPR`
- `retrieve.Embedding`

To use `retrieve.Encoder`, `retrieve.DPR` or `retrieve.Embedding` we will need to install cherche using:

```sh
pip install "cherche[cpu]"
```

If we want to run semantic retrievers on GPU:

```sh
pip install "cherche[gpu]"
```

## Tutorial

The main parameter of retrievers is `on`; it is the field(s) on which the retriever will perform the search. If multiple fields are specified, the retriever will concatenate all fields in the order provided. The `key`
parameter is the name of the field that contain an unique identifier for the document.

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

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents)
```

Calling a retriever with a single query output a list of documents.
```python
retriever("paris", k=30)
[{'id': 0, 'similarity': 0.21638007903488998},
 {'id': 1, 'similarity': 0.13897776006154242}]
```

Calling a retriever with a list of queries output a list of list of documents. If we want to call
retriever on multiples queries, we should opt for the following code:

```python
retriever(["paris", "art", "finance"], k=30)
[[{'id': 0, 'similarity': 0.21638007903488998},  # Query 1
  {'id': 1, 'similarity': 0.13897776006154242}],
 [{'id': 1, 'similarity': 0.03987124117278171}], # Query 2
 [{'id': 1, 'similarity': 0.15208763286878763},  # Query 3
  {'id': 0, 'similarity': 0.02564158475123616}]]
```

## Parameters

|      Retriever     |   Add   |  Semantic  | Batch optmized |
|:------------------:|:---------:|:---------:|:-----------:|
|  retrieve.Encoder   |     ✅     | ✅   | ✅      |
|  retrieve.DPR       |     ✅     | ✅   | ✅      |
|  retrieve.Embedding |     ✅     | ✅   | ✅      |
|   retrieve.Flash    |     ✅     | ❌   | ❌      |
|    retrieve.Fuzz    |     ✅     | ❌   | ❌      |
|   retrieve.TfIdf    |     ❌     | ❌   | ✅      |
|    retrieve.Lunr    |     ❌     | ❌   | ❌      |

- Add: Retriever has a `.add(documents)` method to index new documents along the way.
- Semantic: The Retriever is powered by a language model, enabling semantic similarity-based document retrieval.
- Batch-Optimized: The Retriever is optimized for batch processing, with a batch_size parameter that can be adjusted to handle multiple queries efficiently.

We can call retrievers with a k-parameter, which enables the selection of the number of documents to be retrieved. By default, the value of k is set to None, meaning that the retrievers will retrieve all documents that match the query. However, if a specific value for k is chosen, the retriever will only retrieve the top k documents that are most likely to match the query.

```python
>>> retriever(["paris", "art"], k=3)
[[{'id': 0, 'similarity': 0.21638007903488998},
  {'id': 1, 'similarity': 0.13897776006154242}],
 [{'id': 1, 'similarity': 0.03987124117278171}]]
```

## Matching indexes to documents

It is possible to directly retrieve the content of the documents using the `+` operator between retriever and documents. Documents mapping is helpful if we want to plug our retriever on a `rank.CrossEncoder`.

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
