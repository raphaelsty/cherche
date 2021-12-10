# Retrieve

Retrievers are models that can filter all documents from a query very quickly. They speed up the neural search pipeline by filtering out the majority of documents that are not relevant. Rankers that are slower will then be able to pull up the most relevant documents based on semantic similarity.

Here is the list of available retrievers:

- `retrieve.TfIdf`
- `retrieve.BM25L`
- `retrieve.BM25Okapi`
- `retrieve.BM25Plus`
- `retrieve.Elastic`
- `retrieve.Flash`

The retrievers all have a k-parameter during the initialization which allows to select the number of documents to retrieve. The default value is `None`, i.e the retrievers will retrieves all documents that match the query. If you choose a value for k, retriever will only retrieves k top documents that are more likely to match the query.

Retrievers index and store the set of documents with the `add` method.

```python
>>> from cherche import retrieve

>>> documents = [
...    {"document": "Lorem ipsum dolor sit amet"},
...    {"document": " Duis aute irure dolor in reprehenderit"},
... ]

retriever = retrieve.TfIdf(on="document", k=30)

retriever.add(documents=documents)
```
