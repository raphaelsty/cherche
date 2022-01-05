# Retrieve

Retrievers are models that can filter all documents from a query very quickly. They speed up the neural search pipeline by filtering out the majority of documents that are not relevant. Rankers that are slower will then be able to pull up the most relevant documents based on semantic similarity.

## Retrievers

Here is the list of available retrievers:

- `retrieve.TfIdf`
- `retrieve.BM25L`
- `retrieve.BM25Okapi`
- `retrieve.Elastic`
- `retrieve.Lunr`
- `retrieve.Flash`
- `retrieve.Encoder`

## k and on parameters

The main parameter of retrievers is `on`. This is the field(s) on which the retriever will perform the search. If multiple fields are specified, the retriever will concatenate all fields in the order provided. All the fields defined in `on` must be present in every documents.

The retrievers all have a `k`-parameter during the initialization which allows to select the number of documents to retrieve. The default value is `None`, i.e the retrievers will retrieves all documents that match the query. If you choose a value for k, retriever will only retrieves k top documents that are more likely to match the query.

Retrievers index and store the set of documents with the `add` method.

```python
>>> from cherche import retrieve

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

>>> retriever = retrieve.TfIdf(on="article", k=30)

>>> retriever.add(documents=documents)

>>> retriever("Paris")
```

```python
[{'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris'},
 {'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris'},
 {'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris'}]
```

## Search on multiples fields

It is possible to search on more than one field of the document to find more documents.

```python
>>> retriever = retrieve.TfIdf(on=["title", "article"], k=30)

>>> retriever.add(documents=documents)

>>> retriever("Paris")
```

```python
[{'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris'},
 {'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris'},
 {'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris'}]
```
