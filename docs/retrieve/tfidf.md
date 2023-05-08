# TfIdf

Our TF-IDF retriever relies on the [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) of Sklearn. It computes the dot product between the query TF-IDF vector and the documents TF-IDF matrix and retrieves the highest match. TfIdf retriever stores a sparse matrix and an index that links the rows of the matrix to document identifiers.

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

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k=30)

>>> retriever("france")
[{'id': 0, 'similarity': 0.15137222675009282},
 {'id': 2, 'similarity': 0.10831402366399025},
 {'id': 1, 'similarity': 0.02505818772920329}]
```

We can also initialize the retriever with a custom [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

```python
>>> from cherche import retrieve
>>> from sklearn.feature_extraction.text import TfidfVectorizer

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

>>> tfidf = TfidfVectorizer(
...  lowercase=True, ngram_range=(3, 7), analyzer="char_wb")

>>> retriever = retrieve.TfIdf(
...  key="id", on=["title", "article"], documents=documents, tfidf=tfidf)

>>> retriever("fra", k=3)
[{'id': 0, 'similarity': 0.15055477454160002},
 {'id': 2, 'similarity': 0.022883459495904895}]
```

## Batch retrieval

If we have several queries for which we want to retrieve the top k documents then we can
pass a list of queries to the retriever. This is much faster for multiple queries. In batch-mode,
retriever returns a list of list of documents instead of a list of documents.

```python
>>> retriever(["fra", "arts", "capital"], k=3)
[[{'id': 0, 'similarity': 0.051000705070125066}, # Match query 1
  {'id': 2, 'similarity': 0.03415513704304113}],
 [{'id': 1, 'similarity': 0.07021399356970497}], # Match query 2
 [{'id': 0, 'similarity': 0.25972148184421534}]] # Match query 3
```

## Map keys to documents

We can map documents to retrieved keys.

```python
>>> retriever += documents
>>> retriever("fra")
[{'id': 0,
  'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.15055477454160002},
 {'id': 2,
  'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.022883459495904895}]
```
