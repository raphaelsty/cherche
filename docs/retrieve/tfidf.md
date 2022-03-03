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
[{'id': 0}, {'id': 2}]
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
...  lowercase=True, min_df=0.1, max_df=0.9, ngram_range=(3, 10), analyzer="char_wb")

>>> retriever = retrieve.TfIdf(
...  key="id", on=["title", "article"], documents=documents, tfidf=tfidf, k=30)

>>> retriever("fr")
[{'id': 0}]
```

## Map keys to documents

```python
>>> retriever += documents
>>> retriever("fr")
[{'id': 0,
  'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris'}]
```
