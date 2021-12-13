# TfIdf

The TfIdf retriever is based on the [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) of sklearn. It computes the dot product between the query tf-idf vector and the documents tf-idf matrix and retrieve highest match.

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
TfIdf retriever
    on: article
    documents: 3
```

You can also initialise the retriever with a custom [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

```python
>>> from cherche import retrieve
>>> from sklearn.feature_extraction.text import TfidfVectorizer

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

>>> tfidf = TfidfVectorizer(lowercase=True, min_df=1, max_df=0.5, ngram_range=(4, 10), analyzer="char_wb")

>>> retriever = retrieve.TfIdf(tfidf=tfidf, on="article", k=30)

>>> retriever.add(documents=documents)
TfIdf retriever
    on: article
    documents: 3
```
