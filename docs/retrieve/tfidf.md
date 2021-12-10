# TfIdf

The TfIdf retriever is based on the [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) of sklearn. It computes the dot product between the query tf-idf vector and the documents tf-idf matrix and retrieve highest match.

```python
>>> from cherche import retrieve

>>> documents = [
...    {"document": "Lorem ipsum dolor sit amet"},
...    {"document": " Duis aute irure dolor in reprehenderit"},
... ]

>>> retriever = retrieve.TfIdf(on="document", k=30)

>>> retriever.add(documents=documents)
```

You can also initialise the retriever with a custom [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

```python
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from cherche import retrieve

>>> documents = [
...    {"document": "Lorem ipsum dolor sit amet"},
...    {"document": " Duis aute irure dolor in reprehenderit"},
... ]

>>> tfidf = TfidfVectorizer(lowercase=True, stop_words="english",min_df=2, max_df=0.7)

>>> retriever = retrieve.TfIdf(tfidf=tfidf, on="document", k=30)

>>> retriever.add(documents=documents)
```
