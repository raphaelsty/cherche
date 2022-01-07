# Flash

Flash is a wrapper of [FlashText](https://github.com/vi3k6i5/flashtext). This awesome algorithm can
retrieve keywords in documents faster than anything else. FlashText is explained in the article
[Replace or Retrieve Keywords In Documents At Scale](https://arxiv.org/pdf/1711.00046.pdf).

You can use Flash to find documents from a field that contains a keyword or a list of keywords.
Flash will find documents that contain the keyword or keywords specified in the query.

Flash can be updated with new documents using mini-batch via the `add` method.

```python
>>> from cherche import retrieve

>>> documents = [
...    {
...        "id": 0,
...        "article": "Paris is the capital and most populous city of France",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris",
...        "tags": ["paris", "france", "capital"]
...    },
...    {
...        "id": 1,
...        "article": "Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris",
...        "tags": ["paris", "france", "capital", "fashion"]
...    },
...    {
...        "id": 2,
...        "article": "The City of Paris is the centre and seat of government of the region and province of Île-de-France .",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris",
...        "tags": "paris"
...    }
... ]

>>> retriever = retrieve.Flash(key="id", on="tags")

>>> retriever.add(documents=documents)

>>> retriever("fashion")
[{'id': 1}]
```

## Map keys to documents

```python
>>> retriever += documents
>>> retriever("fashion")
[{'id': 1,
  'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'tags': ['paris', 'france', 'capital', 'fashion']}]
```
