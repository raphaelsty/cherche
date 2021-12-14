# Flash

Flash is a wrapper of [FlashText](https://github.com/vi3k6i5/flashtext). This awesome algorithm can
retrieve keywords in documents faster than anything else. FlashText is explained in the article
[Replace or Retrieve Keywords In Documents At Scale](https://arxiv.org/pdf/1711.00046.pdf).

You can use Flash to find documents from a field that contains a keyword or a list of keywords.
Flash will find documents that contain the keyword or keywords specified in the query.

```python
>>> from cherche import retrieve

>>> documents = [
...    {
...        "article": "Paris is the capital and most populous city of France",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris",
...        "tags": ["paris", "france", "capital"]
...    },
...    {
...        "article": "Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris",
...        "tags": ["paris", "france", "capital", "fashion"]
...    },
...    {
...        "article": "The City of Paris is the centre and seat of government of the region and province of Île-de-France .",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris",
...        "tags": "paris"
...    }
... ]

>>> retriever = retrieve.Flash(on="tags")

>>> retriever.add(documents=documents)

>>> retriever("fashion")
```

```python
[{'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'tags': ['paris', 'france', 'capital', 'fashion']}]
```
