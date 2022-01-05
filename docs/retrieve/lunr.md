# Lunr

`retrieve.Lunr` is a wrapper of [Lunr.py](https://github.com/yeraydiazdiaz/lunr.py). It is a
powerful and practical solution for searching a corpus of documents without having to use a
retriever such as Elasticsearch when it's not needed. Lunr stores an inverted index in memory.

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

>>> retriever = retrieve.Lunr(on=["title", "article"], k=30)

>>> retriever.add(documents=documents)

>>> retriever("france")
```

```python
[{'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris'},
 {'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris'}]
```
