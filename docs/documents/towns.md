# Towns

Cherche provides a dummy dataset made of sentences from Wikipedia that describes towns such as Toulouse, Paris, Bordeaux and Lyon. This dataset is intended to easily test Cherche. It contains ~200 documents.

```python
>>> from cherche import data
>>> documents = data.load_towns()
>>> documents[:3]
```

```python
[
    {
        "id": 0,
        "title": "Paris",
        "url": "https://en.wikipedia.org/wiki/Paris",
        "article": "Paris (French pronunciation: \u200b[paʁi] (listen)) is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018, in an area of more than 105 square kilometres (41 square miles).",
    },
    {
        "id": 1,
        "title": "Paris",
        "url": "https://en.wikipedia.org/wiki/Paris",
        "article": "Since the 17th century, Paris has been one of Europe's major centres of finance, diplomacy, commerce, fashion, gastronomy, science, and arts.",
    },
    {
        "id": 2,
        "title": "Paris",
        "url": "https://en.wikipedia.org/wiki/Paris",
        "article": "The City of Paris is the centre and seat of government of the region and province of Île-de-France, or Paris Region, which has an estimated population of 12,174,880, or about 18 percent of the population of France as of 2017.",
    },
]
```
