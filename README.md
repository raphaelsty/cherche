# Cherche

Cherche (search in French) allows you to create a simple neural search pipeline using pre-trained language models. Cherche retrieves the right documents from a natural language query. 

## Installation

```sh
pip install cherche
```

## Documents

Cherche models allows to index a set of documents. These documents shall be formatted as a Python dictionary list. The names of the dictionary keys do not matter. You will select the dictionary field of your choice to perform the search. Here is a list of documents suitable for neural search with Cherche:

```python
from cherche import data
documents = data.load_towns()

documents[:3]
[{'article': 'Paris (French pronunciation: \u200b[paʁi] (listen)) is the '
             'capital and most populous city of France, with an estimated '
             'population of 2,175,601 residents as of 2018, in an area of more '
             'than 105 square kilometres (41 square miles).',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris'},
 {'article': "Since the 17th century, Paris has been one of Europe's major "
             'centres of finance, diplomacy, commerce, fashion, gastronomy, '
             'science, and arts.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris'},
 {'article': 'The City of Paris is the centre and seat of government of the '
             'region and province of Île-de-France, or Paris Region, which has '
             'an estimated population of 12,174,880, or about 18 percent of '
             'the population of France as of 2017.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris'}]
```

## Retrieve 🔎

Cherche offers different retrievers for information retrieval. A retriever is a very fast model that allows to filter the most relevant documents for a query. 

- retrieve.ElasticSearch
- retrieve.TfIdf
- retrieve.BM25
- retrieve.Flash

```python
>>> from cherche import data, retrieve

# Load the list of dicts
>>> documents = data.load_towns() 

# Initialize retriever
>>> retriever = retrieve.TfIdf(on="article", k=3)

# Index documents
>>> retriever.add(documents=documents)

# Most relevant documents using TF-IDF
>>> retriever("capital of france")
[{'title': 'Toulouse',
  'url': 'https://en.wikipedia.org/wiki/Toulouse',
  'article': 'Founded by the Romans, the city was the capital of the Visigothic Kingdom in the 5th century and the capital of the province of Languedoc in the Late Middle Ages and early modern period (provinces were abolished during the French Revolution), making it the unofficial capital of the cultural region of Occitania (Southern France).'},
 {'title': 'Toulouse',
  'url': 'https://en.wikipedia.org/wiki/Toulouse',
  'article': 'It is now the capital of the Occitanie region, the second largest region in Metropolitan France.'},
 {'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France, or Paris Region, which has an estimated population of 12,174,880, or about 18 percent of the population of France as of 2017.'}]
```

## Rank 🤖

Cherche proposes different models to re-rank the documents out of the retriever. Rankers are based on semantic similarity between the query and the documents proposed by the retriever to establish a new order. We can select the retriever and the ranker of our choice and combine them to improve the search.

- rank.Encoder
- rank.DPR
- rank.ZeroShot

```python
>>> from cherche import data, retrieve, rank
>>> from sentence_transformers import SentenceTransformer

# Load the list of dicts
>>> documents = data.load_towns() 

# Initialize retriever
>>> retriever = retrieve.TfIdf(on="article", k=30)

# Initialize the ranker
>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    on = "title",
...    k = 2,
...    path = "encoder.pkl"
... )

# Intialize the pipeline
>>> search = retriever + ranker

# Index documents
>>> search = search.add(documents)

>>> search("capital of france")
[{'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France, or Paris Region, which has an estimated population of 12,174,880, or about 18 percent of the population of France as of 2017.',
  'cosine_distance': 0.42192113399505615},
 {'title': 'Toulouse',
  'url': 'https://en.wikipedia.org/wiki/Toulouse',
  'article': 'Founded by the Romans, the city was the capital of the Visigothic Kingdom in the 5th century and the capital of the province of Languedoc in the Late Middle Ages and early modern period (provinces were abolished during the French Revolution), making it the unofficial capital of the cultural region of Occitania (Southern France).',
  'cosine_distance': 0.48012280464172363}]
```

## Question Answering

```python
>>> from cherche import data, retrieve, rank, qa
>>> from sentence_transformers import SentenceTransformer
>>> from transformers import pipeline

# Load the list of dicts
>>> documents = data.load_towns() 

# Initialize retriever
>>> retriever = retrieve.TfIdf(on="article", k=3)

# Initialize the ranker
>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    on = "title",
...    k = 2,
...    path = "encoder.pkl"
... )

# Intialize the question answering model
>>> question_answer = qa.QA(
...     model = pipeline("question-answering", model = "deepset/roberta-base-squad2", tokenizer = "deepset/roberta-base-squad2"),
...     on = "title",
...  )

# Intialize the pipeline
>>> search = retriever + ranker + question_answer

# Index documents
>>> search = search.add(documents)

>>> search("What is the capital of france?")
```

We can use a simple `retriever` `question answering` architecture or add a ranker to improve the results of the `retriever`, `ranker`, `question answering` model as in the example above.

Retrievers, Rankers and Readers pipelines.
