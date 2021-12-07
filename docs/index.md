# Cherche

Cherche (search in French) allows you to create a simple neural search pipeline using pre-trained language models. Cherche find the right documents from a list of JSON using the keys of your choice. 

![Alt text](img/explain.png)

## Installation 🛠

```sh
pip install cherche
```

## Documents 📜

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
- retrieve.BM25Okapi
- retrieve.BM25L
- retrieve.BM25Plus
- retrieve.Flash

```python
>>> from cherche import data, retrieve

# Load the list of dicts
>>> documents = data.load_towns() 

# Initialize retriever
>>> retriever = retrieve.TfIdf(on="article", k=3) # on correspond au champ des dictionnaires sur lesquels on va effectuer la recherche.

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

## Retrieve + Rank 🤖

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
...    on = "article",
...    k = 3,
...    path = "encoder.pkl"
... )

# Intialize the pipeline
>>> search = retriever + ranker

# Index documents
>>> search = search.add(documents)

>>> search("capital of france")
[{'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'article': 'Paris (French pronunciation: \u200b[paʁi] (listen)) is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018, in an area of more than 105 square kilometres (41 square miles).',
  'cosine_distance': 0.3019077181816101},
 {'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France, or Paris Region, which has an estimated population of 12,174,880, or about 18 percent of the population of France as of 2017.',
  'cosine_distance': 0.3593599200248718},
 {'title': 'Toulouse',
  'url': 'https://en.wikipedia.org/wiki/Toulouse',
  'article': 'It is now the capital of the Occitanie region, the second largest region in Metropolitan France.',
  'cosine_distance': 0.44300907850265503}]
```

As you can see, for the same query, the ranker manages to improve the results of the retriever.

## Retrieve + Rank + Question Answering 😶

Cherche provide a solution to connect extractive question answering models to retriever and ranker.
The retriever, ranker approach speed up the question answering process.

```python
>>> from cherche import data, retrieve, rank, qa
>>> from sentence_transformers import SentenceTransformer
>>> from transformers import pipeline

# Load the list of dicts
>>> documents = data.load_towns() 

# Initialize retriever
>>> retriever = retrieve.TfIdf(on="article", k=30)

# Initialize the ranker
>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    on = "article",
...    k = 3,
...    path = "encoder.pkl"
... )

# Intialize the question answering model
>>> question_answer = qa.QA(
...     model = pipeline("question-answering", model = "deepset/roberta-base-squad2", tokenizer = "deepset/roberta-base-squad2"),
...     on = "article",
... )

# Intialize the pipeline
>>> search = retriever + ranker + question_answer

# Index documents
>>> search = search.add(documents)

>>> search("What is the capital of france?")
[{'qa_score': 0.6114194989204407,
  'start': 0,
  'end': 5,
  'answer': 'Paris',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'article': 'Paris (French pronunciation: \u200b[paʁi] (listen)) is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018, in an area of more than 105 square kilometres (41 square miles).',
  'cosine_distance': 0.31821733713150024},
 {'qa_score': 0.20747852325439453,
  'start': 12,
  'end': 17,
  'answer': 'Paris',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France, or Paris Region, which has an estimated population of 12,174,880, or about 18 percent of the population of France as of 2017.',
  'cosine_distance': 0.399497926235199},
 {'qa_score': 0.07703403383493423,
  'start': 165,
  'end': 169,
  'answer': 'Nice',
  'title': 'Toulouse',
  'url': 'https://en.wikipedia.org/wiki/Toulouse',
  'article': 'It is the fourth-largest commune in France, with 479,553 inhabitants within its municipal boundaries (as of January 2017), after Paris, Marseille and Lyon, ahead of Nice; it has a population of 1,360,829 within its wider metropolitan area (also as of January 2017).',
  'cosine_distance': 0.4340791702270508}]
```

It is possible to use the question answering module with a simple retriever or with a retriever and a ranker easily.

## Retrieve + Rank + Summary 👾

Cherche provides a summary of relevant documents for a query using HuggingFace's pre-trained models. The retriever and the rankers allows to reduce speed up
the summarization process.

```python
>>> from cherche import data, retrieve, rank, summary
>>> from sentence_transformers import SentenceTransformer
>>> from transformers import pipeline

# Load the list of dicts
>>> documents = data.load_towns()

# Initialize retriever
>>> retriever = retrieve.TfIdf(on="article", k=30)

# Initialize the ranker
>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    on = "article",
...    k = 3,
...    path = "encoder.pkl"
... )

# Intialize the summarizer
>>> summarizer = summary.Summary(
...    model = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", tokenizer="sshleifer/distilbart-cnn-6-6", framework="pt"),
...    on = "article",
... )

# Intialize the pipeline
>>> search = retriever + ranker + summarizer

# Index documents
>>> search = search.add(documents)

>>> search("What is the capital of france?")
' The City of Paris is the centre and seat of government of the region and province of Île-de-France. It is'
```

The choice of the pre-trained model (ranker, question answering and summary) is crucial to obtain meaningful answers.

## References

Cherche is a minimalist solution and meets a need for modularity. Do not hesitate to look at Haystack,
Jina and TxtAi which offer very advanced solutions for neural search.

- [Haystack](https://github.com/deepset-ai/haystack)
- [Jina](https://github.com/jina-ai/jina)
- [txtai](https://github.com/neuml/txtai)

The BM25 models available in Cherche are a wrapper of [rank_bm25](https://github.com/dorianbrown/rank_bm25). Elastic retriever is a wrapper of [Python Elasticsearch Client](https://elasticsearch-py.readthedocs.io/en/v7.15.2/). TfIdf retriever is a wrapper of [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). TfidfVectorizer. Flash retriever is a wrapper of [FlashText](https://github.com/vi3k6i5/flashtext). DPR and Encode rankers are wrappers dedicated to the use of the pre-trained models of [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) in a neural search pipeline. ZeroShot ranker is a wrapper dedicated to the use of the zero-shot sequence classifiers of [Hugging Face](https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=downloads) in a neural search pipeline.
