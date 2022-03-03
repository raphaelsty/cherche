<div align="center">
  <h1>Cherche</h1>
  <p>Neural search</p>
</div>
<br>

<div align="center">
  <!-- Documentation -->
  <a href="https://raphaelsty.github.io/cherche/"><img src="https://img.shields.io/website?label=docs&style=flat-square&url=https%3A%2F%2Fraphaelsty.github.io/cherche/%2F" alt="documentation"></a>
  <!-- Demo -->
  <a href="https://huggingface.co/spaces/raphaelsty/games"><img src="https://img.shields.io/badge/demo-running-blueviolet?style=flat-square" alt="Demo"></a>  
  <!-- License -->
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="license"></a>
</div>
<br>

Cherche allows creating a neural search pipeline using retrievers and pre-trained language models as rankers. We dedicated Cherche to small to medium-sized corpora. Cherche's main strength is its ability to build diverse and end-to-end pipelines.

![Alt text](img/doc.png)

## Installation 🤖

```sh
pip install cherche
```

To install the development version:

```sh
pip install git+https://github.com/raphaelsty/cherche
```

## QuickStart 💨

### Documents 📑

Cherche allows to find the right document within a list of JSON. Here is an example of a corpus.

```python
from cherche import data

documents = data.load_towns()

documents[:3]
[{'id': 0,
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'article': 'Paris is the capital and most populous city of France.'},
 {'id': 1,
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'article': "Since the 17th century, Paris has been one of Europe's major centres of science, and arts."},
 {'id': 2,
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France.'
  }]
```

### Retriever ranker 🔍

Here is an example of a neural search pipeline composed of a TfIdf that quickly retrieves documents followed by a ranking model that sorts the documents at the output of the retriever based on the semantic similarity between the query and the documents.

```python
from cherche import data, retrieve, rank
from sentence_transformers import SentenceTransformer

# List of dicts
documents = data.load_towns() 

# Retrieve on fields title and article
retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k=30)

# Rank on fields title and article
ranker = rank.Encoder(
    key = "id",
    on = ["title", "article"],
    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    k = 3,
    path = "encoder.pkl"
)

# Pipeline creation
search = retriever + ranker

search.add(documents=documents)

search("Bordeaux")
[{'id': 57, 'similarity': 0.69513476},
 {'id': 63, 'similarity': 0.6214991},
 {'id': 65, 'similarity': 0.61809057}]
```

Map the index to documents

```python
search += documents 
search("Bordeaux")
[{'id': 57,
  'title': 'Bordeaux',
  'url': 'https://en.wikipedia.org/wiki/Bordeaux',
  'article': 'Bordeaux ( bor-DOH, French: [bɔʁdo] (listen); Gascon Occitan: Bordèu [buɾˈðɛw]) is a port city on the river Garonne in the Gironde department, Southwestern France.',
  'similarity': 0.69513476},
 {'id': 63,
  'title': 'Bordeaux',
  'url': 'https://en.wikipedia.org/wiki/Bordeaux',
  'article': 'The term "Bordelais" may also refer to the city and its surrounding region.',
  'similarity': 0.6214991},
 {'id': 65,
  'title': 'Bordeaux',
  'url': 'https://en.wikipedia.org/wiki/Bordeaux',
  'article': "Bordeaux is a world capital of wine, with its castles and vineyards of the Bordeaux region that stand on the hillsides of the Gironde and is home to the world's main wine fair, Vinexpo.",
  'similarity': 0.61809057}]
```
