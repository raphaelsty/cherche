<div align="center">
  <h1>Cherche</h1>
  <p>Neural search</p>
</div>
<br>

<div align="center">
  <!-- Documentation -->
  <a href="https://raphaelsty.github.io/cherche/"><img src="https://img.shields.io/website?label=docs&style=flat-square&url=https%3A%2F%2Fraphaelsty.github.io/cherche/%2F" alt="documentation"></a>
  <!-- License -->
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="license">
  </a>
</div>
<br>

Cherche (search in French) allows you to create a neural search pipeline using retrievers and pre-trained language models as rankers. Cherche is meant to be used with small to medium sized corpora. Cherche's main strength is its ability to build diverse and end-to-end pipelines.

![Alt text](docs/img/explain.png)

## Installation 🤖

```sh
pip install cherche
```

To install the development version:

```sh
pip install git+https://github.com/raphaelsty/cherche
```

## [Documentation](https://raphaelsty.github.io/cherche/) 📜

Documentation is available [here](https://raphaelsty.github.io/cherche/). It provides details
about retrievers, rankers, pipelines, question answering, summarization, and examples.

## QuickStart 💨

### Documents 📑

Cherche allows findings the right document within a list of objects. Here is an example of a corpus.

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

Here is an example of a neural search pipeline composed of a TfIdf that quickly retrieves documents, followed by a ranking model. The ranking model sorts the documents produced by the retriever based on the semantic similarity between the query and the documents.

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

Map the index to the documents to access their contents.

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

## Retrieve 👻

Cherche provides different retrievers that filter input documents based on a query.

- retrieve.Elastic
- retrieve.TfIdf
- retrieve.Lunr
- retrieve.BM25Okapi
- retrieve.BM25L
- retrieve.Flash
- retrieve.Encoder

## Rank 🤗

Cherche rankers are compatible with [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) models, [Hugging Face sentence similarity](https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=downloads) models, [Hugging Face zero shot classification](https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=downloads) models, and of course with your own models.

## Summarization and question answering

Cherche provides modules dedicated to summarization and question answering. These modules are compatible with Hugging Face's pre-trained models and can be fully integrated into neural search pipelines.

## Hugging Face spaces

You can find a running demo of Cherche [here](https://huggingface.co/spaces/raphaelsty/games).

## Acknowledgements 👏

The BM25 models available in Cherche are wrappers around [rank_bm25](https://github.com/dorianbrown/rank_bm25). Elastic retriever is a wrapper around [Python Elasticsearch Client](https://elasticsearch-py.readthedocs.io/en/v7.15.2/). TfIdf retriever is a wrapper around [scikit-learn's TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). Lunr retriever is a wrapper around [Lunr.py](https://github.com/yeraydiazdiaz/lunr.py). Flash retriever is a wrapper around [FlashText](https://github.com/vi3k6i5/flashtext). DPR and Encode rankers are wrappers dedicated to the use of the pre-trained models of [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) in a neural search pipeline. ZeroShot ranker is a wrapper dedicated to the use of the zero-shot sequence classifiers of [Hugging Face](https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=downloads) in a neural search pipeline.

## See also 👀

Cherche is a minimalist solution and meets a need for modularity. Cherche is the way to go if you start with a list of documents as JSON with multiple fields to search on and want to create pipelines. Also ,Cherche is well suited for middle sized corpora.

Do not hesitate to look at Haystack, Jina, or TxtAi which offer very advanced solutions for neural search and are great.

- [Haystack](https://github.com/deepset-ai/haystack)
- [Jina](https://github.com/jina-ai/jina)
- [txtai](https://github.com/neuml/txtai)

## Dev Team 💾

The Cherche dev team is made up of [Raphaël Sourty](https://github.com/raphaelsty) and [François-Paul Servant](https://github.com/fpservant) 🥳
