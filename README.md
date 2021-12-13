# Cherche

<html><a href="https://raphaelsty.github.io/cherche/"><img src="https://img.shields.io/website?label=docs&style=flat-square&url=https%3A%2F%2Fraphaelsty.github.io/cherche/%2F" alt="documentation"></a></html>

Cherche (search in French) allows you to create a simple neural search pipeline using retrievers and pre-trained language models as rankers. Cherche is dedicated to corpus of small to middle size.

![Alt text](docs/img/explain.png)

## Installation 🤖

```sh
pip install git+https://github.com/raphaelsty/cherche
```

## [Documentation](https://raphaelsty.github.io/cherche/) 📜

Documentation is available [here](https://raphaelsty.github.io/cherche/). It provides details
about retrievers, rankers, pipelines, question answering, summarization and examples.

## QuickStart 🤯

### Documents 📑

Cherche allows to find the right document within a list of JSON. Here is an example of a corpus.

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

### Retriever ranker 🔍

Here is an example of a neural search pipeline composed of a TfIdf that quickly retrieves documents followed by a ranking model that sorts the documents at the output of the retriever based on the semantic similarity between the query and the documents.

```python
>>> from cherche import data, retrieve, rank
>>> from sentence_transformers import SentenceTransformer

# List of dicts
>>> documents = data.load_towns() 

# Retriever
>>> retriever = retrieve.TfIdf(on="article", k=30) # on correspond au champ des dictionnaires sur lesquels on va effectuer la recherche.

# Ranker
>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    on = "article",
...    k = 3,
...    path = "encoder.pkl"
... )

>>> search = retriever + ranker

# Index documents
>>> search.add(documents=documents)
TfIdf retriever
  on: article
  documents: 105
Encoder ranker
  on: article
  k: 3
  similarity: cosine
  embeddings stored at: encoder.pkl

# Relevant documents
>>> search("capital of france")
[{'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'article': 'Paris (French pronunciation: \u200b[paʁi] (listen)) is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018, in an area of more than 105 square kilometres (41 square miles).',
  'similarity': 0.69809234},
 {'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France, or Paris Region, which has an estimated population of 12,174,880, or about 18 percent of the population of France as of 2017.',
  'similarity': 0.64064014},
 {'title': 'Toulouse',
  'url': 'https://en.wikipedia.org/wiki/Toulouse',
  'article': 'It is now the capital of the Occitanie region, the second largest region in Metropolitan France.',
  'similarity': 0.556991}]
```

## Retrieve 👻

Cherche provides different retrievers that filter input documents based on a query.

- retrieve.ElasticSearch
- retrieve.TfIdf
- retrieve.BM25Okapi
- retrieve.BM25L
- retrieve.Flash

## Rankers 🤗

Cherche rankers are compatible with [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) models, [Hugging Face sentence similarity](https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=downloads) models, [Hugging Face zero shot classification](https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=downloads) models and of course with your own models.

## Acknowledgement

Cherche is a minimalist solution and meets a need for modularity. Do not hesitate to look at Haystack,
Jina and TxtAi which offer very advanced solutions for neural search.

- [Haystack](https://github.com/deepset-ai/haystack)
- [Jina](https://github.com/jina-ai/jina)
- [txtai](https://github.com/neuml/txtai)

The BM25 models available in Cherche are a wrapper of [rank_bm25](https://github.com/dorianbrown/rank_bm25). Elastic retriever is a wrapper of [Python Elasticsearch Client](https://elasticsearch-py.readthedocs.io/en/v7.15.2/). TfIdf retriever is a wrapper of [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). TfidfVectorizer. Flash retriever is a wrapper of [FlashText](https://github.com/vi3k6i5/flashtext). DPR and Encode rankers are wrappers dedicated to the use of the pre-trained models of [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) in a neural search pipeline. ZeroShot ranker is a wrapper dedicated to the use of the zero-shot sequence classifiers of [Hugging Face](https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=downloads) in a neural search pipeline.
