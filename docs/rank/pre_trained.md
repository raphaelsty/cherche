# Pre-trained models

Ranker are compatible with the `Sentence Bert` models of [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) via the `rank.Encoder` model. The `rank.DPR` template is compatible with the `rank.DPR` templates of SentenceBert. The `rank.ZeroShot` model is compatible with the `zero-shot-classification` models of [Hugging Face](https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=downloads).

## rank.Encoder

Cherche's Encoder model aims to integrate into a neural search pipeline a model capable of building a latent representation of a document. The encoder model is particularly aimed at integrating the pre-trained models of [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html).

Here is the list of models provided by [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html). A more detailed version of this table and every details about models can be found on their website.

This list of models is not exhaustive, there are a wide range of models available with [Hugging Face](https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=downloads) and in many languages.

|                                          Model                                      |                      Avg. Performance                       |                      Speed                       |                      Model Size                       |
|:---------------------------------------------------------------------------------------:|:-----------------------------------------------------------:|:------------------------------------------------:|:-----------------------------------------------------:|
|                                    all-mpnet-base-v2                                    |                            63.30                            |                       2800                       |                         418 MB                        |
|                                    all-mpnet-base-v1                                    |                            62.34                            |                       2800                       |                         418 MB                        |
|                                multi-qa-mpnet-base-dot-v1                               |                            62.18                            |                       2800                       |                         418 MB                        |
|                                multi-qa-mpnet-base-cos-v1                               |                            61.88                            |                       2800                       |                         418 MB                        |
|                                   all-roberta-large-v1                                  |                            61.64                            |                        800                       |                        1355 MB                        |
|                                   all-distilroberta-v1                                  |                            59.84                            |                       4000                       |                         292 MB                        |
|                                    all-MiniLM-L12-v1                                    |                            59.80                            |                       7500                       |                         118 MB                        |
|                                    all-MiniLM-L12-v2                                    |                            59.76                            |                       7500                       |                         118 MB                        |
|                                multi-qa-distilbert-dot-v1                               |                            59.59                            |                       4000                       |                         253 MB                        |
|                                multi-qa-distilbert-cos-v1                               |                            59.41                            |                       4000                       |                         253 MB                        |
|                                     all-MiniLM-L6-v2                                    |                            58.80                            |                       14200                      |                         80 MB                         |
|                                multi-qa-MiniLM-L6-cos-v1                                |                            58.08                            |                       14200                      |                         80 MB                         |
|                                     all-MiniLM-L6-v1                                    |                            58.05                            |                       14200                      |                         80 MB                         |
|                                 paraphrase-mpnet-base-v2                                |                            57.70                            |                       2800                       |                         418 MB                        |
|                                 msmarco-bert-base-dot-v5                                |                            57.39                            |                       2800                       |                         418 MB                        |
|                                multi-qa-MiniLM-L6-dot-v1                                |                            56.55                            |                       14200                      |                         80 MB                         |
|                              msmarco-distilbert-base-tas-b                              |                            55.91                            |                       4000                       |                         253 MB                        |
|                                msmarco-distilbert-dot-v5                                |                            55.66                            |                       4000                       |                         253 MB                        |
|                             paraphrase-distilroberta-base-v2                            |                            54.69                            |                       4000                       |                         292 MB                        |
|                                 paraphrase-MiniLM-L12-v2                                |                            54.51                            |                       7500                       |                         118 MB                        |
|                          paraphrase-multilingual-mpnet-base-v2                          |                            53.75                            |                       2500                       |                         969 MB                        |
|                                paraphrase-TinyBERT-L6-v2                                |                            53.63                            |                       4500                       |                         238 MB                        |
|                                 paraphrase-MiniLM-L6-v2                                 |                            52.56                            |                       14200                      |                         80 MB                         |
|                                paraphrase-albert-small-v2                               |                            52.25                            |                       5000                       |                         43 MB                         |
|                          paraphrase-multilingual-MiniLM-L12-v2                          |                            51.72                            |                       7500                       |                         418 MB                        |
|                                 paraphrase-MiniLM-L3-v2                                 |                            50.74                            |                       19000                      |                         61 MB                         |
|                           distiluse-base-multilingual-cased-v1                          |                            45.59                            |                       4000                       |                         482 MB                        |
|                           distiluse-base-multilingual-cased-v2                          |                            43.77                            |                       4000                       |                         482 MB                        |
|                             average_word_embeddings_komninos                            |                            36.39                            |                       22000                      |                         237 MB                        |
|                          average_word_embeddings_glove.6B.300d                          |                            36.25                            |                       34000                      |                         422 MB                        |

Here's how to load a Bert Sentence template into the `search.Encoder` template:

```python
>>> from cherche import retrieve, rank
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {
...        "id": 0,
...        "article": "Paris is the capital and most populous city of France",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    },
...    {
...        "id": 1,
...        "article": "Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    },
...    {
...        "id": 2,
...        "article": "The City of Paris is the centre and seat of government of the region and province of Île-de-France .",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    }
... ]

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k=30)

>>> ranker = rank.Encoder(
...    key = "id",
...    on = "article",
...    encoder = SentenceTransformer(f"sentence-transformers/all-mpnet-base-v2").encode,
...    k = 2,
...    path = "encoder.pkl"
... )

>>> search = retriever + ranker 
>>> search.add(documents)
>>> search(q="france")
[{'id': 0, 'similarity': 0.44967225}, {'id': 2, 'similarity': 0.3609671}]
```

### Map index to documents Encoder

Optionally, you can map the documents to the identifiers proposed by the pipeline.

```python
>>> search += documents
>>> search(q="france")
[{'id': 0,
  'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.44967225},
 {'id': 2,
  'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.3609671}]
```

## rank.DPR

The [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) framework aims to train two distinct models, one that encodes the question and one that encodes the documents.

|               Question Encoder               |             Document encoder            |
|:--------------------------------------------:|:---------------------------------------:|
|  facebook-dpr-question_encoder-multiset-base |  facebook-dpr-ctx_encoder-multiset-base |
| facebook-dpr-question_encoder-single-nq-base | facebook-dpr-ctx_encoder-single-nq-base |

```python
>>> from cherche import rank
>>> from sentence_transformers import SentenceTransformer

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

>>> ranker = rank.DPR(
...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
...    on = "article",
...    k = 5,
...    path = "dpr.pkl"
... )
```

## rank.ZeroShot

Pre-trained models for the zero shot classification task are available from [Hugging Face](https://huggingface.co/models?pipeline_tag=zero-shot-classification). There are more than fifty models for different languages.

```python
>>> from cherche import rank
>>> from sentence_transformers import SentenceTransformer

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

>>> ranker = rank.ZeroShot(
...     encoder = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli"),
...     on = "article",
...     k = 5,
... )
```
