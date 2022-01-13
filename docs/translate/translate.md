# Translation

The translation module is dedicated to the integration of Hugging Face's pre-trained translation models inside neural search pipelines. The list of pre-trained translation models is available on the website of [Hugging Face](https://huggingface.co/models?pipeline_tag=translation).

## Translate queries and documents

- `translate.Translate` is used to translate the fields of documents defined in the `on` parameter.
- `translate.TranslateQuery` is used to translate the input query of the summary in output of `summary.Summary`

Let's create a neural search pipeline for a French user. Let's start by translating the query from French to English, retrieve the relevant documents and finally translate the relevant documents from English to French.

```python
>>> from cherche import data, rank, retrieve, translate
>>> from transformers import pipeline
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

>>> retriever = retrieve.TfIdf(key="id", on="article", k=30, documents=documents)

>>> ranker = rank.Encoder(
...    key = "id",
...    on = ["title", "article"],
...    encoder = SentenceTransformer(f"sentence-transformers/all-mpnet-base-v2").encode,
...    k = 2,
...    path = "encoder.pkl"
... )

# Convert query from french to english
>>> query_translation = translate.TranslateQuery(
...     translation = pipeline("translation_fr_to_en", model = "Helsinki-NLP/opus-mt-fr-en"),
... )

# Convert documents from english to french
>>> document_translation = translate.Translate(
...     on = ["title", "article"],
...     translation = pipeline("translation_en_to_fr", model = "t5-small"),
... )

>>> search = query_translation + retriever + ranker + documents + document_translation
>>> search.add(documents)

# government and diplomacy
>>> search("gouvernement diplomatie")
[{'id': 1,
  'article': "Paris est l'un des grands centres européens de finance, de diplomatie , de commerce , de mode , de gastronomie , de science et d'art.",
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.12924933},
 {'id': 2,
  'article': "La ville de Paris est le centre et siège du gouvernement de la région et de la province de l'Île-de-France .",
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.12912892}]
```

## Translation of a summary

The `translate.TranslateQuery` model can be used to translate the summary output from the `summary.Summary` model.

```python
>>> from cherche import data, rank, retrieve, summary, translate
>>> from transformers import pipeline
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

>>> retriever = retrieve.TfIdf(key="id", on="article", k=30, documents=documents)

>>> ranker = rank.Encoder(
...    key = "id",
...    on = ["title", "article"],
...    encoder = SentenceTransformer(f"sentence-transformers/all-mpnet-base-v2").encode,
...    k = 2,
...    path = "encoder.pkl"
... )

# Convert query from french to english
>>> query_translation = translate.TranslateQuery(
...     translation = pipeline("translation_fr_to_en", model = "Helsinki-NLP/opus-mt-fr-en"),
... )

# Convert answer from english to french
>>> answer_translation = translate.TranslateQuery(
...     translation = pipeline("translation_en_to_fr", model = "t5-small"),
... )

>>> summarization = summary.Summary(
...    model = pipeline(
...         "summarization",
...         model="sshleifer/distilbart-cnn-6-6",
...         tokenizer="sshleifer/distilbart-cnn-6-6",
...         framework="pt"
...    ),
...    on = ["title", "article"],
... )

>>> search = query_translation + retriever + ranker + documents + summarization + answer_translation
>>> search.add(documents)

# government and diplomacy
>>> search("gouvernement diplomatie")
"Paris La ville de Paris est le centre et le siège du gouvernement de la région et de la province de l'Île-de-France."

# "Paris The city of Paris is the centre and seat of government of the region and province of Île-de-France."
```
