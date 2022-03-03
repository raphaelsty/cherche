# RAG Generator

RAG Generator allows using Hugging Face pre-trained sequence to sequence model as part of a neural search pipeline. In addition, these models generate a relevant response to a query by concatenating the query itself with relevant documents.

## Pre-trained

RAG is dedicated to the generator of the paper [Retrieval-Augmented Generation for Knowledge-Instensive NLP Tasks, Lewis et al, NeurIPS 2020](https://arxiv.org/abs/2005.11401).
RAG is also compatible with the generator of the paper [Robust Retrieval Augmented Generation for Zero-shot Slot Filling, Glass et al, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.148/).
The Github page of Glass et al 2021 is available [here](https://github.com/IBM/kgi-slot-filling).

|  Dataset |         Tokenizer        |                 Model                |
|:--------:|:------------------------:|:------------------------------------:|
| wiki_dpr | facebook/rag-sequence-nq |       facebook/rag-sequence-nq       |
| wiki_dpr |   facebook/rag-token-nq  |         facebook/rag-token-nq        |
|   T-REx  |   facebook/rag-token-nq  | michaelrglass/rag-token-nq-kgi0-trex |
|   zsRE   |   facebook/rag-token-nq  | michaelrglass/rag-token-nq-kgi0-zsre |

## Documents

When using `RAG` it is necessary to provide the model with the contents of the documents to generate an answer.

```python
# + documents allows to map documents to ids.
search = retriever + ranker + documents + generation
search = search.add(documents)
```

The `Elastic` retriever returns the contents of documents by default, so there is no need to add documents when using Elasticsearch.

## Example

```python
>>> from transformers import RagTokenForGeneration, RagTokenizer
>>> from cherche import generate, retrieve, rank

>>> documents = [
...    {"id": 0, "title": "Paris", "article": "Paris is the capital of France", "author": "Wiki"},
...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> retriever = retrieve.TfIdf(
...    key = "id", 
...    on = ["title", "article"], 
...    k = 10, 
...    documents = documents,
... )

>>> generation = generate.RAG(
...     on = ["title", "article"],
...     tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq"),
...     model = RagTokenForGeneration.from_pretrained("michaelrglass/rag-token-nq-kgi0-trex", retriever=None),
...     k = 2,
...     num_beams = 2,
...     min_length = 1,
...     max_length = 10,
... )

>>> search = retriever + documents + generation

>>> search(q = "Eiffel Tower [SEP] town")
[{'id': 1,
  'title': 'Eiffel tower',
  'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'similarity': 0.87264,
  'answer': 'Paris'}]
```

## GPU

We can use RAG with a GPU to speed up the generation of answers.

```python
>>> from transformers import RagTokenForGeneration, RagTokenizer
>>> from cherche import generate, retrieve, rank

>>> generation = generate.RAG(
...     on = ["title", "article"],
...     tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq"),
...     model = RagTokenForGeneration.from_pretrained("michaelrglass/rag-token-nq-kgi0-trex", retriever=None, device="cuda"),
...     k = 2,
...     num_beams = 2,
...     min_length = 1,
...     max_length = 10,
... )
```
