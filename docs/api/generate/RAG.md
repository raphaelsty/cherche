# RAG

This model is dedicated to the generator of the paper Retrieval-Augmented Generation for Knowledge-Instensive NLP Tasks, Lewis et al, NeurIPS 2020. The RAG sequence to sequence model allows to generate the answer.



## Parameters

- **on** (*Union[str, list]*)

    Fields to summarize.

- **tokenizer**

    Hugging Face tokenizer for RAG. Number of documents to retrieve. Default is None, i.e all documents that match the query will be retrieved.

- **model**

    Hugging Face RAG model available [here](https://huggingface.co/docs/transformers/model_doc/rag).

- **k** (*int*) – defaults to `None`

- **num_beams** (*int*) – defaults to `10`

    Number of beams for beam search. 1 means no beam search.

- **min_length** (*int*) – defaults to `1`

    Minimum number of token of the generated answer.

- **max_length** (*int*) – defaults to `30`

    Maximum number of token of the generated answer.



## Examples

```python
>>> from pprint import pprint as print
>>> from transformers import RagTokenForGeneration, RagTokenizer
>>> from cherche import generate, retrieve

>>> documents = [
...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> retriever = retrieve.TfIdf(
...    key = "id", on = ["title", "article"], documents = documents, k = 2)

>>> generation = generate.RAG(
...     on = ["title", "article"],
...     tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq"),
...     model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=None),
...     k = 2,
...     num_beams = 2,
...     min_length = 1,
...     max_length = 10,
... )

>>> generation
RAG Generation
     on: ['title', 'article']
     k: 2
     num_beams: 2
     min_length: 1
     max_length: 10

>>> print(generation(q = "Eiffel Tower [SEP] town", documents = documents))
[{'answer': 'paris',
  'article': 'This town is the capital of France',
  'author': 'Wiki',
  'id': 0,
  'title': 'Paris'},
 {'answer': 'town of eiffel',
  'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'id': 1,
  'title': 'Eiffel tower'}]

>>> search = retriever + documents + generation

>>> print(search(q = "Eiffel Tower [SEP] town"))
[{'answer': 'paris',
  'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'id': 1,
  'similarity': 0.71251,
  'title': 'Eiffel tower'},
 {'answer': 'eiffel tower town',
  'article': 'This town is the capital of France',
  'author': 'Wiki',
  'id': 0,
  'similarity': 0.21936,
  'title': 'Paris'}]
```

## Methods

???- note "__call__"

    Main method to retrieve answer from input query.

    **Parameters**

    - **q**     (*str*)    
    - **documents**     (*list*)    
    - **kwargs**    
    
## References

1. [Lewis et al, Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, NeurIPS 2020](https://arxiv.org/pdf/2005.11401.pdf)
2. [Rag Hugging Face](https://huggingface.co/docs/transformers/model_doc/rag)
3. [Haystack RAG implementation](https://github.com/deepset-ai/haystack/blob/c6f23dce8897ab00fcb15e272282d459dcfa564a/haystack/nodes/answer_generator/transformers.py#L151)

