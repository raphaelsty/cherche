# CrossEncoder

Cross-Encoder as a ranker. CrossEncoder takes both the query and the document as input and outputs a score. The score is a similarity score between the query and the document. The CrossEncoder cannot pre-compute the embeddings of the documents since it need both the query and the document.



## Parameters

- **on** (*Union[List[str], str]*)

    Fields to use to match the query to the documents.

- **encoder**

    Sentence Transformer cross-encoder.

- **k** (*Optional[int]*) – defaults to `None`

- **batch_size** (*int*) – defaults to `64`



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve, rank, evaluate, data
>>> from sentence_transformers import CrossEncoder

>>> documents, query_answers = data.arxiv_tags(
...    arxiv_title=True, arxiv_summary=False, comment=False
... )

>>> retriever = retrieve.TfIdf(
...    key="uri",
...    on=["prefLabel_text", "altLabel_text"],
...    documents=documents,
...    k=100,
... )

>>> ranker = rank.CrossEncoder(
...     on = ["prefLabel_text", "altLabel_text"],
...     encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1").predict,
... )

>>> pipeline = retriever + documents + ranker

>>> match = pipeline("graph neural network", k=5)

>>> for m in match:
...     print(m.get("uri", ""))
'http://www.semanlink.net/tag/graph_neural_networks'
'http://www.semanlink.net/tag/artificial_neural_network'
'http://www.semanlink.net/tag/dans_deep_averaging_neural_networks'
'http://www.semanlink.net/tag/recurrent_neural_network'
'http://www.semanlink.net/tag/convolutional_neural_network'
```

## Methods

???- note "__call__"

    Rank inputs documents based on query.

    **Parameters**

    - **q**     (*str*)    
    - **documents**     (*list*)    
    - **batch_size**     (*Optional[int]*)     – defaults to `None`    
    - **k**     (*Optional[int]*)     – defaults to `None`    
    - **kwargs**    
    
## References

1. [Sentence Transformers Cross-Encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html)
2. [Cross-Encoders Hub](https://huggingface.co/cross-encoder)

