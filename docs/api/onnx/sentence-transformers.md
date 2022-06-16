# sentence_transformers

OnxRuntime for sentence transformers. The `sentence_transformers` function converts sentence transformers to the onnx format.



## Parameters

- **model**

    SentenceTransformer model.

- **name** (*'str'*)

    Model file dedicated to session inference.

- **input_names** (*'list'*) – defaults to `['input_ids', 'attention_mask', 'segment_ids']`

    Fields needed by the Transformer.

- **providers** (*'list'*) – defaults to `['CPUExecutionProvider']`

    A provider from the list: ["CUDAExecutionProvider", "CPUExecutionProvider", "TensorrtExecutionProvider", "DnnlExecutionProvider"].

- **quantize** (*'bool'*) – defaults to `True`



## Examples

```python
>>> from cherche import onnx, retrieve
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> model = onnx.sentence_transformers(
...    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),
...    name = "test",
...    quantize = True,
... )

>>> retriever = retrieve.Encoder(
...    encoder = model.encode,
...    key = "id",
...    on = ["title", "article"],
...    k = 2,
... )

>>> retriever = retrieve.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    key = "id",
...    on = ["title", "article"],
...    k = 2,
... )

>>> retriever = retriever.add(documents)

>>> retriever("paris")
[{'id': 0, 'similarity': 1.4728152892007695}, {'id': 1, 'similarity': 1.0293501832829597}]
```

## References

1. [Onnx installation](https://github.com/onnx/onnx/issues/3129)

