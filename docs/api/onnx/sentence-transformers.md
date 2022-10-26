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
Ignore MatMul due to non constant B: /[MatMul_229]
Ignore MatMul due to non constant B: /[MatMul_235]
Ignore MatMul due to non constant B: /[MatMul_332]
Ignore MatMul due to non constant B: /[MatMul_338]
Ignore MatMul due to non constant B: /[MatMul_435]
Ignore MatMul due to non constant B: /[MatMul_441]
Ignore MatMul due to non constant B: /[MatMul_538]
Ignore MatMul due to non constant B: /[MatMul_544]
Ignore MatMul due to non constant B: /[MatMul_641]
Ignore MatMul due to non constant B: /[MatMul_647]
Ignore MatMul due to non constant B: /[MatMul_744]
Ignore MatMul due to non constant B: /[MatMul_750]
Ignore MatMul due to non constant B: /[MatMul_847]
Ignore MatMul due to non constant B: /[MatMul_853]
Ignore MatMul due to non constant B: /[MatMul_950]
Ignore MatMul due to non constant B: /[MatMul_956]
Ignore MatMul due to non constant B: /[MatMul_1053]
Ignore MatMul due to non constant B: /[MatMul_1059]
Ignore MatMul due to non constant B: /[MatMul_1156]
Ignore MatMul due to non constant B: /[MatMul_1162]
Ignore MatMul due to non constant B: /[MatMul_1259]
Ignore MatMul due to non constant B: /[MatMul_1265]
Ignore MatMul due to non constant B: /[MatMul_1362]
Ignore MatMul due to non constant B: /[MatMul_1368]

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

