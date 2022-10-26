# qa

Onnxruntime for question answering models.



## Parameters

- **model**

- **name** (*'str'*)

- **providers** (*'list'*) – defaults to `['CPUExecutionProvider']`

- **quantize** (*'bool'*) – defaults to `True`



## Examples

```python
>>> from cherche import qa, onnx, retrieve
>>> from transformers import pipeline
>>> from pprint import pprint as print

>>> documents = [
...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> model = onnx.qa(
...    model = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2"),
...    name = "test",
...    quantize = False,
... )
ONNX opset version set to: 13
Loading pipeline (model: test, tokenizer: PreTrainedTokenizerFast(name_or_path='test', vocab_size=50265, model_max_len=512, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'eos_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'unk_token': AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'sep_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'pad_token': AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'cls_token': AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'mask_token': AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=True)}))
Creating folder test_onnx
Using framework PyTorch: 1.12.1
Found input input_ids with shape: {0: 'batch', 1: 'sequence'}
Found input attention_mask with shape: {0: 'batch', 1: 'sequence'}
Found output output_0 with shape: {0: 'batch', 1: 'sequence'}
Found output output_1 with shape: {0: 'batch', 1: 'sequence'}
Ensuring inputs are in correct order
token_type_ids is not present in the generated input list.
Generated inputs order: ['input_ids', 'attention_mask']

>>> model = qa.QA(
...     model = model,
...     on = ["title", "article"],
...     k = 2,
...  )

>>> model
Question Answering
    on: title, article

>>> print(model(q="Where is the Eiffel tower?", documents=documents))
[{'answer': 'Paris',
  'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'end': 20,
  'id': 1,
  'qa_score': 0.9940819,
  'start': 19,
  'title': 'Eiffel tower'},
 {'answer': 'Paris',
  'article': 'This town is the capital of France',
  'author': 'Wiki',
  'end': 9,
  'id': 0,
  'qa_score': 0.7963082,
  'start': 8,
  'title': 'Paris'}]
```

