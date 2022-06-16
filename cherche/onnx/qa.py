from __future__ import annotations

__all__ = ["qa"]


import os
import pathlib

import numpy as np
import transformers
from onnxruntime import InferenceSession, quantization
from scipy import special
from transformers import convert_graph_to_onnx

from .base import clean_folder


class QAEncoder:
    """QAEncoder dedicated to run question answering models under onnx runtime."""

    def __init__(self, session, tokenizer) -> None:
        self.session = session
        self.tokenizer = tokenizer
        self.special_tokens = [
            tokenizer.encode(token, add_special_tokens=False)[0]
            for token in self.tokenizer.special_tokens_map.values()
        ]

    def __call__(self, question: str | list, context: str | list, **kwargs) -> list:
        """Question answering dedicated to onnx. The number of question must be the same as the
        number of documents as context.

        Parameters
        ----------
        question
            Questions asked to the model.
        context
            Documents to extract the answer.

        """
        question, context, outputs = (
            [question] if isinstance(question, str) else question,
            [context] if isinstance(context, str) else context,
            [],
        )

        for q, c in zip(question, context):

            inputs = self.tokenizer(
                q, c, add_special_tokens=True, truncation="only_second", return_tensors="np"
            )

            start, end = self.session.run(input_feed=dict(inputs), output_names=None)
            exclude = ~np.isin(inputs["input_ids"], self.special_tokens)

            start, end = start[exclude], end[exclude]
            start, end = special.softmax(start, axis=0), special.softmax(end, axis=0)
            span_start, span_end = np.argmax(start), np.argmax(end)

            inputs = inputs["input_ids"][exclude].tolist()

            outputs.append(
                {
                    "start": span_start,
                    "end": span_end + 1,
                    "score": start[span_start] * end[span_end],
                    "answer": self.tokenizer.decode(inputs[span_start : span_end + 1]).strip(),
                }
            )

        return outputs


def qa(
    model,
    name: str,
    providers: list = ["CPUExecutionProvider"],
    quantize: bool = True,
) -> QAEncoder:
    """Onnxruntime for question answering models.

    Examples
    --------

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
    Using framework PyTorch: 1.12.0.dev20220519
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
        model: test
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

    """
    if os.path.exists(name):
        raise ValueError("Invalid name, {name} exists.")

    model.save_pretrained(name)

    config = transformers.AutoConfig.from_pretrained(name, from_tf=False, local_files_only=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        name, from_tf=False, local_files_only=True, config=config
    )

    folder = f"{name}_onnx"

    path = os.path.join(folder, folder)

    convert_graph_to_onnx.convert(
        model=name,
        framework="pt",
        pipeline_name="question-answering",
        opset=13,
        output=pathlib.Path(path),
        tokenizer=tokenizer,
    )

    if quantize:

        quantization.quantize_dynamic(
            path, f"{path}.qonx", weight_type=quantization.QuantType.QInt8
        )

        path = f"{path}.qonx"

    encoder = QAEncoder(
        session=InferenceSession(path, providers=providers),
        tokenizer=tokenizer,
    )

    clean_folder(name)
    clean_folder(folder)

    return encoder
