from __future__ import annotations

__all__ = ["sentence_transformers", "STEncoder"]

import os
import typing

import torch
import transformers
from onnxruntime import InferenceSession, quantization

from .base import clean_folder


def sentence_transformers(
    model,
    name: str,
    input_names: list = ["input_ids", "attention_mask", "segment_ids"],
    providers: list = ["CPUExecutionProvider"],
    quantize: bool = True,
):
    """OnxRuntime for sentence transformers. The `sentence_transformers` function converts sentence
    transformers to the onnx format.

    Parameters
    ----------
    model
        SentenceTransformer model.
    name
        Model file dedicated to session inference.
    do_lower_case
        Either or not, the model should be uncased.
    input_names
        Fields needed by the Transformer.
    providers
        A provider from the list: ["CUDAExecutionProvider", "CPUExecutionProvider",
        "TensorrtExecutionProvider", "DnnlExecutionProvider"].

    Examples
    --------

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

    References
    ----------
    1. [Onnx installation](https://github.com/onnx/onnx/issues/3129)

    """
    if os.path.exists(name):
        raise ValueError(f"Invalid name, {name} exists.")

    model.save(name)

    configuration = transformers.AutoConfig.from_pretrained(
        name, from_tf=False, local_files_only=True
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        name, from_tf=False, local_files_only=True
    )

    encoder = transformers.AutoModel.from_pretrained(
        name, from_tf=False, config=configuration, local_files_only=True
    )

    clean_folder(name)

    st = ["cherche"]

    inputs = tokenizer(
        st,
        padding=True,
        truncation=True,
        max_length=tokenizer.__dict__["model_max_length"],
        return_tensors="pt",
    )

    model.eval()

    folder = f"{name}_onnx"
    os.mkdir(folder)

    with torch.no_grad():

        symbolic_names = {0: "batch_size", 1: "max_seq_len"}

        torch.onnx.export(
            encoder,
            args=tuple(inputs.values()),
            f=os.path.join(folder, f"{name}.onx"),
            opset_version=13,  # ONX version needs to be >= 13 for sentence transformers.
            do_constant_folding=True,
            input_names=input_names,
            output_names=["start", "end"],
            dynamic_axes={
                "input_ids": symbolic_names,
                "attention_mask": symbolic_names,
                "segment_ids": symbolic_names,
                "start": symbolic_names,
                "end": symbolic_names,
            },
        )

    layers = []
    for modules in model.modules():
        for idx, module in enumerate(modules):
            # Skip the transformer that run under onnx.
            if idx == 0:
                continue
            layers.append(module)
        break

    if quantize:
        quantization.quantize_dynamic(
            os.path.join(folder, f"{name}.onx"),
            os.path.join(folder, f"{name}.qonx"),
            weight_type=quantization.QuantType.QInt8,
        )
        name = os.path.join(folder, f"{name}.qonx")
    else:
        name = os.path.join(folder, f"{name}.onx")

    encoder = STEncoder(
        session=InferenceSession(name, providers=providers),
        tokenizer=tokenizer,
        layers=layers,
    )

    clean_folder(folder)
    return encoder


class STEncoder:
    """Encoder dedicated to run Sentence Transformer models with Onnxruntime.

    Parameters
    ----------
    session
        Onnxruntime inference session.
    tokenizer
        Transformer dedicated tokenizer.
    pooling
        Poolling layers of the Transformer.
    normalization
        Normalization layers of the Transformer.

    """

    def __init__(self, session, tokenizer, layers: list, max_length: typing.Optional[int] = None):
        self.session = session
        self.tokenizer = tokenizer
        self.layers = layers
        self.max_length = (
            max_length if max_length is not None else tokenizer.__dict__["model_max_length"]
        )

    def encode(self, sentences: str | list):
        """Sentence transformer encoding function.

        Parameters
        ----------
        sentences
            Either a sentence or a list of sentences to encode.

        """
        sentences = [sentences] if isinstance(sentences, str) else sentences

        inputs = {
            k: v.numpy()
            for k, v in self.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).items()
        }

        hidden_state = self.session.run(None, inputs)

        features = {
            "token_embeddings": torch.Tensor(hidden_state[0]),
            "attention_mask": torch.Tensor(inputs.get("attention_mask")),
        }

        for layer in self.layers:
            features = layer.forward(
                features=features,
            )

        # Sentence transformers API format:
        sentence = features["sentence_embedding"]
        if sentence.shape[0] == 1:
            sentence = sentence[0]

        return sentence.numpy()
