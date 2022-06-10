__all__ = ["sentence_transformers_onx", "OnxEncoder"]

import torch
import transformers
from sentence_transformers import SentenceTransformer, models


class OnnxEncoder:
    """OnxEncoder dedicated to run SentenceTransformer under OnnxRuntime."""

    def __init__(self, session, tokenizer, pooling, normalization):
        self.session = session
        self.tokenizer = tokenizer
        self.max_length = tokenizer.__dict__["model_max_length"]
        self.pooling = pooling
        self.normalization = normalization

    def encode(self, sentences: list):

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
        sentence_embedding = self.pooling.forward(
            features={
                "token_embeddings": torch.Tensor(hidden_state[0]),
                "attention_mask": torch.Tensor(inputs.get("attention_mask")),
            },
        )

        if self.normalization is not None:
            sentence_embedding = self.normalization.forward(features=sentence_embedding)

        sentence_embedding = sentence_embedding["sentence_embedding"]

        if sentence_embedding.shape[0] == 1:
            sentence_embedding = sentence_embedding[0]

        return sentence_embedding.numpy()


def sentence_transformers_onx(
    model,
    path,
    do_lower_case=True,
    input_names=["input_ids", "attention_mask", "segment_ids"],
    providers=["CPUExecutionProvider"],
):
    """OnxRuntime for sentence transformers.

    Parameters
    ----------
    model
        SentenceTransformer model.
    path
        Model file dedicated to session inference.
    do_lower_case
        Either or not the model is cased.
    input_names
        Fields needed by the Transformer.
    providers
        Either run the model on CPU or GPU: ["CPUExecutionProvider", "CUDAExecutionProvider"].

    Examples
    --------

    >>> from cherche import deploy, retrieve
    >>> from sentence_transformers import SentenceTransformer

    >>> documents = [
    ...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> model = sentence_transformers_onx(
    ...    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),
    ...    path = "onx",
    ... )

    >>> retriever = retrieve.Encoder(
    ...    encoder = model.encode,
    ...    key = "id",
    ...    on = ["title", "article"],
    ...    k = 2,
    ... )

    >>> retriever = retriever.add(documents)

    >>> retriever("paris")
    [{'id': 0, 'similarity': 1.4728151599072867}, {'id': 1, 'similarity': 1.029349867509033}]

    """

    try:
        import onnxruntime
    except:
        raise ValueError("You need to install onnxruntime.")

    model.save(path)

    configuration = transformers.AutoConfig.from_pretrained(
        path, from_tf=False, local_files_only=True
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        path, do_lower_case=do_lower_case, from_tf=False, local_files_only=True
    )

    encoder = transformers.AutoModel.from_pretrained(
        path, from_tf=False, config=configuration, local_files_only=True
    )

    st = ["cherche"]

    inputs = tokenizer(
        st,
        padding=True,
        truncation=True,
        max_length=tokenizer.__dict__["model_max_length"],
        return_tensors="pt",
    )

    model.eval()

    with torch.no_grad():

        symbolic_names = {0: "batch_size", 1: "max_seq_len"}

        torch.onnx.export(
            encoder,
            args=tuple(inputs.values()),
            f=f"{path}.onx",
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

        normalization = None
        for modules in model.modules():
            for idx, module in enumerate(modules):
                if idx == 1:
                    pooling = module
                if idx == 2:
                    normalization = module
            break

        return OnnxEncoder(
            session=onnxruntime.InferenceSession(f"{path}.onx", providers=providers),
            tokenizer=tokenizer,
            pooling=pooling,
            normalization=normalization,
        )
