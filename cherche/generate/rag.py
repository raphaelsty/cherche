__all__ = ["RAG"]

import typing

import torch
from scipy.special import softmax

from .base import Generation


class RAG(Generation):
    """This model is dedicated to the generator of the paper Retrieval-Augmented Generation for
    Knowledge-Instensive NLP Tasks, Lewis et al, NeurIPS 2020. The RAG sequence to sequence model
    allows to generate the answer.

    Parameters
    ----------
    on
        Fields to summarize.
    model
        Hugging Face RAG model available [here](https://huggingface.co/docs/transformers/model_doc/rag).
    tokenizer
        Hugging Face tokenizer for RAG.
    k
        Number of answers to generate.
    num_beams
        Number of beams for beam search. 1 means no beam search.
    min_length
        Minimum number of token of the generated answer.
    max_length
        Maximum number of token of the generated answer.

    Examples
    --------

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

    References
    ----------
    1. [Lewis et al, Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, NeurIPS 2020](https://arxiv.org/pdf/2005.11401.pdf)
    2. [Rag Hugging Face](https://huggingface.co/docs/transformers/model_doc/rag)
    3. [Haystack RAG implementation](https://github.com/deepset-ai/haystack/blob/c6f23dce8897ab00fcb15e272282d459dcfa564a/haystack/nodes/answer_generator/transformers.py#L151)

    """

    def __init__(
        self,
        on: typing.Union[str, list],
        tokenizer,
        model,
        k: int = None,
        num_beams: int = 10,
        min_length: int = 1,
        max_length: int = 30,
    ) -> None:

        super().__init__(
            on=on,
            tokenizer=tokenizer,
            model=model,
            k=k,
            num_beams=num_beams,
            min_length=min_length,
            max_length=max_length,
        )

    def __repr__(self) -> str:
        repr = "RAG Generation"
        repr += f"\n\t on: {self.on}"
        repr += f"\n\t k: {self.k}"
        repr += f"\n\t num_beams: {self.num_beams}"
        repr += f"\n\t min_length: {self.min_length}"
        repr += f"\n\t max_length: {self.max_length}"
        return repr

    def __call__(self, q: str, documents: list, **kwargs) -> list:
        """Main method to retrieve answer from input query.

        Parameters
        ----------
        q
            Query.
        documents
            Documents which contains the answer of the query.

        """
        if not documents:
            return []

        q_documents = [
            f"{' '.join([document.get(field, '') for field in self.on]).strip()} {self.model.config.doc_sep} {q}"
            for document in documents
        ]

        doc_scores = torch.tensor(
            [softmax([document.get("similarity", 1.0) for document in documents], axis=0)],
            dtype=torch.float,
        )

        with torch.no_grad():

            context = self.tokenizer.generator.batch_encode_plus(
                q_documents,
                max_length=self.model.config.max_combined_length,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )

            # Get generated ids from generator
            generator_ids = self.model.generate(
                context_input_ids=context["input_ids"],
                context_attention_mask=context["attention_mask"],
                doc_scores=doc_scores,
                num_return_sequences=min(self.k, len(documents)),
                num_beams=min(self.num_beams, self.k, len(documents)),
                max_length=self.max_length,
                min_length=self.min_length,
                n_docs=len(documents),
            )

            return [
                {**document, **{"answer": answer.strip()}}
                for document, answer in zip(
                    documents[: self.k] if self.k is not None else documents,
                    self.tokenizer.batch_decode(generator_ids, skip_special_tokens=True),
                )
            ]
