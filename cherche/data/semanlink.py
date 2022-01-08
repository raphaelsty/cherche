import collections
import json
import pathlib

__all__ = ["arxiv_tags"]


def arxiv_tags(
    arxiv_title: bool = True,
    arxiv_summary: bool = True,
    comment: bool = False,
    broader_prefLabel_text: bool = True,
    broader_altLabel_text: bool = True,
    prefLabel_text: bool = True,
    altLabel_text: bool = True,
) -> tuple:
    """Semanlink tags arXiv documents. The objective of this dataset is to evaluate a neural
    search pipeline for automatic tagging of arXiv documents. This function returns the set of tags
    and the pairs arXiv documents and tags.

    Parameters
    ----------
    arxiv_title
        Include title of the arxiv paper inside the query.
    arxiv_summary
        Include summary of the arxiv paper inside the query.
    comment
        Include comment of the arxiv paper inside the query.
    broader_prefLabel_text
        Include broader_prefLabel as a text field.
    broader_altLabel_text
        Include broader_altLabel_text as a text field.
    prefLabel_text
        Include prefLabel_text as a text field.
    altLabel_text
        Include altLabel_text as a text field.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import data

    >>> documents, query_answers = data.arxiv_tags()

    >>> print(list(documents[0].keys()))
    ['prefLabel',
     'type',
     'broader',
     'creationTime',
     'creationDate',
     'comment',
     'uri',
     'broader_prefLabel',
     'broader_related']

    """
    with open(pathlib.Path(__file__).parent.joinpath("semanlink/arxiv.json"), "r") as input_file:
        docs = json.load(input_file)

    with open(pathlib.Path(__file__).parent.joinpath("semanlink/tags.json"), "r") as input_file:
        tags = json.load(input_file)

    # Filter arxiv tags
    counter = collections.defaultdict(int)

    query_answers = []
    for doc in docs:
        query = ""
        answers = []
        for field, include in [
            ("arxiv_title", arxiv_title),
            ("arxiv_summary", arxiv_summary),
            ("comment", comment),
        ]:
            if include:
                query = f"{query} {doc[field]}"

        for tag in doc["tag"]:
            answers.append({"uri": tags[tag]["uri"]})
            counter[tag] += 1

        query_answers.append((query, answers))

    # Filter arxiv tags
    documents = []
    for tag in counter:
        documents.append({key: value for key, value in tags[tag].items() if len(value) >= 1})

    for tag in documents:
        for field, include in [
            ("broader_prefLabel", broader_prefLabel_text),
            ("broader_altLabel", broader_altLabel_text),
            ("prefLabel", prefLabel_text),
            ("altLabel", altLabel_text),
        ]:
            if include and len(tag.get(field, "")) >= 1:
                tag[f"{field}_text"] = " ".join(tag[field])

    return documents, query_answers
