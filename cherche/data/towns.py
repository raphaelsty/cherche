import json
import pathlib

__all__ = ["load_towns"]


def load_towns():
    """Sample of Wikipedia dataset that contains informations about Toulouse, Paris, Lyon and
    Bordeaux.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import data

    >>> towns = data.load_towns()

    >>> print(towns[:3])
    [{'article': 'Paris (French pronunciation: \u200b[paʁi] (listen)) is the '
                 'capital and most populous city of France, with an estimated '
                 'population of 2,175,601 residents as of 2018, in an area of more '
                 'than 105 square kilometres (41 square miles).',
      'id': 0,
      'title': 'Paris',
      'url': 'https://en.wikipedia.org/wiki/Paris'},
     {'article': "Since the 17th century, Paris has been one of Europe's major "
                 'centres of finance, diplomacy, commerce, fashion, gastronomy, '
                 'science, and arts.',
      'id': 1,
      'title': 'Paris',
      'url': 'https://en.wikipedia.org/wiki/Paris'},
     {'article': 'The City of Paris is the centre and seat of government of the '
                 'region and province of Île-de-France, or Paris Region, which has '
                 'an estimated population of 12,174,880, or about 18 percent of '
                 'the population of France as of 2017.',
      'id': 2,
      'title': 'Paris',
      'url': 'https://en.wikipedia.org/wiki/Paris'}]


    """
    with open(pathlib.Path(__file__).parent.joinpath("towns.json"), "r") as towns_json:
        return json.load(towns_json)
