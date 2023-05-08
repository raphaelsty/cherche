__all__ = ["yield_batch", "yield_batch_single"]

import typing

import numpy as np
import tqdm


def yield_batch_single(
    array: typing.Union[
        typing.Union[typing.List[str], str],
        typing.List[typing.Dict[str, typing.Any]],
    ],
    desc: str,
):
    """Yield successive n-sized chunks from array."""
    if isinstance(array, str):
        yield array
    else:
        for batch in tqdm.tqdm(
            array,
            position=0,
            desc=desc,
            total=len(array),
        ):
            yield batch


def yield_batch(
    array: typing.Union[
        typing.Union[
            typing.Union[typing.List[str], str],
            typing.List[typing.Dict[str, typing.Any]],
        ],
        np.ndarray,
    ],
    batch_size: int,
    desc: str,
) -> typing.Generator:
    """Yield successive n-sized chunks from array."""
    if isinstance(array, str):
        yield [array]
    else:
        for batch in tqdm.tqdm(
            [array[pos : pos + batch_size] for pos in range(0, len(array), batch_size)],
            position=0,
            desc=desc,
            total=1 + len(array) // batch_size,
        ):
            yield batch
