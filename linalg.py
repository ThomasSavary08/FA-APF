# Libraries
import jax  # type: ignore

from jax import Array  # type: ignore
from typing import Callable


def transpose(A: Callable[[Array], Array], x: Array) -> Callable[[Array], Array]:
    """
    Transpose of a linear operator
    Input(s)
        - A (Callable[[Array], Array]): the linear operator
        - x (Array): array used for the computation
    """
    _, vjp = jax.vjp(A, x)

    def At(y):
        return next(iter(vjp(y)))

    return At
