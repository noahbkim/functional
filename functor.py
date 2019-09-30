from __future__ import annotations

import abc
from typing import *
from dataclasses import dataclass

T = TypeVar("T")
C = TypeVar("C", bound=type)
D = TypeVar("D", bound=type)
X = TypeVar("X")
Y = TypeVar("Y")
Z = TypeVar("Z")


class Morphism(Generic[X, Y], abc.ABC):
    """A mapping from one object in T to another."""

    def __call__(self, e: X) -> Y:
        """This is the functional implementation of m(x)."""


class Identity(Morphism):
    """A generic identity mapping we'll use for type hinting."""


def identity(x: X) -> X:
    return x


def compose(f: Morphism[X, Y], g: Morphism[Y, Z]) -> Morphism[X, Z]:
    return lambda x: f(g(x))


class FunctorCast(Generic[C, D], abc.ABC):
    """Represents a typecast for mapping x to F(x)."""

    def __getitem__(self, x: C) -> D:
        """Should yield F(x) in D."""


class Functor(Generic[C, D], abc.ABC):
    """A relationship between elements and morphisms in categories."""

    Cast: FunctorCast[C, D]

    @classmethod
    @abc.abstractmethod
    def lift(cls, f: Morphism[X, Y]) -> Morphism[Cast[X], Cast[Y]]:
        """Associate morphisms across the categories."""

    @classmethod
    def map(cls, f: Morphism[X, Y], a: Cast[X]) -> Cast[Y]:
        """Actually apply the lifted morphism on a casted X."""

        return cls.lift(f)(a)


@dataclass
class Maybe(Generic[T]):
    value: Optional[T] = None


class MaybeFunctor(Functor):
    """A functor that operates on maybes."""

    Cast = Maybe

    @classmethod
    def lift(cls, f: Morphism[X, Y]) -> Morphism[Cast[X], Cast[Y]]:
        return lambda a: Maybe(f(a.value)) if a.value is not None else a


class IterableFunctor(Functor):
    """Formalize how map() and iter() are functors."""

    Cast = Iterable

    @classmethod
    def lift(cls, f: Morphism[Morphism[X, Y]]) -> Morphism[Cast[X], Cast[Y]]:
        return lambda a: map(f, a)


def test():
    assert MaybeFunctor.Cast[int] == Maybe[int]

    f: Morphism[int, int] = lambda x: x + 1
    assert MaybeFunctor.map(f, Maybe(1)) == Maybe(2)
    assert MaybeFunctor.map(f, Maybe(None)) == Maybe(None)

    # F(id_x) = id_Fx
    identity_int: Identity[int] = identity
    identity_maybe_int: Identity[List[int]] = identity
    assert MaybeFunctor.map(identity_int, Maybe(1)) == identity_maybe_int(Maybe(1))

    # F(f * g) = F(f) * F(g)
    mf = MaybeFunctor
    g: Morphism[int, int] = lambda x: x * 2
    assert mf.map(compose(f, g), Maybe(1)) == compose(mf.lift(f), mf.lift(g))(Maybe(1))
    assert mf.map(compose(f, g), Maybe(None)) == compose(mf.lift(f), mf.lift(g))(Maybe(None))

    assert list(map(f, range(10))) == list(IterableFunctor.map(f, range(10)))


if __name__ == "__main__":
    test()
