import weakref
from typing import Callable, Optional


class AlgoParam[T]:
    _registry: weakref.WeakSet["AlgoParam"] = weakref.WeakSet()

    def __init__(
        self,
        mutate: Callable[[T], T],
        value: T,
        name: Optional[str] = None,
        freeze: bool = False,
    ):
        self.mutate = mutate
        self._value = value
        self.name = name
        self.freeze = freeze
        AlgoParam._registry.add(self)

    @classmethod
    def parameters(cls):
        yield from cls._registry

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: T):
        if not self.freeze:
            self._value = value

    @property
    def next_value(self):
        self.value = self.mutate(self.value)
        return self.value

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, value={self.value}, \
                mutate_func={getattr(self.mutate, '__name__', None)})"


if __name__ == "__main__":

    def ff(x):
        return x + 1

    a = AlgoParam[int](ff, name="a", value=1)
    c = AlgoParam[int](lambda x: x * 2, value=2, name="c")

    def f():
        AlgoParam[str](lambda x: x * 2, value="hello", name="b")

    f()

    for item in AlgoParam.parameters():
        print(item)
